import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


def index_reverse(index: torch.Tensor) -> torch.Tensor:
    """Compute the inverse permutation per batch.

    Args:
        index: Long tensor of shape [B, N] containing a per-row permutation.

    Returns:
        Long tensor of shape [B, N] where out[b, index[b, i]] == i.
    """
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1], device=index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def semantic_neighbor(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Reorder features along the token dimension using the provided index.

    This is the SGN "unfold/fold" operation used by ASSM.
    """
    dim = index.dim()
    assert x.shape[:dim] == index.shape, (
        "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)
    )

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x


class ASSM(nn.Module):
    """Attentive State Space Module (ASSM).

    Key steps:
    - Route each token to a dictionary entry via Gumbel-Softmax.
    - Build a prompt from selected token embeddings.
    - Apply SGN-unfold -> Selective_Scan -> SGN-fold.
    """

    def __init__(
        self,
        dim: int,
        d_state: int,
        input_resolution: Tuple[int, int],
        num_tokens: int = 64,
        inner_rank: int = 128,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        # Mamba params
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )

        # Channel-wise positional encoding (depthwise conv).
        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )

        # Token dictionary for routing.
        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        # Route token selection logits.
        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x: torch.Tensor, x_size: Tuple[int, int], token: nn.Embedding) -> torch.Tensor:
        B, n, C = x.shape
        H, W = x_size

        # Compose dictionary embeddings: [num_tokens, inner_rank] @ [inner_rank, d_state].
        full_embedding = self.embeddingB.weight @ token.weight  # [num_tokens, d_state]

        # Route each token to a dictionary entry and build prompt.
        pred_route = self.route(x)  # [B, HW, num_tokens]
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)
        prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)

        # Sort by routing decisions to build semantic neighborhoods.
        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)
        _, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        # Project to hidden channels and apply CPE.
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)

        # SGN-unfold -> selective scan -> SGN-fold.
        semantic_x = semantic_neighbor(x, x_sort_indices)
        y = self.selectiveScan(semantic_x, prompt)
        y = self.out_proj(self.out_norm(y))
        x = semantic_neighbor(y, x_sort_indices_reverse)

        return x


class Selective_Scan(nn.Module):
    """Selective scan block used by MambaIRv2.

    This is a lightly wrapped copy of the Mamba selective scan module with
    prompt injection for ASSM.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: float = 2.0,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(
        dt_rank: int,
        d_inner: int,
        dt_scale: float = 1.0,
        dt_init: str = "random",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        **factory_kwargs,
    ) -> nn.Linear:
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Preserve variance at initialization.
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize bias so softplus(dt_bias) is in [dt_min, dt_max].
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state: int, d_inner: int, copies: int = 1, device=None, merge: bool = True) -> nn.Parameter:
        # S4D real initialization.
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner: int, copies: int = 1, device=None, merge: bool = True) -> nn.Parameter:
        # D "skip" parameter.
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        K = 1  # MambaIRv2 uses a single scan.
        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)

        # Prompt injection (ASE).
        Cs = Cs.float().view(B, K, -1, L) + prompt
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, prompt: torch.Tensor, **kwargs) -> torch.Tensor:
        b, l, c = prompt.shape
        prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        y = self.forward_core(x, prompt)
        y = y.permute(0, 2, 1).contiguous()
        return y


class ASSM2DBlock(nn.Module):
    def __init__(self, cfg, inchannels):
        super().__init__()
        model_cfg = cfg.get("model_cfg", {})
        d_state = model_cfg.get("d_state", 16)
        num_tokens = model_cfg.get("assm2d_num_tokens", model_cfg.get("assm_num_tokens", 32))
        inner_rank = model_cfg.get("assm2d_inner_rank", model_cfg.get("assm_inner_rank", 64))
        mlp_ratio = model_cfg.get("assm2d_mlp_ratio", model_cfg.get("assm_mlp_ratio", 2.0))
        self.hid_feature = inchannels
        self.assm = ASSM(
            dim=inchannels,
            d_state=d_state,
            input_resolution=(1, 1),
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            mlp_ratio=mlp_ratio,
        )
        self.token = nn.Embedding(inner_rank, d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        b, c, t, f = x.shape
        assert c == self.hid_feature

        residual = x
        x_seq = x.permute(0, 2, 3, 1).contiguous().view(b, t * f, c)
        y_seq = self.assm(x_seq, x_size=(t, f), token=self.token)
        y = y_seq.view(b, t, f, c).permute(0, 3, 1, 2).contiguous()

        assert y.shape == residual.shape
        if not torch.isfinite(y).all():
            raise RuntimeError("ASSM2DBlock output contains NaN/Inf")

        out = residual + y
        return out


class TASSMBlock(nn.Module):
    def __init__(self, cfg, inchannels):
        super().__init__()
        model_cfg = cfg.get("model_cfg", {})
        d_state = model_cfg.get("d_state", 16)
        num_tokens = model_cfg.get("tassm_num_tokens", model_cfg.get("assm_num_tokens", 32))
        inner_rank = model_cfg.get("tassm_inner_rank", model_cfg.get("assm_inner_rank", 64))
        mlp_ratio = model_cfg.get("tassm_mlp_ratio", model_cfg.get("assm_mlp_ratio", 2.0))
        self.hid_feature = inchannels
        self.assm = ASSM(
            dim=inchannels,
            d_state=d_state,
            input_resolution=(1, 1),
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            mlp_ratio=mlp_ratio,
        )
        self.token = nn.Embedding(inner_rank, d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        b, c, t, f = x.shape
        assert c == self.hid_feature

        residual = x
        x_seq = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        y_seq = self.assm(x_seq, x_size=(t, 1), token=self.token)
        y = y_seq.view(b, f, t, c).permute(0, 3, 2, 1).contiguous()

        assert y.shape == residual.shape
        if not torch.isfinite(y).all():
            raise RuntimeError("TASSMBlock output contains NaN/Inf")

        out = residual + y
        return out


class FASSMBlock(nn.Module):
    def __init__(self, cfg, inchannels):
        super().__init__()
        model_cfg = cfg.get("model_cfg", {})
        d_state = model_cfg.get("d_state", 16)
        num_tokens = model_cfg.get("assm_num_tokens", 32)
        inner_rank = model_cfg.get("assm_inner_rank", 64)
        mlp_ratio = model_cfg.get("assm_mlp_ratio", 2.0)
        self.hid_feature = inchannels
        self.assm = ASSM(
            dim=inchannels,
            d_state=d_state,
            input_resolution=(1, 1),
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            mlp_ratio=mlp_ratio,
        )
        self.token = nn.Embedding(inner_rank, d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        b, c, t, f = x.shape
        assert c == self.hid_feature

        residual = x
        x_seq = x.permute(0, 2, 3, 1).contiguous().view(b * t, f, c)
        y_seq = self.assm(x_seq, x_size=(1, f), token=self.token)
        y = y_seq.view(b, t, f, c).permute(0, 3, 1, 2).contiguous()

        assert y.shape == residual.shape
        if not torch.isfinite(y).all():
            raise RuntimeError("FASSMBlock output contains NaN/Inf")

        out = residual + y
        return out


class TFASSMBlock(nn.Module):
    def __init__(self, cfg, inchannels):
        super().__init__()
        model_cfg = cfg.get("model_cfg", {})
        d_state = model_cfg.get("d_state", 16)

        t_num_tokens = model_cfg.get("tassm_num_tokens", model_cfg.get("assm_num_tokens", 32))
        t_inner_rank = model_cfg.get("tassm_inner_rank", model_cfg.get("assm_inner_rank", 64))
        t_mlp_ratio = model_cfg.get("tassm_mlp_ratio", model_cfg.get("assm_mlp_ratio", 2.0))

        f_num_tokens = model_cfg.get("assm_num_tokens", 32)
        f_inner_rank = model_cfg.get("assm_inner_rank", 64)
        f_mlp_ratio = model_cfg.get("assm_mlp_ratio", 2.0)

        self.hid_feature = inchannels
        self.time_assm = ASSM(
            dim=inchannels,
            d_state=d_state,
            input_resolution=(1, 1),
            num_tokens=t_num_tokens,
            inner_rank=t_inner_rank,
            mlp_ratio=t_mlp_ratio,
        )
        self.time_token = nn.Embedding(t_inner_rank, d_state)

        self.freq_assm = ASSM(
            dim=inchannels,
            d_state=d_state,
            input_resolution=(1, 1),
            num_tokens=f_num_tokens,
            inner_rank=f_inner_rank,
            mlp_ratio=f_mlp_ratio,
        )
        self.freq_token = nn.Embedding(f_inner_rank, d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, f = x.size()
        assert c == self.hid_feature

        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.time_assm(x, x_size=(t, 1), token=self.time_token) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.freq_assm(x, x_size=(1, f), token=self.freq_token) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


if __name__ == "__main__":
    # Minimal shape check for ASSM wiring.
    B, H, W, C = 1, 8, 8, 32
    d_state = 8
    inner_rank = 16
    num_tokens = 32

    x = torch.randn(B, H * W, C)
    token = nn.Embedding(inner_rank, d_state)

    assm = ASSM(
        dim=C,
        d_state=d_state,
        input_resolution=(H, W),
        num_tokens=num_tokens,
        inner_rank=inner_rank,
        mlp_ratio=2.0,
    )

    y = assm(x, (H, W), token)
    print("ASSM output shape:", y.shape)
