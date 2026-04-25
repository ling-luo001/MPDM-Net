import torch
import torch.nn as nn
from einops import rearrange
from .lsigmoid import LearnableSigmoid2D
import typing as tp
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

# from .transformer import get_mask, MultiheadAttention, MyGroupNorm, LayerScale


def get_padding(kernel_size, dilation=1):
    """
    Calculate the padding size for a convolutional layer.
    
    Args:
    - kernel_size (int): Size of the convolutional kernel.
    - dilation (int, optional): Dilation rate of the convolution. Defaults to 1.
    
    Returns:
    - int: Calculated padding size.
    """
    return int((kernel_size * dilation - dilation) / 2)

def get_padding_2d(kernel_size, dilation=(1, 1)):
    """
    Calculate the padding size for a 2D convolutional layer.
    
    Args:
    - kernel_size (tuple): Size of the convolutional kernel (height, width).
    - dilation (tuple, optional): Dilation rate of the convolution (height, width). Defaults to (1, 1).
    
    Returns:
    - tuple: Calculated padding size (height, width).
    """
    return (int((kernel_size[0] * dilation[0] - dilation[0]) / 2), 
            int((kernel_size[1] * dilation[1] - dilation[1]) / 2))

class DenseBlock(nn.Module):
    """
    DenseBlock module consisting of multiple convolutional layers with dilation.
    """
    def __init__(self, cfg, kernel_size=(3, 3), depth=4):
        super(DenseBlock, self).__init__()
        self.cfg = cfg
        self.depth = depth
        self.dense_block = nn.ModuleList()
        self.hid_feature = cfg['model_cfg']['hid_feature']

        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(self.hid_feature * (i + 1), self.hid_feature, kernel_size, 
                          dilation=(dil, 1), padding=get_padding_2d(kernel_size, (dil, 1))),
                nn.InstanceNorm2d(self.hid_feature, affine=True),
                nn.PReLU(self.hid_feature)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        """
        Forward pass for the DenseBlock module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Output tensor after processing through the dense block.
        """
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x

class DenseEncoder(nn.Module):
    """
    DenseEncoder module consisting of initial convolution, dense block, and a final convolution.
    """
    def __init__(self, cfg):
        super(DenseEncoder, self).__init__()
        self.cfg = cfg
        self.input_channel = cfg['model_cfg']['input_channel']
        self.hid_feature = cfg['model_cfg']['hid_feature']

        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(self.input_channel, self.hid_feature, (1, 1)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature)
        )

        self.dense_block = DenseBlock(cfg, depth=4)

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature)
        )

    def forward(self, x):
        """
        Forward pass for the DenseEncoder module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Encoded tensor.
        """
        x = self.dense_conv_1(x)  # [batch, hid_feature, time, freq]
        x = self.dense_block(x)   # [batch, hid_feature, time, freq]
        x = self.dense_conv_2(x)  # [batch, hid_feature, time, freq//2]
        return x

class MagDecoder(nn.Module):
    """
    MagDecoder module for decoding magnitude information.
    """
    def __init__(self, cfg):
        super(MagDecoder, self).__init__()
        self.dense_block = DenseBlock(cfg, depth=4)
        self.hid_feature = cfg['model_cfg']['hid_feature']
        self.output_channel = cfg['model_cfg']['output_channel']
        self.n_fft = cfg['stft_cfg']['n_fft']
        self.beta = cfg['model_cfg']['beta']

        self.mask_conv = nn.Sequential(
            nn.Conv2d(self.hid_feature, self.hid_feature * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(self.hid_feature, self.hid_feature, kernel_size=(1, 3), stride=(2, 1), padding=(0, 1),
                      groups=self.hid_feature, bias=False),
            nn.Conv2d(self.hid_feature, self.output_channel, (1, 1)),
            nn.InstanceNorm2d(self.output_channel, affine=True),
            nn.PReLU(self.output_channel),
            nn.Conv2d(self.output_channel, self.output_channel, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid2D(self.n_fft // 2 + 1, beta=self.beta)

    def forward(self, x):
        """
        Forward pass for the MagDecoder module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Decoded tensor with magnitude information.
        """
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = rearrange(x, 'b c t f -> b f t c').squeeze(-1)
        x = self.lsigmoid(x)
        x = rearrange(x, 'b f t -> b t f').unsqueeze(1)
        return x

class PhaseDecoder(nn.Module):
    """
    PhaseDecoder module for decoding phase information.
    """
    def __init__(self, cfg):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DenseBlock(cfg, depth=4)
        self.hid_feature = cfg['model_cfg']['hid_feature']
        self.output_channel = cfg['model_cfg']['output_channel'] * 2  # produce cos/sin correction

        self.phase_conv = nn.Sequential(
            nn.Conv2d(self.hid_feature, self.hid_feature * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(self.hid_feature, self.hid_feature, kernel_size=(1, 3), stride=(2, 1), padding=(0, 1),
                      groups=self.hid_feature, bias=False),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature)
        )

        self.phase_conv_out = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))

    def forward(self, x):
        """
        Forward pass for the PhaseDecoder module.
        Returns 2-channel correction (cos-like, sin-like), each in [-1,1].
        """
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x = torch.tanh(self.phase_conv_out(x))
        return x


"""可插拔的跨域交叉注意力模块。

本文件复用 `transformer.CrossTransformerEncoderLayer` 的设计思路，
提供一个可迁移的 Cross-Domain Mamba-Transformer Block (CDMT block)，
支持时域/时频域特征的双向交叉注意力，并暴露稀疏注意力相关参数。
"""

def scaled_query_key_softmax(q, k, att_mask):
    from xformers.ops import masked_matmul
    q = q / (k.size(-1)) ** 0.5
    att = masked_matmul(q, k.transpose(-2, -1), att_mask)
    att = torch.nn.functional.softmax(att, -1)
    return att


def scaled_dot_product_attention(q, k, v, att_mask, dropout):
    att = scaled_query_key_softmax(q, k, att_mask=att_mask)
    att = dropout(att)
    y = att @ v
    return y


def _compute_buckets(x, R):
    qq = torch.einsum('btf,bfhi->bhti', x, R)
    qq = torch.cat([qq, -qq], dim=-1)
    buckets = qq.argmax(dim=-1)

    return buckets.permute(0, 2, 1).byte().contiguous()

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        auto_sparsity=None,
    ):
        super().__init__()
        assert auto_sparsity is not None, "sanity check"
        self.num_heads = num_heads
        self.q = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = torch.nn.Dropout(dropout)
        self.proj = torch.nn.Linear(embed_dim, embed_dim, bias)
        self.proj_drop = torch.nn.Dropout(dropout)
        self.batch_first = batch_first
        self.auto_sparsity = auto_sparsity

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
    ):

        if not self.batch_first:  # N, B, C
            query = query.permute(1, 0, 2)  # B, N_q, C
            key = key.permute(1, 0, 2)  # B, N_k, C
            value = value.permute(1, 0, 2)  # B, N_k, C
        B, N_q, C = query.shape
        B, N_k, C = key.shape

        q = (
            self.q(query)
            .reshape(B, N_q, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        q = q.flatten(0, 1)
        k = (
            self.k(key)
            .reshape(B, N_k, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = k.flatten(0, 1)
        v = (
            self.v(value)
            .reshape(B, N_k, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = v.flatten(0, 1)

        if self.auto_sparsity:
            assert attn_mask is None
            x = dynamic_sparse_attention(q, k, v, sparsity=self.auto_sparsity)
        else:
            x = scaled_dot_product_attention(q, k, v, attn_mask, dropout=self.attn_drop)
        x = x.reshape(B, self.num_heads, N_q, C // self.num_heads)

        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if not self.batch_first:
            x = x.permute(1, 0, 2)
        return x, None


def dynamic_sparse_attention(query, key, value, sparsity, infer_sparsity=True, attn_bias=None):
    # assert False, "The code for the custom sparse kernel is not ready for release yet."
    from xformers.ops import find_locations, sparse_memory_efficient_attention
    n_hashes = 32
    proj_size = 4
    query, key, value = [x.contiguous() for x in [query, key, value]]
    with torch.no_grad():
        R = torch.randn(1, query.shape[-1], n_hashes, proj_size // 2, device=query.device)
        bucket_query = _compute_buckets(query, R)
        bucket_key = _compute_buckets(key, R)
        row_offsets, column_indices = find_locations(
            bucket_query, bucket_key, sparsity, infer_sparsity)
    return sparse_memory_efficient_attention(
        query, key, value, row_offsets, column_indices, attn_bias)



class CrossAttentionUnit(nn.Module):
    """
    单向交叉注意力 + 前馈层。

    输入输出形状遵循 `batch_first` 约定：
    - batch_first=True: (B, L, C)
    - batch_first=False: (L, B, C)

    稀疏注意力相关参数全部暴露，默认使用密集注意力。
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: tp.Union[str, tp.Callable] = F.relu,
        layer_norm_eps: float = 1e-5,
        layer_scale: bool = False,
        init_values: float = 1e-4,
        norm_first: bool = False,
        group_norm: tp.Union[bool, int] = False,
        norm_out: tp.Union[bool, int] = False,
        sparse: bool = False,
        mask_type: str = "diag",
        mask_random_seed: int = 42,
        sparse_attn_window: int = 500,
        global_window: int = 50,
        sparsity: float = 0.95,
        auto_sparsity: tp.Optional[float] = None,
        batch_first: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.batch_first = batch_first
        self.sparse = sparse
        self.auto_sparsity = auto_sparsity
        if sparse:
            self.mask_type = mask_type
            self.sparse_attn_window = sparse_attn_window
            self.global_window = global_window
            self.sparsity = sparsity
            self.mask_random_seed = mask_random_seed
            self.register_buffer("mask", torch.zeros(1, 1), persistent=False)
        else:
            self.mask = None
            self.mask_random_seed = None

        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        if group_norm:
            self.norm1: nn.Module = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2: nn.Module = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm3: nn.Module = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.norm_out = None
        if self.norm_first and norm_out:
            self.norm_out = MyGroupNorm(num_groups=int(norm_out), num_channels=d_model)

        self.gamma_1 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        self.gamma_2 = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = self._get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, q: torch.Tensor, k: torch.Tensor, attn_mask: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        q: (B, Lq, C) 或 (Lq, B, C)
        k: (B, Lk, C) 或 (Lk, B, C)
        返回 shape 与 q 相同。
        """
        device = q.device
        if self.batch_first:
            B, Lq, C = q.shape
            _, Lk, _ = k.shape
        else:
            Lq, B, C = q.shape
            Lk, _, _ = k.shape

        if self.sparse:
            if attn_mask is None or attn_mask.shape[-1] != Lk or attn_mask.shape[-2] != Lq:
                attn_mask = get_mask(
                    Lk,
                    Lq,
                    self.mask_type,
                    self.sparse_attn_window,
                    self.global_window,
                    self.mask_random_seed,
                    self.sparsity,
                    device,
                )
                self.mask = attn_mask
        if self.norm_first:
            x = q + self.gamma_1(self._ca_block(self.norm1(q), self.norm2(k), attn_mask))
            x = x + self.gamma_2(self._ff_block(self.norm3(x)))
            if self.norm_out:
                x = self.norm_out(x)
        else:
            x = self.norm1(q + self.gamma_1(self._ca_block(q, k, attn_mask)))
            x = self.norm2(x + self.gamma_2(self._ff_block(x)))

        return x

    def _ca_block(self, q, k, attn_mask=None):
        # need_weights=False 加速；稀疏/密集接口兼容
        x = self.cross_attn(q, k, k, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class CrossDomainMambaTransformerBlock(nn.Module):
    """
    CDMT-block：两个分支（时域 Xt: B,C,L；时频域 Xtf: B,C,T,F）之间的双向交叉注意力。

    - 输入输出保持原始维度不变。
    - `share_weights=True` 表示两条方向复用同一组参数（参数更少，对称性强）；
      `False` 表示各向独立，分别一套参数（更灵活）。
    - `batch_first` 控制内部注意力维度布局，默认 True 以避免转置开销，
      但无论如何都会还原为输入形状。
    - 稀疏/掩码参数全部暴露，便于大尺度输入时控制窗口与稀疏率。
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: tp.Union[str, tp.Callable] = F.relu,
        layer_norm_eps: float = 1e-5,
        layer_scale: bool = False,
        init_values: float = 1e-4,
        norm_first: bool = False,
        group_norm: tp.Union[bool, int] = False,
        norm_out: tp.Union[bool, int] = False,
        share_weights: bool = True,
        batch_first: bool = True,
        sparse: bool = False,
        mask_type: str = "diag",
        mask_random_seed: int = 42,
        sparse_attn_window: int = 500,
        global_window: int = 50,
        sparsity: float = 0.95,
        auto_sparsity: tp.Optional[float] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        unit_kwargs = dict(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            layer_scale=layer_scale,
            init_values=init_values,
            norm_first=norm_first,
            group_norm=group_norm,
            norm_out=norm_out,
            sparse=sparse,
            mask_type=mask_type,
            mask_random_seed=mask_random_seed,
            sparse_attn_window=sparse_attn_window,
            global_window=global_window,
            sparsity=sparsity,
            auto_sparsity=auto_sparsity,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )

        self.share_weights = share_weights
        if share_weights:
            shared = CrossAttentionUnit(**unit_kwargs)
            self.t_to_tf = shared
            self.tf_to_t = shared
        else:
            self.t_to_tf = CrossAttentionUnit(**unit_kwargs)
            self.tf_to_t = CrossAttentionUnit(**unit_kwargs)

        self.batch_first = batch_first

    def forward(
        self,
        xt: torch.Tensor,
        xtf: torch.Tensor,
        attn_mask_t_to_tf: tp.Optional[torch.Tensor] = None,
        attn_mask_tf_to_t: tp.Optional[torch.Tensor] = None,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xt: 时域分支，形状 (B, C, L)
            xtf: 时频分支，形状 (B, C, T, F)
            attn_mask_t_to_tf: 可选，形状 (Lt, Ltf) 或稀疏掩码
            attn_mask_tf_to_t: 可选，形状 (Ltf, Lt) 或稀疏掩码
        Returns:
            xt_hat: (B, C, L)
            xtf_hat: (B, C, T, F)
        """
        B, C, L = xt.shape
        B2, C2, T, Freq = xtf.shape
        assert B == B2 and C == C2, "Xt 和 Xtf 的 batch/channel 需匹配"

        # 展平成序列：时域长度 Lt=L，时频长度 Ltf=T*F
        xt_seq = rearrange(xt, "b c l -> b l c") if self.batch_first else rearrange(xt, "b c l -> l b c")
        xtf_seq = rearrange(xtf, "b c t f -> b (t f) c") if self.batch_first else rearrange(xtf, "b c t f -> (t f) b c")

        # 双向交叉注意力
        xt_updated = self.tf_to_t(xt_seq, xtf_seq, attn_mask_tf_to_t)
        xtf_updated = self.t_to_tf(xtf_seq, xt_seq, attn_mask_t_to_tf)

        # 还原维度
        xt_out = rearrange(xt_updated, "b l c -> b c l") if self.batch_first else rearrange(xt_updated, "l b c -> b c l")
        xtf_out = rearrange(xtf_updated, "b (t f) c -> b c t f", t=T, f=Freq) if self.batch_first else rearrange(xtf_updated, "(t f) b c -> b c t f", t=T, f=Freq)
        return xt_out, xtf_out


class PluggableCrossAttention(nn.Module):
    """
    通用可插拔交叉注意力：输入 Q/K/V，输出与 Q 同形状（仅融合 KV 信息）。
    - 适合在任意网络里直接替换/插入。
    - batch_first=True 时输入输出为 (B, L, C)；False 时为 (L, B, C)。
    - 稀疏/掩码参数与 `CrossAttentionUnit` 保持一致，便于大尺度场景。
    -norm_first:如果为 True (Pre-Norm)：先归一化再做注意力，模型更稳定，适合深层网络，
        否则为 Post-Norm（先注意力后归一化），适合浅层网络。
    -layer_scale: 引入 LayerScale。这是一种在残差连接处引入可学习缩放因子的技术（gamma_attn），能有效解决极深网络训练不稳定的问题。
    -group_norm: 允许使用 GroupNorm 替代 LayerNorm。这在处理 2D 音频特征（如 $B, C, T, F$）时非常有用。
    -sparse_attn_window / global_window: 决定了搜索字典时的“视力范围”，显著降低计算复杂度。
    -sparse: 是否开启稀疏。开启后，Query 不会看所有的 Key，而只看特定窗口内的。


    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: tp.Union[str, tp.Callable] = F.relu,
        layer_norm_eps: float = 1e-5,
        layer_scale: bool = False,
        init_values: float = 1e-4,
        norm_first: bool = False,
        group_norm: tp.Union[bool, int] = False,
        norm_out: tp.Union[bool, int] = False,
        batch_first: bool = True,
        sparse: bool = False,
        mask_type: str = "diag",
        mask_random_seed: int = 42,
        sparse_attn_window: int = 500,
        global_window: int = 50,
        sparsity: float = 0.95,
        auto_sparsity: tp.Optional[float] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.sparse = sparse
        self.auto_sparsity = auto_sparsity
        if sparse:
            self.mask_type = mask_type
            self.sparse_attn_window = sparse_attn_window
            self.global_window = global_window
            self.sparsity = sparsity
            self.mask_random_seed = mask_random_seed
            self.register_buffer("mask", torch.zeros(1, 1), persistent=False)
        else:
            self.mask = None
            self.mask_random_seed = None

        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        if group_norm:
            self.norm_q: nn.Module = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm_k: nn.Module = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm_ff: nn.Module = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm_k = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm_ff = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.norm_out = None
        if self.norm_first and norm_out:
            self.norm_out = MyGroupNorm(num_groups=int(norm_out), num_channels=d_model)

        self.gamma_attn = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        self.gamma_ffn = LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = self._get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q: 查询序列，(B, Lq, C) 或 (Lq, B, C)
            k: 键序列，形状与 q 类似（长度可不同）
            v: 值序列，形状与 k 相同
            attn_mask: 可选，形状 (Lq, Lk) 或稀疏掩码
        Returns:
            与 q 同形状的融合结果，仅改变特征，长度/布局保持不变。
        """
        device = q.device
        if self.batch_first:
            B, Lq, C = q.shape
            _, Lk, _ = k.shape
        else:
            Lq, B, C = q.shape
            Lk, _, _ = k.shape

        if self.sparse:
            if attn_mask is None or attn_mask.shape[-1] != Lk or attn_mask.shape[-2] != Lq:
                attn_mask = get_mask(
                    Lk,
                    Lq,
                    self.mask_type,
                    self.sparse_attn_window,
                    self.global_window,
                    self.mask_random_seed,
                    self.sparsity,
                    device,
                )
                self.mask = attn_mask

        if self.norm_first:
            x = q + self.gamma_attn(self._attn_block(self.norm_q(q), self.norm_k(k), v, attn_mask))
            x = x + self.gamma_ffn(self._ff_block(self.norm_ff(x)))
            if self.norm_out:
                x = self.norm_out(x)
        else:
            x = self.norm_q(q + self.gamma_attn(self._attn_block(q, k, v, attn_mask)))
            x = self.norm_ff(x + self.gamma_ffn(self._ff_block(x)))
        return x

    def _attn_block(self, q, k, v, attn_mask=None):
        x = self.cross_attn(q, k, v, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout_attn(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout_ffn(x)

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def get_mask(T1: int,
             T2: int,
             mask_type: str = "diag",
             sparse_attn_window: int = 500,
             global_window: int = 50,
             mask_random_seed: int = 42,
             sparsity: float = 0.95,
             device=None) -> torch.Tensor:
    """构造用于稀疏注意力的掩码，shape (T2, T1)，与 torch.nn.MultiheadAttention 对齐。
    mask_type=diag: 仅保留对角附近窗口，其余位置为 -inf；可选全局 token。
    """
    if mask_type != "diag":
        raise NotImplementedError("当前仅支持 mask_type='diag'")
    if device is None:
        device = "cpu"
    mask = torch.full((T2, T1), float('-inf'), device=device)
    gw = max(0, global_window)
    if gw > 0:
        mask[:, :gw] = 0
        mask[:, max(T1 - gw, 0):] = 0
    w = max(1, sparse_attn_window)
    idx_k = torch.arange(T1, device=device).unsqueeze(0)  # [1, T1]
    start = (torch.arange(T2, device=device).unsqueeze(1) - w).clamp(min=0)  # [T2,1]
    end = (torch.arange(T2, device=device).unsqueeze(1) + w + 1).clamp(max=T1)  # [T2,1]
    window_mask = (idx_k >= start) & (idx_k < end)  # [T2,T1]
    mask = torch.where(window_mask, torch.zeros_like(mask), mask)
    return mask

