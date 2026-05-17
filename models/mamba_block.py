# Reference: https://github.com/RoyChao19477/SEMamba/models/mamba_block

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp
from torch.nn import init
from torch.nn.parameter import Parameter
from functools import partial
from einops import rearrange
from torch.nn import MultiheadAttention, GRU, Linear, LayerNorm, Dropout
import math
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layernorm import RMSNorm

# github: https://github.com/state-spaces/mamba/blob/9127d1f47f367f5c9cc49c73ad73557089d02cb8/mamba_ssm/models/mixer_seq_simple.py
def create_block(
    d_model, cfg, layer_idx=0, rms_norm=True, fused_add_norm=False, residual_in_fp32=False,
    ):
    d_state = cfg['model_cfg']['d_state'] # 16
    d_conv = cfg['model_cfg']['d_conv'] # 4
    expand = cfg['model_cfg']['expand'] # 4
    norm_epsilon = cfg['model_cfg']['norm_epsilon'] # 0.00001

    mixer_cls = partial(Mamba, layer_idx=layer_idx, d_state=d_state, d_conv=d_conv, expand=expand)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon
    )
    block = Block(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            )
    block.layer_idx = layer_idx
    return block

class MambaBlock(nn.Module):
    def __init__(self, in_channels, cfg):
        super(MambaBlock, self).__init__()
        n_layer = 1
        self.forward_blocks  = nn.ModuleList( create_block(in_channels, cfg) for i in range(n_layer) )
        self.backward_blocks = nn.ModuleList( create_block(in_channels, cfg) for i in range(n_layer) )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
            )
        )

    def forward(self, x):
        x_forward, x_backward = x.clone(), torch.flip(x, [1])
        resi_forward, resi_backward = None, None

        # Forward
        for layer in self.forward_blocks:
            x_forward, resi_forward = layer(x_forward, resi_forward)
        y_forward = (x_forward + resi_forward) if resi_forward is not None else x_forward

        # Backward
        for layer in self.backward_blocks:
            x_backward, resi_backward = layer(x_backward, resi_backward)
        y_backward = torch.flip((x_backward + resi_backward), [1]) if resi_backward is not None else torch.flip(x_backward, [1])

        return torch.cat([y_forward, y_backward], -1)


class TFMambaBlock(nn.Module):
    """
    Temporal-Frequency Mamba block for sequence modeling.

    Attributes:
    cfg (Config): Configuration for the block.
    time_mamba (MambaBlock): Mamba block for temporal dimension.
    freq_mamba (MambaBlock): Mamba block for frequency dimension.
    tlinear (ConvTranspose1d): ConvTranspose1d layer for temporal dimension.
    flinear (ConvTranspose1d): ConvTranspose1d layer for frequency dimension.
    """

    def __init__(self, cfg, inchannels):
        super(TFMambaBlock, self).__init__()
        self.cfg = cfg
        self.hid_feature = inchannels

        # Initialize Mamba blocks
        self.time_mamba = MambaBlock(in_channels=self.hid_feature, cfg=cfg)
        self.freq_mamba = MambaBlock(in_channels=self.hid_feature, cfg=cfg)

        # Initialize ConvTranspose1d layers
        self.tlinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)
        self.flinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)

    def forward(self, x):
        """
        Forward pass of the TFMamba block.

        Parameters:
        x (Tensor): Input tensor with shape (batch, channels, time, freq).

        Returns:
        Tensor: Output tensor after applying temporal and frequency Mamba blocks.
        """
        b, c, t, f = x.size()

        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.tlinear(self.time_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.flinear(self.freq_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


class TMambaBlock(nn.Module):
    def __init__(self, cfg, inchannels):
        super(TMambaBlock, self).__init__()
        self.hid_feature = inchannels
        self.time_mamba = MambaBlock(in_channels=self.hid_feature, cfg=cfg)
        self.tlinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        # 保证送入 Mamba 的通道维与定义一致
        assert x.shape[-1] == self.hid_feature, f"TMambaBlock expect {self.hid_feature} channels, got {x.shape[-1]}"
        x = self.tlinear(self.time_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        x = x.view(b, f, t, c).permute(0, 3, 2, 1)
        return x


class FMambaBlock(nn.Module):
    def __init__(self, cfg, inchannels):
        super(FMambaBlock, self).__init__()
        self.hid_feature = inchannels
        self.freq_mamba = MambaBlock(in_channels=self.hid_feature, cfg=cfg)
        self.flinear = nn.ConvTranspose1d(self.hid_feature * 2, self.hid_feature, 1, stride=1)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(b * t, f, c)
        x = self.flinear(self.freq_mamba(x).permute(0, 2, 1)).permute(0, 2, 1) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


class ComplexRMSNorm1D(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, x_r, x_i):
        rms = torch.sqrt(torch.mean(x_r ** 2 + x_i ** 2, dim=-1, keepdim=True) + self.eps)
        y_r = x_r / rms * self.weight
        y_i = x_i / rms * self.weight
        return y_r, y_i


class ComplexLinearNoBias(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.real = nn.Linear(in_features, out_features, bias=False)
        self.imag = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x_r, x_i):
        y_r = self.real(x_r) - self.imag(x_i)
        y_i = self.real(x_i) + self.imag(x_r)
        return y_r, y_i


class ComplexDepthwiseConv1dNoBias(nn.Module):
    def __init__(self, channels, d_conv):
        super().__init__()
        self.conv_re = nn.Conv1d(
            channels,
            channels,
            kernel_size=d_conv,
            groups=channels,
            padding=d_conv - 1,
            bias=False,
        )
        self.conv_im = nn.Conv1d(
            channels,
            channels,
            kernel_size=d_conv,
            groups=channels,
            padding=d_conv - 1,
            bias=False,
        )

    def forward(self, x_r, x_i):
        b, l, c = x_r.shape
        x_r = x_r.permute(0, 2, 1)
        x_i = x_i.permute(0, 2, 1)
        y_r = self.conv_re(x_r) - self.conv_im(x_i)
        y_i = self.conv_re(x_i) + self.conv_im(x_r)
        y_r = y_r[..., :l]
        y_i = y_i[..., :l]
        y_r = y_r.permute(0, 2, 1)
        y_i = y_i.permute(0, 2, 1)
        return y_r, y_i


class ComplexConv2dNoBias(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        super().__init__()
        self.conv_re = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.conv_im = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

    def forward(self, x_r, x_i):
        y_r = self.conv_re(x_r) - self.conv_im(x_i)
        y_i = self.conv_re(x_i) + self.conv_im(x_r)
        return y_r, y_i


class ComplexRMSNorm2D(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x_r, x_i):
        rms = torch.sqrt(torch.mean(x_r ** 2 + x_i ** 2, dim=1, keepdim=True) + self.eps)
        y_r = x_r / rms * self.weight
        y_i = x_i / rms * self.weight
        return y_r, y_i


class ComplexPointwiseGate2D(nn.Module):
    def __init__(self, cond_channels, out_channels, scale=1.0):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Conv2d(cond_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, cond):
        logits = self.net(cond)
        return 1.0 + self.scale * torch.tanh(logits)


class ComplexDownsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if out_channels % 4 != 0:
            raise ValueError("ComplexDownsample2D requires out_channels divisible by 4.")
        self.pre = ComplexConv2dNoBias(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
        )
        self.reduce = ComplexConv2dNoBias(in_channels, out_channels // 4, kernel_size=1)
        self.unshuffle = nn.PixelUnshuffle(2)

    def forward(self, x_r, x_i):
        y_r, y_i = self.pre(x_r, x_i)
        y_r, y_i = self.reduce(y_r, y_i)
        y_r = self.unshuffle(y_r)
        y_i = self.unshuffle(y_i)
        return y_r, y_i


class ComplexUpsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pre = ComplexConv2dNoBias(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
        )
        self.expand = ComplexConv2dNoBias(in_channels, out_channels * 4, kernel_size=1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x_r, x_i):
        y_r, y_i = self.pre(x_r, x_i)
        y_r, y_i = self.expand(y_r, y_i)
        y_r = self.shuffle(y_r)
        y_i = self.shuffle(y_i)
        return y_r, y_i


class ComplexConcatFuse2D(nn.Module):
    def __init__(self, in_channels, out_channels, eps=1e-5):
        super().__init__()
        self.conv = ComplexConv2dNoBias(in_channels, out_channels, kernel_size=1)
        self.norm = ComplexRMSNorm2D(out_channels, eps=eps)

    def forward(self, x_r, x_i, skip_r, skip_i):
        cat_r = torch.cat([x_r, skip_r], dim=1)
        cat_i = torch.cat([x_i, skip_i], dim=1)
        y_r, y_i = self.conv(cat_r, cat_i)
        y_r, y_i = self.norm(y_r, y_i)
        return y_r, y_i


class EqMamba1D(nn.Module):
    def __init__(self, channels, cfg):
        super().__init__()
        d_state = cfg['model_cfg'].get('d_state', 16)
        d_conv = cfg['model_cfg'].get('d_conv', 4)
        expand = cfg['model_cfg'].get('expand', 4)
        norm_epsilon = cfg['model_cfg'].get('norm_epsilon', 1e-5)
        eq_mamba_dropout = cfg['model_cfg'].get('eq_mamba_dropout', 0.0)
        eq_mamba_use_complex_conv = cfg['model_cfg'].get('eq_mamba_use_complex_conv', True)

        self.channels = channels
        self.d_inner = channels * expand
        self.d_state = d_state
        self.dt_rank = math.ceil(channels / 16)
        self.eps = norm_epsilon

        self.norm = ComplexRMSNorm1D(channels, eps=norm_epsilon)
        self.in_proj = ComplexLinearNoBias(channels, self.d_inner)
        self.conv1d = ComplexDepthwiseConv1dNoBias(self.d_inner, d_conv) if eq_mamba_use_complex_conv else None
        self.param_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.gate_norm = nn.LayerNorm(self.d_inner)
        self.gate_act = nn.SiLU()
        self.gate_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.out_proj = ComplexLinearNoBias(self.d_inner, channels)
        self.dropout = nn.Dropout(eq_mamba_dropout)

        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

    def forward(self, x_r, x_i):
        x_r, x_i = self.norm(x_r, x_i)
        u_r, u_i = self.in_proj(x_r, x_i)
        if self.conv1d is not None:
            u_r, u_i = self.conv1d(u_r, u_i)

        inv = torch.sqrt(u_r ** 2 + u_i ** 2 + self.eps)
        gate = self.gate_act(self.gate_norm(inv))
        gate = torch.sigmoid(self.gate_proj(gate))
        u_r = u_r * gate
        u_i = u_i * gate

        b, l, _ = inv.shape
        param = self.param_proj(inv.reshape(b * l, self.d_inner))
        dt_raw, b_param, c_param = torch.split(param, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt_raw.t()
        dt = rearrange(dt, "d (b l) -> b d l", b=b, l=l)
        b_param = rearrange(b_param, "(b l) n -> b n l", b=b, l=l).contiguous()
        c_param = rearrange(c_param, "(b l) n -> b n l", b=b, l=l).contiguous()

        u_r_scan = rearrange(u_r, "b l d -> b d l").contiguous()
        u_i_scan = rearrange(u_i, "b l d -> b d l").contiguous()
        A = -torch.exp(self.A_log.float())

        y_r = selective_scan_fn(
            u_r_scan,
            dt,
            A,
            b_param,
            c_param,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        y_i = selective_scan_fn(
            u_i_scan,
            dt,
            A,
            b_param,
            c_param,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )

        y_r = rearrange(y_r, "b d l -> b l d")
        y_i = rearrange(y_i, "b d l -> b l d")
        y_r = y_r * gate
        y_i = y_i * gate
        y_r, y_i = self.out_proj(y_r, y_i)
        y_r = self.dropout(y_r)
        y_i = self.dropout(y_i)
        return y_r, y_i


class EqBiMamba1D(nn.Module):
    def __init__(self, channels, cfg):
        super().__init__()
        self.forward_core = EqMamba1D(channels, cfg)
        self.backward_core = EqMamba1D(channels, cfg)
        self.fuse = ComplexLinearNoBias(2 * channels, channels)

    def forward(self, x_r, x_i):
        y_fr, y_fi = self.forward_core(x_r, x_i)
        x_br = torch.flip(x_r, dims=[1])
        x_bi = torch.flip(x_i, dims=[1])
        y_br, y_bi = self.backward_core(x_br, x_bi)
        y_br = torch.flip(y_br, dims=[1])
        y_bi = torch.flip(y_bi, dims=[1])
        cat_r = torch.cat([y_fr, y_br], dim=-1)
        cat_i = torch.cat([y_fi, y_bi], dim=-1)
        y_r, y_i = self.fuse(cat_r, cat_i)
        return y_r, y_i


class EqMambaAdapter1D(nn.Module):
    # Eq-Mamba-v1 Adapter for phase bottleneck.
    # The internal complex Mamba core uses a complex value path and
    # rotation-invariant SSM parameter generation.
    # Since the surrounding phase branch is still real-valued,
    # this module is an Eq-Mamba-inspired adapter rather than a full GRE phase stream.
    def __init__(self, cfg, channels):
        super().__init__()
        eq_mamba_res_scale = cfg['model_cfg'].get('eq_mamba_res_scale', 1.0)
        eq_mamba_bidirectional = cfg['model_cfg'].get('eq_mamba_bidirectional', True)
        self.to_complex = nn.Linear(channels, 2 * channels, bias=False)
        self.to_real = nn.Linear(2 * channels, channels, bias=False)
        self.eq_core = EqBiMamba1D(channels, cfg) if eq_mamba_bidirectional else EqMamba1D(channels, cfg)
        self.res_scale = nn.Parameter(torch.tensor(eq_mamba_res_scale))

    def forward(self, x):
        x_ri = self.to_complex(x)
        x_r, x_i = torch.chunk(x_ri, 2, dim=-1)
        y_r, y_i = self.eq_core(x_r, x_i)
        y_ri = torch.cat([y_r, y_i], dim=-1)
        y = self.to_real(y_ri)
        return x + self.res_scale * y


class EqTMambaBlock(nn.Module):
    def __init__(self, cfg, inchannels):
        super().__init__()
        self.hid_feature = inchannels
        self.eq_mamba_adapter = EqMambaAdapter1D(cfg, inchannels)

    def forward(self, x):
        b, c, t, f = x.size()
        x_seq = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        assert x_seq.shape[-1] == self.hid_feature, f"EqTMambaBlock expect {self.hid_feature} channels, got {x_seq.shape[-1]}"
        y_seq = self.eq_mamba_adapter(x_seq)
        y = y_seq.view(b, f, t, c).permute(0, 3, 2, 1).contiguous()
        return y


class EqFMambaBlock(nn.Module):
    def __init__(self, cfg, inchannels):
        super().__init__()
        self.hid_feature = inchannels
        self.eq_mamba_adapter = EqMambaAdapter1D(cfg, inchannels)

    def forward(self, x):
        b, c, t, f = x.size()
        x_seq = x.permute(0, 2, 3, 1).contiguous().view(b * t, f, c)
        assert x_seq.shape[-1] == self.hid_feature, f"EqFMambaBlock expect {self.hid_feature} channels, got {x_seq.shape[-1]}"
        y_seq = self.eq_mamba_adapter(x_seq)
        y = y_seq.view(b, t, f, c).permute(0, 3, 1, 2).contiguous()
        return y


class ComplexEqTMambaBlock(nn.Module):
    def __init__(self, cfg, inchannels):
        super().__init__()
        eq_mamba_res_scale = cfg['model_cfg'].get('eq_mamba_res_scale', 1.0)
        eq_mamba_bidirectional = cfg['model_cfg'].get('eq_mamba_bidirectional', True)
        self.hid_feature = inchannels
        self.eq_core = EqBiMamba1D(inchannels, cfg) if eq_mamba_bidirectional else EqMamba1D(inchannels, cfg)
        self.res_scale = nn.Parameter(torch.tensor(eq_mamba_res_scale))

    def forward(self, x_r, x_i):
        b, c, t, f = x_r.size()
        x_r_seq = x_r.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x_i_seq = x_i.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        dy_r, dy_i = self.eq_core(x_r_seq, x_i_seq)
        y_r_seq = x_r_seq + self.res_scale * dy_r
        y_i_seq = x_i_seq + self.res_scale * dy_i
        y_r = y_r_seq.view(b, f, t, c).permute(0, 3, 2, 1).contiguous()
        y_i = y_i_seq.view(b, f, t, c).permute(0, 3, 2, 1).contiguous()
        return y_r, y_i


class ComplexEqFMambaBlock(nn.Module):
    def __init__(self, cfg, inchannels):
        super().__init__()
        eq_mamba_res_scale = cfg['model_cfg'].get('eq_mamba_res_scale', 1.0)
        eq_mamba_bidirectional = cfg['model_cfg'].get('eq_mamba_bidirectional', True)
        self.hid_feature = inchannels
        self.eq_core = EqBiMamba1D(inchannels, cfg) if eq_mamba_bidirectional else EqMamba1D(inchannels, cfg)
        self.res_scale = nn.Parameter(torch.tensor(eq_mamba_res_scale))

    def forward(self, x_r, x_i):
        b, c, t, f = x_r.size()
        x_r_seq = x_r.permute(0, 2, 3, 1).contiguous().view(b * t, f, c)
        x_i_seq = x_i.permute(0, 2, 3, 1).contiguous().view(b * t, f, c)
        dy_r, dy_i = self.eq_core(x_r_seq, x_i_seq)
        y_r_seq = x_r_seq + self.res_scale * dy_r
        y_i_seq = x_i_seq + self.res_scale * dy_i
        y_r = y_r_seq.view(b, t, f, c).permute(0, 3, 1, 2).contiguous()
        y_i = y_i_seq.view(b, t, f, c).permute(0, 3, 1, 2).contiguous()
        return y_r, y_i


class PhaseMidToComplex2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(channels, 2 * channels, kernel_size=1, bias=False)

    def forward(self, x):
        x_ri = self.proj(x)
        x_r, x_i = torch.chunk(x_ri, 2, dim=1)
        return x_r, x_i


class PhaseMidToReal2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False)

    def forward(self, x_r, x_i):
        x = self.proj(torch.cat([x_r, x_i], dim=1))
        return x


class CoupledMambaFusion(nn.Module):
    """Simple mag/pha融合: 可选预归一化，concat -> reduce -> gate，零初始化保证初始不融合。"""

    def __init__(self, channels, use_bn=True, use_prenorm=True):
        super().__init__()
        self.pre_norm_mag = nn.BatchNorm2d(channels) if use_prenorm else nn.Identity()
        self.pre_norm_pha = nn.BatchNorm2d(channels) if use_prenorm else nn.Identity()
        # 路A: [A,B] -> brainA
        self.reduce_a = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.bn_a = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.gate_a = nn.Conv2d(channels, 1, 1)
        self.mix_scale_a = nn.Parameter(torch.zeros(1))
        # 路B: [B,A] -> brainB
        self.reduce_b = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.bn_b = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.gate_b = nn.Conv2d(channels, 1, 1)
        self.mix_scale_b = nn.Parameter(torch.zeros(1))
        self.act = nn.GELU()
        # 零初始化，初始退化为恒等
        for layer in [self.reduce_a, self.reduce_b, self.gate_a, self.gate_b]:
            nn.init.zeros_(layer.weight)
            if getattr(layer, 'bias', None) is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, mag_feat, pha_feat):
        mag_n = self.pre_norm_mag(mag_feat)
        pha_n = self.pre_norm_pha(pha_feat)
        # 路A: concat(A,B)
        concat_ab = torch.cat([mag_n, pha_n], dim=1)
        f_a = self.act(self.bn_a(self.reduce_a(concat_ab)))
        gate_a = torch.sigmoid(self.gate_a(f_a))
        f_a = self.mix_scale_a * f_a
        mag_out = mag_feat + gate_a * f_a
        # 路B: concat(B,A)
        concat_ba = torch.cat([pha_n, mag_n], dim=1)
        f_b = self.act(self.bn_b(self.reduce_b(concat_ba)))
        gate_b = torch.sigmoid(self.gate_b(f_b))
        f_b = self.mix_scale_b * f_b
        pha_out = pha_feat + gate_b * f_b
        return mag_out, pha_out


class GlobalFusionGate(nn.Module):
    """双脑全局融合：路A(B)各自看[A,B]/[B,A]后加回自身，零初始化保持恒等。"""

    def __init__(self, channels, pool_t=1, pool_f=1, use_prenorm=True, use_bn=True):
        super().__init__()
        self.pool_t = pool_t
        self.pool_f = pool_f
        self.pool = nn.AvgPool2d((pool_t, pool_f), ceil_mode=True) if (pool_t > 1 or pool_f > 1) else None
        self.pre_norm_mag = nn.BatchNorm2d(channels) if use_prenorm else nn.Identity()
        self.pre_norm_pha = nn.BatchNorm2d(channels) if use_prenorm else nn.Identity()
        # 路A: concat(A,B)
        self.reduce_a = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.bn_a = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.gate_a = nn.Conv2d(channels, 1, 1)
        self.mix_scale_a = nn.Parameter(torch.zeros(1))
        # 路B: concat(B,A)
        self.reduce_b = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.bn_b = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.gate_b = nn.Conv2d(channels, 1, 1)
        self.mix_scale_b = nn.Parameter(torch.zeros(1))
        self.act = nn.GELU()
        for layer in [self.reduce_a, self.reduce_b, self.gate_a, self.gate_b]:
            nn.init.zeros_(layer.weight)
            if getattr(layer, 'bias', None) is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, mag_feat, pha_feat):
        mag_low, pha_low = self.pre_norm_mag(mag_feat), self.pre_norm_pha(pha_feat)
        if self.pool:
            mag_low = self.pool(mag_low)
            pha_low = self.pool(pha_low)

        # 路A: concat(A,B)
        concat_ab = torch.cat([mag_low, pha_low], dim=1)
        f_a = self.act(self.bn_a(self.reduce_a(concat_ab)))
        gate_a = torch.sigmoid(self.gate_a(f_a))
        fused_a = self.mix_scale_a * gate_a * f_a
        # 路B: concat(B,A)
        concat_ba = torch.cat([pha_low, mag_low], dim=1)
        f_b = self.act(self.bn_b(self.reduce_b(concat_ba)))
        gate_b = torch.sigmoid(self.gate_b(f_b))
        fused_b = self.mix_scale_b * gate_b * f_b

        if self.pool:
            fused_a = F.interpolate(fused_a, size=mag_feat.shape[2:], mode='nearest')
            fused_b = F.interpolate(fused_b, size=pha_feat.shape[2:], mode='nearest')

        mag_out = mag_feat + fused_a
        pha_out = pha_feat + fused_b
        return mag_out, pha_out


class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.RMSNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block_Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.RMSNorm,
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.B_Attention = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )


        # if counter in transformer_blocks:
        #     self.mixer = Attention(
        #     dim,
        #     num_heads=num_heads,
        #     qkv_bias=qkv_bias,
        #     qk_norm=qk_scale,
        #     attn_drop=attn_drop,
        #     proj_drop=drop,
        #     norm_layer=norm_layer,
        # )
        # else:
        #     self.mixer = MambaVisionMixer(d_model=dim,
        #                                   d_state=8,
        #                                   d_conv=3,
        #                                   expand=1
        #                                   )

        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        # self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        # self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1


    def forward(self, x):
        # x = x + self.drop_path(self.gamma_1 * self.B_Attention(self.norm1(x)))

        # x = x + self.B_Attention(self.norm1(x))
        # x = x + self.mlp(self.norm2(x))
        # x = self.mlp(self.norm2(self.B_Attention(self.norm1(x))))
        x = self.B_Attention(x)
        x = self.mlp(x)
        # x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))


        return x

class TF_Attention_Block(nn.Module):

    def __init__(self, cfg, inchannels,num_heads=4 ):
        super(TF_Attention_Block, self).__init__()
        self.cfg = cfg
        self.hid_feature = inchannels

        # 时域使用注意力机制
        self.time_block_attention = Block_Attention(
            dim=self.hid_feature,
            num_heads=num_heads,  # 必须参数
            mlp_ratio=4.0,  # 保持与原配置一致
            qkv_bias=False,  # 示例值，按需调整
            qk_scale=False,
            drop=0.0,
            attn_drop=0,
            drop_path=0,
            act_layer=nn.GELU,
            norm_layer=nn.RMSNorm,
            # Mlp_block=Mlp,
            layer_scale=None
        )
        self.freq_block_attention = Block_Attention(
            dim=self.hid_feature,
            num_heads=num_heads,  # 必须参数
            mlp_ratio=4.0,  # 保持与原配置一致
            qkv_bias=False,  # 示例值，按需调整
            qk_scale=False,
            drop=0.0,
            attn_drop=0,
            drop_path=0,
            act_layer=nn.GELU,
            norm_layer=nn.RMSNorm,
            # Mlp_block=Mlp,
            layer_scale=None
        )

    def forward(self, x):
        """
        Forward pass of the TFMamba block.

        Parameters:
        x (Tensor): Input tensor with shape (batch, channels, time, freq).

        Returns:
        Tensor: Output tensor after applying temporal and frequency Mamba blocks.
        """
        b, c, t, f = x.size()

        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_block_attention(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_block_attention(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)

        return x

class ChannelAttention(nn.Module):
    r""" Args:
            in_planes (int): Number of input image channels.
            ratio (int): Ratio of downscaling.
        """
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result
