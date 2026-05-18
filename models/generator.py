# Reference: https://github.com/huaidanquede/MUSE-Speech-Enhancement/tree/main/models/generator

import torch
import torch.nn as nn
import math
from torchvision.ops.deform_conv import DeformConv2d
from einops import rearrange
from copy import deepcopy
from .mamba_block import (
    TMambaBlock,
    FMambaBlock,
    TFMambaBlock,
    ComplexEqTMambaBlock,
    ComplexEqFMambaBlock,
    ComplexRMSNorm1D,
    ComplexLinearNoBias,
    ComplexDepthwiseConv1dNoBias,
    ComplexConv2dNoBias,
    ComplexRMSNorm2D,
    ComplexPointwiseGate2D,
    ComplexDownsample2D,
    ComplexUpsample2D,
    ComplexConcatFuse2D,
)
from .cross import GREVSSFinalFusion
from .codec_module import DenseEncoder, MagDecoder
import torch.nn.functional as F


#####################################
class DWConv2d_BN(nn.Module):

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
            bn_weight_init=1,
            offset_clamp=(-1, 1)
    ):
        super().__init__()

        self.offset_clamp = offset_clamp
        self.offset_generator = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3,
                                                        stride=1, padding=1, bias=False, groups=in_ch),
                                              nn.Conv2d(in_channels=in_ch, out_channels=18,
                                                        kernel_size=1,
                                                        stride=1, padding=0, bias=False)
                                              )
        self.dcn = DeformConv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=in_ch
        )
        self.pwconv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.act = act_layer() if act_layer is not None else nn.Identity()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        offset = self.offset_generator(x)

        if self.offset_clamp:
            offset = torch.clamp(offset, min=self.offset_clamp[0], max=self.offset_clamp[1])
        x = self.dcn(x, offset)

        x = self.pwconv(x)
        x = self.act(x)
        return x


class MB_Deform_Embedding(nn.Module):

    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 patch_size=16,
                 stride=1,
                 act_layer=nn.Hardswish,
                 offset_clamp=(-1, 1)):
        super().__init__()

        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=act_layer,
            offset_clamp=offset_clamp
        )

    def forward(self, x):
        """foward function"""
        x = self.patch_conv(x)

        return x


class Patch_Embed_stage(nn.Module):
    """Depthwise Convolutional Patch Embedding stage comprised of
    `DWCPatchEmbed` layers."""

    def __init__(self, in_chans, embed_dim, isPool=False, offset_clamp=(-1, 1)):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = MB_Deform_Embedding(
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_size=3,
            stride=1,
            offset_clamp=offset_clamp)

    def forward(self, x):
        """foward function"""

        att_inputs = self.patch_embeds(x)

        return att_inputs


#####################################
class Downsample(nn.Module):
    def __init__(self, input_feat, out_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            # dw
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1, padding=1, groups=input_feat, bias=False),
            # pw-linear
            nn.Conv2d(input_feat, out_feat // 4, 1, 1, 0, bias=False),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, input_feat, out_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            # dw
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1, padding=1, groups=input_feat, bias=False),
            # pw-linear
            nn.Conv2d(input_feat, out_feat * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class ComplexPhaFFN1D(nn.Module):
    def __init__(
        self,
        channels,
        expansion=4,
        dropout=0.0,
        norm_epsilon=1e-5,
        use_complex_conv=True,
        kernel_size=3,
    ):
        super().__init__()
        hidden = channels * expansion
        self.eps = norm_epsilon
        self.in_proj = ComplexLinearNoBias(channels, hidden * 2)
        self.dwconv = ComplexDepthwiseConv1dNoBias(hidden * 2, kernel_size) if use_complex_conv else None
        self.gate_ln = nn.LayerNorm(hidden)
        self.gate_act = nn.SiLU()
        self.out_proj = ComplexLinearNoBias(hidden, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_r, x_i):
        z_r, z_i = self.in_proj(x_r, x_i)
        if self.dwconv is not None:
            z_r, z_i = self.dwconv(z_r, z_i)

        v_r, g_r = torch.chunk(z_r, 2, dim=-1)
        v_i, g_i = torch.chunk(z_i, 2, dim=-1)

        gate_mag = torch.sqrt(g_r ** 2 + g_i ** 2 + self.eps)
        gate = self.gate_act(self.gate_ln(gate_mag))

        y_r = v_r * gate
        y_i = v_i * gate

        y_r, y_i = self.out_proj(y_r, y_i)
        y_r = self.dropout(y_r)
        y_i = self.dropout(y_i)
        return y_r, y_i


class PhaFFNAdapter2D(nn.Module):
    def __init__(
        self,
        channels,
        dropout=0.0,
        res_scale=1.0,
        expansion=4,
        norm_epsilon=1e-5,
        use_complex_conv=True,
        kernel_size=3,
    ):
        super().__init__()
        self.to_complex = nn.Conv2d(channels, 2 * channels, kernel_size=1, bias=False)
        self.cnorm_in = ComplexRMSNorm1D(channels, eps=norm_epsilon)
        self.pha_ffn = ComplexPhaFFN1D(
            channels=channels,
            expansion=expansion,
            dropout=dropout,
            norm_epsilon=norm_epsilon,
            use_complex_conv=use_complex_conv,
            kernel_size=kernel_size,
        )
        self.cnorm_out = ComplexRMSNorm1D(channels, eps=norm_epsilon)
        self.to_real = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False)
        self.res_scale = nn.Parameter(torch.tensor(res_scale))

    def forward(self, x):
        b, c, t, f = x.shape
        x_ri = self.to_complex(x)
        x_r, x_i = torch.chunk(x_ri, 2, dim=1)

        x_r = rearrange(x_r, 'b c t f -> b (t f) c')
        x_i = rearrange(x_i, 'b c t f -> b (t f) c')

        x_r, x_i = self.cnorm_in(x_r, x_i)
        y_r, y_i = self.pha_ffn(x_r, x_i)
        x_r = x_r + y_r
        x_i = x_i + y_i
        x_r, x_i = self.cnorm_out(x_r, x_i)

        x_r = rearrange(x_r, 'b (t f) c -> b c t f', t=t, f=f)
        x_i = rearrange(x_i, 'b (t f) c -> b c t f', t=t, f=f)

        y = self.to_real(torch.cat([x_r, x_i], dim=1))
        return x + self.res_scale * y


class ComplexPhaseStem(nn.Module):
    def __init__(
        self,
        phase_channels,
        eps=1e-5,
        gate_cond_channels=None,
        gate_scale=1.0,
        freq_stride=2,
        use_post_conv=False,
    ):
        super().__init__()
        self.conv = ComplexConv2dNoBias(
            1,
            phase_channels,
            kernel_size=(1, 3),
            stride=(1, freq_stride),
            padding=(0, 1),
        )
        self.norm = ComplexRMSNorm2D(phase_channels, eps=eps)
        self.post = None
        if use_post_conv:
            self.post = nn.Sequential(
                ComplexConv2dNoBias(phase_channels, phase_channels, kernel_size=3, padding=1),
                ComplexRMSNorm2D(phase_channels, eps=eps),
            )
        self.gate = None
        if gate_cond_channels is not None:
            self.gate = ComplexPointwiseGate2D(gate_cond_channels, phase_channels, scale=gate_scale)

    def forward(self, noisy_pha_4d, cond=None):
        x_r = torch.cos(noisy_pha_4d)
        x_i = torch.sin(noisy_pha_4d)
        y_r, y_i = self.conv(x_r, x_i)
        y_r, y_i = self.norm(y_r, y_i)
        if self.post is not None:
            y_r, y_i = self.post(y_r, y_i)
        if self.gate is not None and cond is not None:
            mod = self.gate(cond)
            y_r = y_r * mod
            y_i = y_i * mod
        return y_r, y_i


class ComplexPatchEmbed2D(nn.Module):
    def __init__(self, channels, eps=1e-5, use_mag_gate=False):
        super().__init__()
        self.conv = ComplexConv2dNoBias(channels, channels, kernel_size=3, padding=1)
        self.norm = ComplexRMSNorm2D(channels, eps=eps)
        self.use_mag_gate = use_mag_gate
        if use_mag_gate:
            self.mag_gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x_r, x_i):
        y_r, y_i = self.conv(x_r, x_i)
        y_r, y_i = self.norm(y_r, y_i)
        if self.use_mag_gate:
            mag = torch.sqrt(y_r ** 2 + y_i ** 2 + 1e-8)
            gate = torch.sigmoid(self.mag_gate(mag))
            y_r = y_r * gate
            y_i = y_i * gate
        return y_r, y_i


class ComplexPhaseRefineBlock(nn.Module):
    def __init__(
        self,
        channels,
        eps=1e-5,
        use_pha_ffn=False,
        ffn_expansion=4,
        ffn_dropout=0.0,
        ffn_use_complex_conv=True,
        ffn_kernel_size=3,
        ffn_res_scale=1.0,
    ):
        super().__init__()
        self.conv = ComplexConv2dNoBias(channels, channels, kernel_size=3, padding=1)
        self.norm = ComplexRMSNorm2D(channels, eps=eps)
        self.use_pha_ffn = use_pha_ffn
        if use_pha_ffn:
            self.ffn = ComplexPhaFFN1D(
                channels=channels,
                expansion=ffn_expansion,
                dropout=ffn_dropout,
                norm_epsilon=eps,
                use_complex_conv=ffn_use_complex_conv,
                kernel_size=ffn_kernel_size,
            )
            self.ffn_res_scale = nn.Parameter(torch.tensor(ffn_res_scale))

    def forward(self, x_r, x_i):
        y_r, y_i = self.conv(x_r, x_i)
        y_r, y_i = self.norm(y_r, y_i)
        if self.use_pha_ffn:
            b, c, t, f = y_r.shape
            y_r_seq = rearrange(y_r, 'b c t f -> b (t f) c')
            y_i_seq = rearrange(y_i, 'b c t f -> b (t f) c')
            z_r, z_i = self.ffn(y_r_seq, y_i_seq)
            z_r = rearrange(z_r, 'b (t f) c -> b c t f', t=t, f=f)
            z_i = rearrange(z_i, 'b (t f) c -> b c t f', t=t, f=f)
            y_r = y_r + self.ffn_res_scale * z_r
            y_i = y_i + self.ffn_res_scale * z_i
        return y_r, y_i


class ComplexPhaseDecoderHead(nn.Module):
    def __init__(self, channels, eps=1e-8, upsample_mode="bilinear"):
        super().__init__()
        self.eps = eps
        self.upsample_mode = upsample_mode
        self.head = ComplexConv2dNoBias(channels, 1, kernel_size=1)

    def forward(self, x_r, x_i, target_size):
        out_r, out_i = self.head(x_r, x_i)
        pred_cos = out_r.squeeze(1)
        pred_sin = out_i.squeeze(1)

        if pred_cos.shape[-2:] != target_size:
            align = False if self.upsample_mode in ["bilinear", "bicubic"] else None
            pred_cos = F.interpolate(
                pred_cos.unsqueeze(1),
                size=target_size,
                mode=self.upsample_mode,
                align_corners=align,
            ).squeeze(1)
            pred_sin = F.interpolate(
                pred_sin.unsqueeze(1),
                size=target_size,
                mode=self.upsample_mode,
                align_corners=align,
            ).squeeze(1)

        norm = torch.sqrt(pred_cos ** 2 + pred_sin ** 2 + self.eps)
        pred_cos = pred_cos / norm
        pred_sin = pred_sin / norm
        return pred_cos, pred_sin


class MambaSEUNet(nn.Module):
    """
    解耦双塔 Mamba 语音增强模型 (Research Version)

    架构逻辑：
    1. Magnitude Tower (Mag): 采用 FMamba，固定时间轴观察频率轴，学习谱包络（谐波、共振峰）。
    2. Phase Tower (Pha): 采用 TMamba，固定频率轴观察时间轴，学习相位的时域演变（连续性）。
    """

    def __init__(self, cfg):
        super(MambaSEUNet, self).__init__()
        self.cfg = cfg
        self.num_tscblocks = cfg['model_cfg'].get('num_tfmamba', 4)
        self.num_mid_pairs = int(cfg['model_cfg'].get('num_mid_pairs', 2))
        self.num_mid_pairs = max(1, min(4, self.num_mid_pairs))  # 保证1-4范围
        self.num_mid_stages = self.num_mid_pairs * 2  # 每对包含2次交替，需2个fusion
        # 交叉注意力稀疏/局部窗口配置
        self.cross_sparse = False
        self.cross_sparse_window = cfg['model_cfg'].get('cross_sparse_window', 64)
        self.cross_global_window = cfg['model_cfg'].get('cross_global_window', 8)
        self.cross_sparsity = cfg['model_cfg'].get('cross_sparsity', 0.9)

        # 维度设置: Mag 保持原始，Pha 减半
        mag_base = cfg['model_cfg']['hid_feature']
        pha_base = max(1, mag_base // 2)
        self.mag_dim = [mag_base, mag_base * 2, mag_base * 3]
        self.pha_dim = [pha_base, pha_base * 2, pha_base * 3]  # 约等于 [H/2, H, 1.5H]
        mag_dim, pha_dim = self.mag_dim, self.pha_dim
        self.cross_pool_t = cfg['model_cfg'].get('cross_pool_t', 1)
        self.cross_pool_f = cfg['model_cfg'].get('cross_pool_f', 1)
        self.cross_pool_t_final = cfg['model_cfg'].get('cross_pool_t_final', self.cross_pool_t)
        self.cross_pool_f_final = cfg['model_cfg'].get('cross_pool_f_final', self.cross_pool_f)

        # --- 1. 初始化输入配置 ---
        mag_cfg = deepcopy(cfg)
        mag_cfg['model_cfg']['input_channel'] = 2
        mag_cfg['model_cfg']['hid_feature'] = mag_base


        # --- 2. Magnitude Tower 模块定义 (频域建模) ---
        self.mag_encoder = DenseEncoder(mag_cfg)
        # Encoder 路径
        self.mag_patch_embed_encoder_level1 = Patch_Embed_stage(mag_dim[0], mag_dim[0])
        self.mag_TSMamba1_encoder = nn.ModuleList([TFMambaBlock(cfg, mag_dim[0]) for _ in range(self.num_tscblocks)])
        self.mag_down1_2 = Downsample(mag_dim[0], mag_dim[1])

        self.mag_patch_embed_encoder_level2 = Patch_Embed_stage(mag_dim[1], mag_dim[1])
        self.mag_TSMamba2_encoder = nn.ModuleList([TFMambaBlock(cfg, mag_dim[1]) for _ in range(self.num_tscblocks)])
        self.mag_down2_3 = Downsample(mag_dim[1], mag_dim[2])

        # Bottleneck 中间层
        self.mag_patch_embed_middle = Patch_Embed_stage(mag_dim[2], mag_dim[2])
        self.mag_FM_middle = nn.ModuleList([FMambaBlock(cfg, mag_dim[2]) for _ in range(self.num_mid_pairs)])
        self.mag_TM_middle = nn.ModuleList([TMambaBlock(cfg, mag_dim[2]) for _ in range(self.num_mid_pairs)])

        # Decoder 路径
        self.mag_up3_2 = Upsample(mag_dim[2], mag_dim[1])
        self.mag_concat_level2 = nn.Sequential(nn.Conv2d(mag_dim[1] * 2, mag_dim[1], 1, 1, 0, bias=False))
        self.mag_patch_embed_decoder_level2 = Patch_Embed_stage(mag_dim[1], mag_dim[1])
        self.mag_TSMamba2_decoder = nn.ModuleList([TFMambaBlock(cfg, mag_dim[1]) for _ in range(self.num_tscblocks)])

        self.mag_up2_1 = Upsample(mag_dim[1], mag_dim[0])
        self.mag_concat_level1 = nn.Sequential(nn.Conv2d(mag_dim[0] * 2, mag_dim[0], 1, 1, 0, bias=False))
        self.mag_patch_embed_decoder_level1 = Patch_Embed_stage(mag_dim[0], mag_dim[0])
        self.mag_TSMamba1_decoder = nn.ModuleList([TFMambaBlock(cfg, mag_dim[0]) for _ in range(self.num_tscblocks)])

        # Refinement 细化层
        self.mag_patch_embed_refinement = Patch_Embed_stage(mag_dim[0], mag_dim[0])
        self.mag_refinement = nn.ModuleList([TFMambaBlock(cfg, mag_dim[0]) for _ in range(self.num_tscblocks)])
        self.mag_output = nn.Sequential(nn.Conv2d(mag_dim[0], mag_dim[0], 3, 1, 1, bias=False))

        # --- 3. Phase Tower 模块定义 (full complex GRE-safe) ---
        self.pha_complex_stem = ComplexPhaseStem(
            phase_channels=pha_dim[0],
            freq_stride=cfg['model_cfg'].get('full_phase_stem_freq_stride', 2),
            use_post_conv=cfg['model_cfg'].get('full_phase_stem_use_post_conv', False),
            eps=cfg['model_cfg'].get('norm_epsilon', 1e-5),
        )
        self.pha_complex_patch_embed_encoder_level1 = ComplexPatchEmbed2D(pha_dim[0])
        self.pha_complex_TSMamba1_encoder = nn.ModuleList([
            ComplexEqTMambaBlock(cfg, pha_dim[0]) for _ in range(self.num_tscblocks)
        ])
        self.pha_complex_down1_2 = ComplexDownsample2D(pha_dim[0], pha_dim[1])

        self.pha_complex_patch_embed_encoder_level2 = ComplexPatchEmbed2D(pha_dim[1])
        self.pha_complex_TSMamba2_encoder = nn.ModuleList([
            ComplexEqTMambaBlock(cfg, pha_dim[1]) for _ in range(self.num_tscblocks)
        ])
        self.pha_complex_down2_3 = ComplexDownsample2D(pha_dim[1], pha_dim[2])
        self.pha_complex_patch_embed_middle = ComplexPatchEmbed2D(pha_dim[2])

        self.pha_TM_middle = nn.ModuleList([
            ComplexEqTMambaBlock(cfg, pha_dim[2]) for _ in range(self.num_mid_pairs)
        ])
        self.pha_FM_middle = nn.ModuleList([
            ComplexEqFMambaBlock(cfg, pha_dim[2]) for _ in range(self.num_mid_pairs)
        ])

        self.pha_complex_up3_2 = ComplexUpsample2D(pha_dim[2], pha_dim[1])
        self.pha_complex_concat_level2 = ComplexConcatFuse2D(pha_dim[1] * 2, pha_dim[1])
        self.pha_complex_patch_embed_decoder_level2 = ComplexPatchEmbed2D(pha_dim[1])
        self.pha_complex_TSMamba2_decoder = nn.ModuleList([
            ComplexEqTMambaBlock(cfg, pha_dim[1]) for _ in range(self.num_tscblocks)
        ])

        self.pha_complex_up2_1 = ComplexUpsample2D(pha_dim[1], pha_dim[0])
        self.pha_complex_concat_level1 = ComplexConcatFuse2D(pha_dim[0] * 2, pha_dim[0])
        self.pha_complex_patch_embed_decoder_level1 = ComplexPatchEmbed2D(pha_dim[0])
        self.pha_complex_TSMamba1_decoder = nn.ModuleList([
            ComplexEqTMambaBlock(cfg, pha_dim[0]) for _ in range(self.num_tscblocks)
        ])

        self.pha_complex_patch_embed_refinement = ComplexPatchEmbed2D(pha_dim[0])
        self.pha_complex_refinement = nn.ModuleList([
            ComplexEqTMambaBlock(cfg, pha_dim[0]) for _ in range(self.num_tscblocks)
        ])
        self.pha_complex_output = ComplexPhaseRefineBlock(
            pha_dim[0],
            eps=cfg['model_cfg'].get('norm_epsilon', 1e-5),
            use_pha_ffn=cfg['model_cfg'].get('use_complex_phase_refine_ffn', True),
            ffn_expansion=cfg['model_cfg'].get('pha_ffn_expansion', 4),
            ffn_dropout=cfg['model_cfg'].get('pha_ffn_dropout', 0.0),
            ffn_use_complex_conv=cfg['model_cfg'].get('pha_ffn_use_complex_conv', True),
            ffn_kernel_size=cfg['model_cfg'].get('pha_ffn_kernel_size', 3),
            ffn_res_scale=cfg['model_cfg'].get('pha_ffn_res_scale', 1.0),
        )
        self.gre_final_fusion = GREVSSFinalFusion(
            hidden_dim=mag_dim[0],
            phase_channels=pha_dim[0],
            cfg=cfg,
            d_state=cfg['model_cfg'].get('d_state', 16),
        )
        self.complex_phase_head = ComplexPhaseDecoderHead(
            pha_dim[0],
            eps=1e-8,
            upsample_mode=cfg['model_cfg'].get('phase_head_upsample_mode', 'bilinear'),
        )
        # --- 4. 最终解码器 ---
        self.mag_to_mask_proj = nn.Conv2d(mag_dim[0], mag_base, 1, 1, 0, bias=False)
        self.mask_decoder = MagDecoder(cfg)

    def forward(self, noisy_mag, noisy_pha):
        if not torch.isfinite(noisy_mag).all():
             raise RuntimeError('Input noisy_mag contains NaN/Inf')
        if not torch.isfinite(noisy_pha).all():
             raise RuntimeError('Input noisy_pha contains NaN/Inf')

        # [B, F, T] -> [B, 1, T, F]
        noisy_mag_4d = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)
        noisy_pha_4d = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)

        # 双塔均使用原始 cat 输入
        mag_in = torch.cat((noisy_mag_4d, noisy_pha_4d), dim=1)

        # ---------------------------
        # Magnitude Tower Encoder
        # ---------------------------
        mag_x1 = self.mag_encoder(mag_in)
        mag_copy1 = mag_x1
        mag_x1 = self.mag_patch_embed_encoder_level1(mag_x1)
        for block in self.mag_TSMamba1_encoder:
            mag_x1 = block(mag_x1)
        mag_x1 = mag_copy1 + mag_x1
        mag_skip1 = mag_x1

        mag_x2 = self.mag_down1_2(mag_x1)
        mag_copy2 = mag_x2
        mag_x2 = self.mag_patch_embed_encoder_level2(mag_x2)
        for block in self.mag_TSMamba2_encoder:
            mag_x2 = block(mag_x2)
        mag_x2 = mag_copy2 + mag_x2
        mag_skip2 = mag_x2

        mag_x3 = self.mag_down2_3(mag_x2)
        mag_x3 = self.mag_patch_embed_middle(mag_x3)
        # Bottleneck without middle cross fusion.
        # Magnitude and phase streams are modeled independently here.
        # Cross-stream interaction is only performed at the final GREVSSFinalFusion.

        pha_r1, pha_i1 = self.pha_complex_stem(noisy_pha_4d)
        pha_copy1_r, pha_copy1_i = pha_r1, pha_i1
        pha_r1, pha_i1 = self.pha_complex_patch_embed_encoder_level1(pha_r1, pha_i1)
        for block in self.pha_complex_TSMamba1_encoder:
            pha_r1, pha_i1 = block(pha_r1, pha_i1)
        pha_r1 = pha_copy1_r + pha_r1
        pha_i1 = pha_copy1_i + pha_i1
        pha_skip1_r, pha_skip1_i = pha_r1, pha_i1

        pha_r2, pha_i2 = self.pha_complex_down1_2(pha_r1, pha_i1)
        pha_copy2_r, pha_copy2_i = pha_r2, pha_i2
        pha_r2, pha_i2 = self.pha_complex_patch_embed_encoder_level2(pha_r2, pha_i2)
        for block in self.pha_complex_TSMamba2_encoder:
            pha_r2, pha_i2 = block(pha_r2, pha_i2)
        pha_r2 = pha_copy2_r + pha_r2
        pha_i2 = pha_copy2_i + pha_i2
        pha_skip2_r, pha_skip2_i = pha_r2, pha_i2

        pha_r3, pha_i3 = self.pha_complex_down2_3(pha_r2, pha_i2)
        pha_r3, pha_i3 = self.pha_complex_patch_embed_middle(pha_r3, pha_i3)
        for idx in range(self.num_mid_stages):
            pair_idx = idx // 2
            if idx % 2 == 0:
                mag_x3 = self.mag_FM_middle[pair_idx](mag_x3)
                pha_r3, pha_i3 = self.pha_TM_middle[pair_idx](pha_r3, pha_i3)
            else:
                mag_x3 = self.mag_TM_middle[pair_idx](mag_x3)
                pha_r3, pha_i3 = self.pha_FM_middle[pair_idx](pha_r3, pha_i3)

        # ---------------------------
        # Decoder Level2 (after upsample+concat -> mamba -> residual)
        # ---------------------------
        mag_y2 = self.mag_up3_2(mag_x3)
        mag_y2 = torch.cat([mag_y2, mag_skip2], 1)
        mag_y2 = self.mag_concat_level2(mag_y2)
        mag_y2_copy = mag_y2
        mag_y2 = self.mag_patch_embed_decoder_level2(mag_y2)
        for block in self.mag_TSMamba2_decoder:
            mag_y2 = block(mag_y2)
        mag_y2 = mag_y2_copy + mag_y2

        pha_r2d, pha_i2d = self.pha_complex_up3_2(pha_r3, pha_i3)
        pha_r2d, pha_i2d = self.pha_complex_concat_level2(
            pha_r2d,
            pha_i2d,
            pha_skip2_r,
            pha_skip2_i,
        )
        copy_r, copy_i = pha_r2d, pha_i2d
        pha_r2d, pha_i2d = self.pha_complex_patch_embed_decoder_level2(pha_r2d, pha_i2d)
        for block in self.pha_complex_TSMamba2_decoder:
            pha_r2d, pha_i2d = block(pha_r2d, pha_i2d)
        pha_r2d = copy_r + pha_r2d
        pha_i2d = copy_i + pha_i2d

        # ---------------------------
        # Decoder Level1（已移除交叉注意力）
        # ---------------------------
        mag_y1 = self.mag_up2_1(mag_y2)
        mag_y1 = torch.cat([mag_y1, mag_skip1], 1)
        mag_y1 = self.mag_concat_level1(mag_y1)
        mag_y1_copy = mag_y1
        mag_y1 = self.mag_patch_embed_decoder_level1(mag_y1)
        for block in self.mag_TSMamba1_decoder:
            mag_y1 = block(mag_y1)
        mag_y1 = mag_y1_copy + mag_y1

        pha_r1d, pha_i1d = self.pha_complex_up2_1(pha_r2d, pha_i2d)
        pha_r1d, pha_i1d = self.pha_complex_concat_level1(
            pha_r1d,
            pha_i1d,
            pha_skip1_r,
            pha_skip1_i,
        )
        copy_r, copy_i = pha_r1d, pha_i1d
        pha_r1d, pha_i1d = self.pha_complex_patch_embed_decoder_level1(pha_r1d, pha_i1d)
        for block in self.pha_complex_TSMamba1_decoder:
            pha_r1d, pha_i1d = block(pha_r1d, pha_i1d)
        pha_r1d = copy_r + pha_r1d
        pha_i1d = copy_i + pha_i1d

        # Mag Refinement & Output
        mag_copy_ref = mag_y1
        mag_y1 = self.mag_patch_embed_refinement(mag_y1)
        for block in self.mag_refinement:
            mag_y1 = block(mag_y1)
        mag_final = self.mag_output(mag_y1 + mag_copy_ref) + mag_skip1

        copy_r, copy_i = pha_r1d, pha_i1d
        pha_r1d, pha_i1d = self.pha_complex_patch_embed_refinement(pha_r1d, pha_i1d)
        for block in self.pha_complex_refinement:
            pha_r1d, pha_i1d = block(pha_r1d, pha_i1d)
        pha_r_final = pha_r1d + copy_r + pha_skip1_r
        pha_i_final = pha_i1d + copy_i + pha_skip1_i
        pha_r_final, pha_i_final = self.pha_complex_output(pha_r_final, pha_i_final)

        mag_fused, pha_r_fused, pha_i_fused = self.gre_final_fusion(
            mag_final,
            pha_r_final,
            pha_i_final,
        )

        target_size = noisy_pha_4d.shape[-2:]
        pred_cos_t_f, pred_sin_t_f = self.complex_phase_head(
            pha_r_fused,
            pha_i_fused,
            target_size=target_size,
        )
        pred_cos = rearrange(pred_cos_t_f, 'b t f -> b f t')
        pred_sin = rearrange(pred_sin_t_f, 'b t f -> b f t')
        pred_pha = torch.atan2(pred_sin, pred_cos)

        mag_mask = self.mask_decoder(self.mag_to_mask_proj(mag_fused))
        if not torch.isfinite(mag_mask).all():
            raise RuntimeError('mag_mask contains NaN/Inf')
        denoised_mag = rearrange(mag_mask * noisy_mag_4d, 'b c t f -> b f t c').squeeze(-1)

        if pred_cos.shape != denoised_mag.shape:
            raise RuntimeError(
                f"phase/magnitude output mismatch: pred_cos={pred_cos.shape}, "
                f"denoised_mag={denoised_mag.shape}"
            )

        denoised_com = torch.stack(
            (denoised_mag * pred_cos,
             denoised_mag * pred_sin),
            dim=-1
        )
        if not torch.isfinite(denoised_com).all():
            raise RuntimeError('denoised_com contains NaN/Inf')

        return denoised_mag, pred_pha, denoised_com
