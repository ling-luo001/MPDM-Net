# Reference: https://github.com/huaidanquede/MUSE-Speech-Enhancement/tree/main/models/generator

import torch
import torch.nn as nn
import math
from torchvision.ops.deform_conv import DeformConv2d
from einops import rearrange
from copy import deepcopy
from .mamba_block import TMambaBlock, FMambaBlock, TFMambaBlock
from .codec_module import DenseEncoder, MagDecoder, PhaseDecoder
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


class MambaSEUNet(nn.Module):
    """
    Progressive suppression-restoration Mamba speech enhancement model.

    Stage 1 predicts a coarse complex spectrum from a magnitude mask and a
    lightweight phase rotation. Stage 2 receives both the original and coarse
    complex spectra, then predicts a gated complex residual. Suppression
    features guide restoration through a one-way bottleneck connection.
    """

    def __init__(self, cfg):
        super(MambaSEUNet, self).__init__()
        self.cfg = cfg
        self.num_tscblocks = cfg['model_cfg'].get('num_tfmamba', 4)
        self.num_mid_pairs = int(cfg['model_cfg'].get('num_mid_pairs', 2))
        self.num_mid_pairs = max(1, min(4, self.num_mid_pairs))

        # Keep suppression full-width and use a narrower restoration tower.
        mag_base = cfg['model_cfg']['hid_feature']
        restore_width_ratio = float(cfg['model_cfg'].get('restoration_width_ratio', 0.5))
        if not 0.0 < restore_width_ratio <= 1.0:
            raise ValueError('restoration_width_ratio must be in (0, 1]')
        restore_base = max(1, int(round(mag_base * restore_width_ratio)))
        if restore_base % 4 != 0:
            raise ValueError(
                'The restoration base width must be divisible by 4 for PixelUnshuffle.'
            )
        self.mag_dim = [mag_base, mag_base * 2, mag_base * 3]
        self.restore_dim = [restore_base, restore_base * 2, restore_base * 3]
        mag_dim, restore_dim = self.mag_dim, self.restore_dim

        # --- 1. 初始化输入配置 ---
        mag_cfg = deepcopy(cfg)
        mag_cfg['model_cfg']['input_channel'] = 2
        mag_cfg['model_cfg']['hid_feature'] = mag_base

        restore_cfg = deepcopy(cfg)
        restore_cfg['model_cfg']['input_channel'] = 4
        restore_cfg['model_cfg']['hid_feature'] = restore_base

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

        # --- 3. Narrow complex restoration tower ---
        self.restore_encoder = DenseEncoder(restore_cfg)
        self.restore_patch_embed_encoder_level1 = Patch_Embed_stage(restore_dim[0], restore_dim[0])
        self.restore_TMamba1_encoder = nn.ModuleList(
            [TMambaBlock(cfg, restore_dim[0]) for _ in range(self.num_tscblocks)]
        )
        self.restore_down1_2 = Downsample(restore_dim[0], restore_dim[1])

        self.restore_patch_embed_encoder_level2 = Patch_Embed_stage(restore_dim[1], restore_dim[1])
        self.restore_TMamba2_encoder = nn.ModuleList(
            [TMambaBlock(cfg, restore_dim[1]) for _ in range(self.num_tscblocks)]
        )
        self.restore_down2_3 = Downsample(restore_dim[1], restore_dim[2])

        self.restore_patch_embed_middle = Patch_Embed_stage(restore_dim[2], restore_dim[2])
        self.restore_TM_middle = nn.ModuleList(
            [TMambaBlock(cfg, restore_dim[2]) for _ in range(self.num_mid_pairs)]
        )
        self.restore_FM_middle = nn.ModuleList(
            [FMambaBlock(cfg, restore_dim[2]) for _ in range(self.num_mid_pairs)]
        )

        self.restore_up3_2 = Upsample(restore_dim[2], restore_dim[1])
        self.restore_concat_level2 = nn.Conv2d(restore_dim[1] * 2, restore_dim[1], 1, 1, 0, bias=False)
        self.restore_patch_embed_decoder_level2 = Patch_Embed_stage(restore_dim[1], restore_dim[1])
        self.restore_TMamba2_decoder = nn.ModuleList(
            [TMambaBlock(cfg, restore_dim[1]) for _ in range(self.num_tscblocks)]
        )

        self.restore_up2_1 = Upsample(restore_dim[1], restore_dim[0])
        self.restore_concat_level1 = nn.Conv2d(restore_dim[0] * 2, restore_dim[0], 1, 1, 0, bias=False)
        self.restore_patch_embed_decoder_level1 = Patch_Embed_stage(restore_dim[0], restore_dim[0])
        self.restore_TMamba1_decoder = nn.ModuleList(
            [TMambaBlock(cfg, restore_dim[0]) for _ in range(self.num_tscblocks)]
        )

        self.restore_patch_embed_refinement = Patch_Embed_stage(restore_dim[0], restore_dim[0])
        self.restore_refinement = nn.ModuleList(
            [TMambaBlock(cfg, restore_dim[0]) for _ in range(self.num_tscblocks)]
        )
        self.restore_output = nn.Conv2d(restore_dim[0], restore_dim[0], 3, 1, 1, bias=False)

        # One-way suppression context. Zero initialization lets restoration
        # first learn from [X, S0], then opt into bottleneck guidance.
        self.suppress_to_restore = nn.Sequential(
            nn.Conv2d(mag_dim[2], restore_dim[2], 1, 1, 0, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=restore_dim[2]),
        )
        self.suppress_context_scale = nn.Parameter(torch.zeros(()))

        # --- 4. Coarse and restoration output heads ---
        self.mag_to_mask_proj = nn.Conv2d(mag_dim[0], mag_base, 1, 1, 0, bias=False)
        self.mask_decoder = MagDecoder(cfg)
        coarse_phase_cfg = deepcopy(cfg)
        coarse_phase_cfg['model_cfg']['hid_feature'] = mag_base
        self.coarse_phase_decoder = PhaseDecoder(coarse_phase_cfg)
        nn.init.zeros_(self.coarse_phase_decoder.phase_conv_out.weight)
        nn.init.zeros_(self.coarse_phase_decoder.phase_conv_out.bias)
        with torch.no_grad():
            self.coarse_phase_decoder.phase_conv_out.bias[0] = 1.0

        self.phase_eps = cfg['model_cfg'].get('phase_eps', 1e-3)
        self.complex_residual_gate_bias = cfg['model_cfg'].get('complex_residual_gate_bias', -2.0)
        self.restoration_residual_decoder = nn.Sequential(
            nn.Conv2d(restore_dim[0], restore_dim[0] * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(
                restore_dim[0],
                restore_dim[0],
                kernel_size=(1, 3),
                stride=(2, 1),
                padding=(0, 1),
                groups=restore_dim[0],
                bias=False
            ),
            nn.InstanceNorm2d(restore_dim[0], affine=True),
            nn.PReLU(restore_dim[0]),
            nn.Conv2d(restore_dim[0], cfg['model_cfg']['output_channel'] * 2, (1, 1))
        )
        nn.init.zeros_(self.restoration_residual_decoder[-1].weight)
        nn.init.zeros_(self.restoration_residual_decoder[-1].bias)
        self.restoration_gate = nn.Sequential(
            nn.Conv2d(restore_dim[0], restore_dim[0] * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(
                restore_dim[0],
                restore_dim[0],
                kernel_size=(1, 3),
                stride=(2, 1),
                padding=(0, 1),
                groups=restore_dim[0],
                bias=False
            ),
            nn.InstanceNorm2d(restore_dim[0], affine=True),
            nn.PReLU(restore_dim[0]),
            nn.Conv2d(restore_dim[0], cfg['model_cfg']['output_channel'], (1, 1))
        )
        nn.init.zeros_(self.restoration_gate[-1].weight)
        nn.init.constant_(self.restoration_gate[-1].bias, self.complex_residual_gate_bias)
        self.latest_aux = {}

    def forward(self, noisy_mag, noisy_pha):
        if noisy_mag.ndim != 3 or noisy_pha.ndim != 3:
            raise ValueError('Expected noisy_mag and noisy_pha with shape [B, F, T].')
        if noisy_mag.shape != noisy_pha.shape:
            raise ValueError(
                f'Input shapes differ: {tuple(noisy_mag.shape)} vs {tuple(noisy_pha.shape)}'
            )
        encoded_freq_bins = (noisy_mag.shape[1] + 1) // 2
        if noisy_mag.shape[2] % 4 != 0 or encoded_freq_bins % 4 != 0:
            raise ValueError(
                'Time frames and encoded frequency bins must be divisible by 4; '
                f'got T={noisy_mag.shape[2]}, encoded F={encoded_freq_bins}.'
            )
        if not torch.isfinite(noisy_mag).all():
            raise RuntimeError('Input noisy_mag contains NaN/Inf')
        if not torch.isfinite(noisy_pha).all():
            raise RuntimeError('Input noisy_pha contains NaN/Inf')

        # [B, F, T] -> [B, 1, T, F]
        noisy_mag_4d = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)
        noisy_pha_4d = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)
        # Joint magnitude/phase input for coarse suppression.
        mag_in = torch.cat((noisy_mag_4d, noisy_pha_4d), dim=1)

        # ---------------------------
        # Stage 1: coarse suppression encoder
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
        for fm_block, tm_block in zip(self.mag_FM_middle, self.mag_TM_middle):
            mag_x3 = fm_block(mag_x3)
            mag_x3 = tm_block(mag_x3)
        suppress_bottleneck = mag_x3

        # Stage 1 decoder and coarse complex-spectrum reconstruction.
        mag_y2 = self.mag_up3_2(mag_x3)
        mag_y2 = self.mag_concat_level2(torch.cat([mag_y2, mag_skip2], dim=1))
        mag_y2_copy = mag_y2
        mag_y2 = self.mag_patch_embed_decoder_level2(mag_y2)
        for block in self.mag_TSMamba2_decoder:
            mag_y2 = block(mag_y2)
        mag_y2 = mag_y2_copy + mag_y2

        # Decode to the original encoder resolution.
        mag_y1 = self.mag_up2_1(mag_y2)
        mag_y1 = self.mag_concat_level1(torch.cat([mag_y1, mag_skip1], dim=1))
        mag_y1_copy = mag_y1
        mag_y1 = self.mag_patch_embed_decoder_level1(mag_y1)
        for block in self.mag_TSMamba1_decoder:
            mag_y1 = block(mag_y1)
        mag_y1 = mag_y1_copy + mag_y1

        # Refine Stage 1 features before coarse reconstruction.
        mag_copy_ref = mag_y1
        mag_y1 = self.mag_patch_embed_refinement(mag_y1)
        for block in self.mag_refinement:
            mag_y1 = block(mag_y1)
        mag_final = self.mag_output(mag_y1 + mag_copy_ref) + mag_skip1

        # Coarse signal reconstruction.
        mag_mask = self.mask_decoder(self.mag_to_mask_proj(mag_final))
        if not torch.isfinite(mag_mask).all():
            raise RuntimeError('mag_mask contains NaN/Inf')
        coarse_mag_4d = mag_mask * noisy_mag_4d

        # Predict a unit complex rotation and apply it on the noisy phase unit vector.
        rot_vec = self.coarse_phase_decoder(mag_final)
        rot_vec = F.normalize(rot_vec, dim=1, p=2, eps=self.phase_eps)
        delta_cos, delta_sin = torch.chunk(rot_vec, 2, dim=1)

        noisy_cos = torch.cos(noisy_pha_4d)
        noisy_sin = torch.sin(noisy_pha_4d)

        coarse_cos = noisy_cos * delta_cos - noisy_sin * delta_sin
        coarse_sin = noisy_sin * delta_cos + noisy_cos * delta_sin

        coarse_real_4d = coarse_mag_4d * coarse_cos
        coarse_imag_4d = coarse_mag_4d * coarse_sin
        if not torch.isfinite(coarse_real_4d).all() or not torch.isfinite(coarse_imag_4d).all():
            raise RuntimeError('Coarse complex spectrum contains NaN/Inf')

        # Stage 2 restores details from the original and coarse complex spectra.
        noisy_real_4d = noisy_mag_4d * noisy_cos
        noisy_imag_4d = noisy_mag_4d * noisy_sin
        restore_in = torch.cat(
            [noisy_real_4d, noisy_imag_4d, coarse_real_4d, coarse_imag_4d],
            dim=1
        )

        restore_x1 = self.restore_encoder(restore_in)
        restore_copy1 = restore_x1
        restore_x1 = self.restore_patch_embed_encoder_level1(restore_x1)
        for block in self.restore_TMamba1_encoder:
            restore_x1 = block(restore_x1)
        restore_x1 = restore_copy1 + restore_x1
        restore_skip1 = restore_x1

        restore_x2 = self.restore_down1_2(restore_x1)
        restore_copy2 = restore_x2
        restore_x2 = self.restore_patch_embed_encoder_level2(restore_x2)
        for block in self.restore_TMamba2_encoder:
            restore_x2 = block(restore_x2)
        restore_x2 = restore_copy2 + restore_x2
        restore_skip2 = restore_x2

        restore_x3 = self.restore_down2_3(restore_x2)
        restore_x3 = self.restore_patch_embed_middle(restore_x3)
        suppression_context = self.suppress_to_restore(suppress_bottleneck)
        if suppression_context.shape != restore_x3.shape:
            raise RuntimeError(
                'Suppression/restoration bottleneck shapes differ: '
                f'{tuple(suppression_context.shape)} vs {tuple(restore_x3.shape)}'
            )
        context_scale = torch.tanh(self.suppress_context_scale)
        restore_x3 = restore_x3 + context_scale * suppression_context
        for tm_block, fm_block in zip(self.restore_TM_middle, self.restore_FM_middle):
            restore_x3 = tm_block(restore_x3)
            restore_x3 = fm_block(restore_x3)

        restore_y2 = self.restore_up3_2(restore_x3)
        restore_y2 = self.restore_concat_level2(
            torch.cat([restore_y2, restore_skip2], dim=1)
        )
        restore_y2_copy = restore_y2
        restore_y2 = self.restore_patch_embed_decoder_level2(restore_y2)
        for block in self.restore_TMamba2_decoder:
            restore_y2 = block(restore_y2)
        restore_y2 = restore_y2_copy + restore_y2

        restore_y1 = self.restore_up2_1(restore_y2)
        restore_y1 = self.restore_concat_level1(
            torch.cat([restore_y1, restore_skip1], dim=1)
        )
        restore_y1_copy = restore_y1
        restore_y1 = self.restore_patch_embed_decoder_level1(restore_y1)
        for block in self.restore_TMamba1_decoder:
            restore_y1 = block(restore_y1)
        restore_y1 = restore_y1_copy + restore_y1

        restore_copy_ref = restore_y1
        restore_y1 = self.restore_patch_embed_refinement(restore_y1)
        for block in self.restore_refinement:
            restore_y1 = block(restore_y1)
        restore_final = self.restore_output(restore_y1 + restore_copy_ref) + restore_skip1

        complex_residual = torch.tanh(self.restoration_residual_decoder(restore_final))
        if not torch.isfinite(complex_residual).all():
            raise RuntimeError('complex_residual contains NaN/Inf')
        restoration_gate = torch.sigmoid(self.restoration_gate(restore_final))
        if not torch.isfinite(restoration_gate).all():
            raise RuntimeError('restoration_gate contains NaN/Inf')

        residual_real_4d, residual_imag_4d = torch.chunk(complex_residual, 2, dim=1)
        reference_mag_4d = 0.5 * (noisy_mag_4d + coarse_mag_4d)
        applied_real_4d = restoration_gate * residual_real_4d * reference_mag_4d
        applied_imag_4d = restoration_gate * residual_imag_4d * reference_mag_4d

        coarse_real = rearrange(coarse_real_4d.squeeze(1), 'b t f -> b f t')
        coarse_imag = rearrange(coarse_imag_4d.squeeze(1), 'b t f -> b f t')
        applied_real = rearrange(applied_real_4d.squeeze(1), 'b t f -> b f t')
        applied_imag = rearrange(applied_imag_4d.squeeze(1), 'b t f -> b f t')
        enh_real = coarse_real + applied_real
        enh_imag = coarse_imag + applied_imag
        denoised_mag = torch.sqrt(torch.clamp(enh_real ** 2 + enh_imag ** 2, min=1e-12))
        if not torch.isfinite(denoised_mag).all():
            raise RuntimeError('denoised_mag contains NaN/Inf')
        phase_floor = torch.full_like(enh_real, self.phase_eps)
        phase_real = torch.where(denoised_mag.detach() < self.phase_eps, phase_floor, enh_real)
        pred_pha = torch.atan2(enh_imag, phase_real)
        if not torch.isfinite(pred_pha).all():
            raise RuntimeError('pred_pha contains NaN/Inf')

        denoised_com = torch.stack((enh_real, enh_imag), dim=-1)
        if not torch.isfinite(denoised_com).all():
            raise RuntimeError('denoised_com contains NaN/Inf')

        self.latest_aux = {
            'coarse_complex': torch.stack((coarse_real, coarse_imag), dim=-1),
            'complex_residual': complex_residual,
            'restoration_gate': restoration_gate,
            'complex_residual_applied': torch.stack((applied_real, applied_imag), dim=-1),
            'suppression_context_scale': context_scale,
        }
        return denoised_mag, pred_pha, denoised_com
