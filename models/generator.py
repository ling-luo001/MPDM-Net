# Reference: https://github.com/huaidanquede/MUSE-Speech-Enhancement/tree/main/models/generator

import torch
import torch.nn as nn
import math
from torchvision.ops.deform_conv import DeformConv2d
from einops import rearrange
from copy import deepcopy
from .mamba_block import TMambaBlock, FMambaBlock, TFMambaBlock, CBAM
from .cross import VSSBlock_Cross_new
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
    解耦双塔 Mamba 语音增强模型 (Research Version)

    架构逻辑：
    1. Magnitude Tower (Mag): 采用 FMamba，固定时间轴观察频率轴，学习谱包络（谐波、共振峰）。
    2. Phase Tower (Pha): 采用 TFMamba，联合时频建模相位演变（连续性）。
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

        # 维度设置: Mag 与 Pha 保持一致（不再减半，用于可比性）
        mag_base = cfg['model_cfg']['hid_feature']
        pha_base = mag_base
        self.mag_dim = [mag_base, mag_base * 2, mag_base * 3]
        self.pha_dim = [mag_base, mag_base * 2, mag_base * 3]
        mag_dim, pha_dim = self.mag_dim, self.pha_dim
        self.cross_pool_t = cfg['model_cfg'].get('cross_pool_t', 1)
        self.cross_pool_f = cfg['model_cfg'].get('cross_pool_f', 1)
        # 独立控制中间层与最终层的交叉注意力下采样，middle 默认不下采样
        self.cross_pool_t_mid = cfg['model_cfg'].get('cross_pool_t_mid', 1)
        self.cross_pool_f_mid = cfg['model_cfg'].get('cross_pool_f_mid', 1)
        self.cross_pool_t_final = cfg['model_cfg'].get('cross_pool_t_final', self.cross_pool_t)
        self.cross_pool_f_final = cfg['model_cfg'].get('cross_pool_f_final', self.cross_pool_f)

        # --- 1. 初始化输入配置 ---
        mag_cfg = deepcopy(cfg)
        mag_cfg['model_cfg']['input_channel'] = 2
        mag_cfg['model_cfg']['hid_feature'] = mag_base

        pha_cfg = deepcopy(cfg)
        pha_cfg['model_cfg']['input_channel'] = 2
        pha_cfg['model_cfg']['hid_feature'] = mag_base

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

        # --- 3. Phase Tower 模块定义 (时域建模) ---
        self.pha_encoder = DenseEncoder(pha_cfg)
        # Encoder 路径
        self.pha_patch_embed_encoder_level1 = Patch_Embed_stage(pha_dim[0], pha_dim[0])
        self.pha_TSMamba1_encoder = nn.ModuleList([TFMambaBlock(cfg, pha_dim[0]) for _ in range(self.num_tscblocks)])
        self.pha_down1_2 = Downsample(pha_dim[0], pha_dim[1])

        self.pha_patch_embed_encoder_level2 = Patch_Embed_stage(pha_dim[1], pha_dim[1])
        self.pha_TSMamba2_encoder = nn.ModuleList([TFMambaBlock(cfg, pha_dim[1]) for _ in range(self.num_tscblocks)])
        self.pha_down2_3 = Downsample(pha_dim[1], pha_dim[2])

        # Bottleneck 中间层
        self.pha_patch_embed_middle = Patch_Embed_stage(pha_dim[2], pha_dim[2])
        self.pha_TM_middle = nn.ModuleList([TMambaBlock(cfg, pha_dim[2]) for _ in range(self.num_mid_pairs)])
        self.pha_FM_middle = nn.ModuleList([FMambaBlock(cfg, pha_dim[2]) for _ in range(self.num_mid_pairs)])

        # Decoder 路径
        self.pha_up3_2 = Upsample(pha_dim[2], pha_dim[1])
        self.pha_concat_level2 = nn.Sequential(nn.Conv2d(pha_dim[1] * 2, pha_dim[1], 1, 1, 0, bias=False))
        self.pha_patch_embed_decoder_level2 = Patch_Embed_stage(pha_dim[1], pha_dim[1])
        self.pha_TSMamba2_decoder = nn.ModuleList([TFMambaBlock(cfg, pha_dim[1]) for _ in range(self.num_tscblocks)])

        self.pha_up2_1 = Upsample(pha_dim[1], pha_dim[0])
        self.pha_concat_level1 = nn.Sequential(nn.Conv2d(pha_dim[0] * 2, pha_dim[0], 1, 1, 0, bias=False))
        self.pha_patch_embed_decoder_level1 = Patch_Embed_stage(pha_dim[0], pha_dim[0])
        self.pha_TSMamba1_decoder = nn.ModuleList([TFMambaBlock(cfg, pha_dim[0]) for _ in range(self.num_tscblocks)])

        # Refinement 细化层
        self.pha_patch_embed_refinement = Patch_Embed_stage(pha_dim[0], pha_dim[0])
        self.pha_refinement = nn.ModuleList([TFMambaBlock(cfg, pha_dim[0]) for _ in range(self.num_tscblocks)])
        self.pha_output = nn.Sequential(nn.Conv2d(pha_dim[0], pha_dim[0], 3, 1, 1, bias=False))

        # --- 5. 双流 VSS 融合模块 ---
        # 1) 独立投影到高维 3H
        self.mid_in_proj_mag = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mag_dim[2], mag_dim[2], 1, 1, 0, bias=False),
                nn.GroupNorm(num_groups=1, num_channels=mag_dim[2])
            ) for _ in range(self.num_mid_stages)
        ])
        self.mid_in_proj_pha = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(pha_dim[2], mag_dim[2], 1, 1, 0, bias=False),
                nn.GroupNorm(num_groups=1, num_channels=mag_dim[2])
            ) for _ in range(self.num_mid_stages)
        ])
        # 2) VSS 交互（同维 3H）
        self.mid_fusions = nn.ModuleList([VSSBlock_Cross_new(hidden_dim=mag_dim[2]) for _ in range(self.num_mid_stages)])
        # 3) 融合后还原各自宽度
        self.mid_fusion_proj_mag = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mag_dim[2] * 2, mag_dim[2], 1, 1, 0, bias=False),
                nn.GroupNorm(num_groups=1, num_channels=mag_dim[2])
            ) for _ in range(self.num_mid_stages)
        ])
        self.mid_fusion_proj_pha = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(pha_dim[2] + mag_dim[2], pha_dim[2], 1, 1, 0, bias=False),
                nn.GroupNorm(num_groups=1, num_channels=pha_dim[2])
            ) for _ in range(self.num_mid_stages)
        ])
        # 全局融合前将 Pha 提升到 Mag 宽度
        self.global_in_proj_pha = nn.Sequential(
            nn.Conv2d(pha_dim[0], mag_dim[0], 1, 1, 0, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=mag_dim[0])
        )
        self.global_fusion = VSSBlock_Cross_new(hidden_dim=mag_dim[0])
        self.global_out_proj_pha = nn.Sequential(
            nn.Conv2d(pha_dim[0] + mag_dim[0], pha_dim[0], 1, 1, 0, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=pha_dim[0])
        )

        # --- 4. 最终解码器 ---
        self.mag_to_mask_proj = nn.Conv2d(mag_dim[0], mag_base, 1, 1, 0, bias=False)
        self.mask_decoder = MagDecoder(cfg)
        pha_dec_cfg = deepcopy(cfg)
        pha_dec_cfg['model_cfg']['hid_feature'] = pha_base
        self.phase_decoder = PhaseDecoder(pha_dec_cfg)

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
        pha_in = torch.cat((noisy_mag_4d, noisy_pha_4d), dim=1)

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
        mag_fm_blocks = self.mag_FM_middle
        mag_tm_blocks = self.mag_TM_middle
        mag_prev = mag_x3

        # ---------------------------
        # Phase Tower Encoder (RI input)
        # ---------------------------
        pha_x1 = self.pha_encoder(pha_in)
        pha_copy1 = pha_x1
        pha_x1 = self.pha_patch_embed_encoder_level1(pha_x1)
        for block in self.pha_TSMamba1_encoder:
            pha_x1 = block(pha_x1)
        pha_x1 = pha_copy1 + pha_x1
        pha_skip1 = pha_x1

        pha_x2 = self.pha_down1_2(pha_x1)
        pha_copy2 = pha_x2
        pha_x2 = self.pha_patch_embed_encoder_level2(pha_x2)
        for block in self.pha_TSMamba2_encoder:
            pha_x2 = block(pha_x2)
        pha_x2 = pha_copy2 + pha_x2
        pha_skip2 = pha_x2

        pha_x3 = self.pha_down2_3(pha_x2)
        pha_x3 = self.pha_patch_embed_middle(pha_x3)
        pha_tm_blocks = self.pha_TM_middle
        pha_fm_blocks = self.pha_FM_middle
        pha_prev = pha_x3

        # ---------------------------
        # Middle交替：FM/TM 处理后立即耦合融合（取代交叉注意力）
        stage_pairs = []
        for idx in range(self.num_mid_stages):
            pair_idx = idx // 2
            if idx % 2 == 0:
                stage_pairs.append((mag_fm_blocks[pair_idx], pha_tm_blocks[pair_idx]))
            else:
                stage_pairs.append((mag_tm_blocks[pair_idx], pha_fm_blocks[pair_idx]))
        for idx, (mag_block, pha_block) in enumerate(stage_pairs):
            mag_res, pha_res = mag_x3, pha_x3
            mag_feat = mag_block(mag_x3)
            pha_feat = pha_block(pha_x3)

            mag_in_fuse = self.mid_in_proj_mag[idx](mag_feat)
            pha_in_fuse = self.mid_in_proj_pha[idx](pha_feat)
            mag_fused, pha_fused = self.mid_fusions[idx](mag_in_fuse, pha_in_fuse)

            mag_cat = torch.cat([mag_feat, mag_fused], dim=1)
            pha_cat = torch.cat([pha_feat, pha_fused], dim=1)
            mag_x3 = self.mid_fusion_proj_mag[idx](mag_cat)
            pha_x3 = self.mid_fusion_proj_pha[idx](pha_cat)
            # 保持残差链路
            mag_x3 = mag_x3 + mag_res
            pha_x3 = pha_x3 + pha_res

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

        pha_y2 = self.pha_up3_2(pha_x3)
        pha_y2 = torch.cat([pha_y2, pha_skip2], 1)
        pha_y2 = self.pha_concat_level2(pha_y2)
        pha_y2_copy = pha_y2
        pha_y2 = self.pha_patch_embed_decoder_level2(pha_y2)
        for block in self.pha_TSMamba2_decoder:
            pha_y2 = block(pha_y2)

        mag_y2 = mag_y2_copy + mag_y2
        pha_y2 = pha_y2_copy + pha_y2

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

        pha_y1 = self.pha_up2_1(pha_y2)
        pha_y1 = torch.cat([pha_y1, pha_skip1], 1)
        pha_y1 = self.pha_concat_level1(pha_y1)
        pha_y1_copy = pha_y1
        pha_y1 = self.pha_patch_embed_decoder_level1(pha_y1)
        for block in self.pha_TSMamba1_decoder:
            pha_y1 = block(pha_y1)

        mag_y1 = mag_y1_copy + mag_y1
        pha_y1 = pha_y1_copy + pha_y1

        # Mag Refinement & Output
        mag_copy_ref = mag_y1
        mag_y1 = self.mag_patch_embed_refinement(mag_y1)
        for block in self.mag_refinement:
            mag_y1 = block(mag_y1)
        mag_final = self.mag_output(mag_y1 + mag_copy_ref) + mag_skip1

        # Pha Refinement & Output
        pha_copy_ref = pha_y1
        pha_y1 = self.pha_patch_embed_refinement(pha_y1)
        for block in self.pha_refinement:
            pha_y1 = block(pha_y1)
        pha_final = self.pha_output(pha_y1 + pha_copy_ref) + pha_skip1

        # Final耦合融合（相位先升维到 Mag 再交互，输出后压回相位宽度）
        mag_fused, pha_fused_high = self.global_fusion(mag_final, self.global_in_proj_pha(pha_final))
        pha_fused = self.global_out_proj_pha(torch.cat([pha_final, pha_fused_high], dim=1))

        if not torch.isfinite(pha_fused).all():
             raise RuntimeError('pha_fused contains NaN/Inf')

        # ---------------------------
        # Final Signal Reconstruction
        # ---------------------------
        mag_mask = self.mask_decoder(self.mag_to_mask_proj(mag_fused))
        if not torch.isfinite(mag_mask).all():
            raise RuntimeError('mag_mask contains NaN/Inf')
        denoised_mag = rearrange(mag_mask * noisy_mag_4d, 'b c t f -> b f t c').squeeze(-1)

        # Phase residual (tanh*pi) then add to noisy_pha
        if not torch.isfinite(denoised_mag).all():
            raise RuntimeError('denoised_mag contains NaN/Inf')
        rot_vec = self.phase_decoder(pha_fused)
        rot_vec = F.normalize(rot_vec, dim=1, p=2, eps=1e-8)
        delta_cos, delta_sin = torch.chunk(rot_vec, 2, dim=1)
        delta_cos = delta_cos.squeeze(1)  # [B, T, F]
        delta_sin = delta_sin.squeeze(1)

        # 直接基于 noisy_pha 计算 sin/cos
        noisy_cos = torch.cos(noisy_pha_4d).squeeze(1)  # [B, T, F]
        noisy_sin = torch.sin(noisy_pha_4d).squeeze(1)  # [B, T, F]

        pred_cos = noisy_cos * delta_cos - noisy_sin * delta_sin
        pred_sin = noisy_sin * delta_cos + noisy_cos * delta_sin

        # 对齐到 [B, F, T] 以匹配幅度分支
        pred_cos = rearrange(pred_cos, 'b t f -> b f t')
        pred_sin = rearrange(pred_sin, 'b t f -> b f t')

        pred_pha = torch.atan2(pred_sin, pred_cos)
        if not torch.isfinite(pred_pha).all():
            raise RuntimeError('pred_pha contains NaN/Inf')

        denoised_com = torch.stack(
            (denoised_mag * pred_cos,
             denoised_mag * pred_sin),
            dim=-1
        )
        if not torch.isfinite(denoised_com).all():
            raise RuntimeError('denoised_com contains NaN/Inf')

        return denoised_mag, pred_pha, denoised_com
