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
        rng_state = torch.random.get_rng_state()
        self.shortcut = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(input_feat * 4, out_feat, 1, 1, 0, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_feat),
        )
        torch.random.set_rng_state(rng_state)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return self.body(x) + torch.tanh(self.residual_scale) * self.shortcut(x)


class Upsample(nn.Module):
    def __init__(self, input_feat, out_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            # dw
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1, padding=1, groups=input_feat, bias=False),
            # pw-linear
            nn.Conv2d(input_feat, out_feat * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2))
        rng_state = torch.random.get_rng_state()
        self.shortcut = nn.Sequential(
            nn.Conv2d(input_feat, out_feat * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2),
            nn.GroupNorm(num_groups=1, num_channels=out_feat),
        )
        torch.random.set_rng_state(rng_state)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return self.body(x) + torch.tanh(self.residual_scale) * self.shortcut(x)


class ResidualDenseBridge(nn.Module):
    """Fuse same-resolution states through a zero-start residual adapter."""

    def __init__(self, target_channels, context_channels, width_ratio=0.5):
        super().__init__()
        self.context_channels = tuple(int(channels) for channels in context_channels)
        total_channels = target_channels + sum(self.context_channels)
        hidden_channels = max(4, int(round(target_channels * width_ratio)))
        self.fusion = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=total_channels),
            nn.Conv2d(total_channels, hidden_channels, 1, 1, 0, bias=False),
            nn.PReLU(hidden_channels),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                3,
                1,
                1,
                groups=hidden_channels,
                bias=False
            ),
            nn.GroupNorm(num_groups=1, num_channels=hidden_channels),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, target_channels, 1, 1, 0, bias=False),
        )
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, target, *contexts):
        if len(contexts) != len(self.context_channels):
            raise ValueError(
                f'Expected {len(self.context_channels)} contexts, got {len(contexts)}.'
            )
        for index, (context, channels) in enumerate(
            zip(contexts, self.context_channels)
        ):
            if context.shape[1] != channels:
                raise ValueError(
                    f'Context {index} has {context.shape[1]} channels, expected {channels}.'
                )
            if context.shape[0] != target.shape[0] or context.shape[2:] != target.shape[2:]:
                raise ValueError(
                    f'Context {index} shape {tuple(context.shape)} is incompatible with '
                    f'target {tuple(target.shape)}.'
                )
        update = self.fusion(torch.cat((target, *contexts), dim=1))
        return target + torch.tanh(self.residual_scale) * update


class MultiScaleLocalChannelRefiner(nn.Module):
    """Refine local TF detail with stable intra-module residual reuse."""

    def __init__(
        self,
        channels,
        strip_kernel=7,
        initial_scale=0.05,
        dense_initial_scale=0.0,
    ):
        super().__init__()
        if strip_kernel < 3 or strip_kernel % 2 == 0:
            raise ValueError('strip_kernel must be an odd integer >= 3.')
        if not 0.0 < initial_scale < 1.0:
            raise ValueError('initial_scale must be in (0, 1).')
        if not -1.0 < dense_initial_scale < 1.0:
            raise ValueError('dense_initial_scale must be in (-1, 1).')

        strip_padding = strip_kernel // 2
        self.pre_norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.input_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.local_3x3 = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.temporal_strip = nn.Conv2d(
            channels,
            channels,
            (strip_kernel, 1),
            padding=(strip_padding, 0),
            groups=channels,
            bias=False,
        )
        self.frequency_strip = nn.Conv2d(
            channels,
            channels,
            (1, strip_kernel),
            padding=(0, strip_padding),
            groups=channels,
            bias=False,
        )
        self.output_projection = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

        # ECA-style local cross-channel interaction without channel reduction.
        self.channel_attention = nn.Conv1d(
            1, 1, kernel_size=3, padding=1, bias=False
        )
        # Keep the original Conv1d construction (and RNG consumption), then
        # center the channel gain at exactly one for a neutral initialization.
        nn.init.zeros_(self.channel_attention.weight)
        self.residual_scale = nn.Parameter(
            torch.tensor(math.atanh(float(initial_scale)), dtype=torch.float32)
        )
        dense_parameter = math.atanh(float(dense_initial_scale))
        self.dense_residual_scales = nn.Parameter(
            torch.full((3,), dense_parameter, dtype=torch.float32)
        )
        self.branch_logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self.latest_diagnostics = {}

    def _compute_branches(self, projected):
        dense_scales = torch.tanh(self.dense_residual_scales)
        local_3x3 = self.local_3x3(projected)
        temporal = self.temporal_strip(
            projected + dense_scales[0] * local_3x3
        )
        frequency = self.frequency_strip(
            projected
            + dense_scales[1] * local_3x3
            + dense_scales[2] * temporal
        )
        return local_3x3, temporal, frequency

    def _branch_weights(self):
        return math.sqrt(3.0) * torch.softmax(self.branch_logits, dim=0)

    def _channel_gain(self, update):
        channel_descriptor = F.adaptive_avg_pool2d(update, 1)
        channel_descriptor = channel_descriptor.squeeze(-1).transpose(-1, -2)
        gain = 2.0 * torch.sigmoid(self.channel_attention(channel_descriptor))
        return gain.transpose(-1, -2).unsqueeze(-1)

    def forward(self, x):
        projected = self.input_projection(self.pre_norm(x))
        branches = self._compute_branches(projected)
        branch_weights = self._branch_weights()
        local = sum(
            weight * branch for weight, branch in zip(branch_weights, branches)
        )
        update = self.output_projection(local)
        channel_gain = self._channel_gain(update)
        applied_update = torch.tanh(self.residual_scale) * update * channel_gain

        with torch.no_grad():
            input_rms = x.detach().float().square().mean().sqrt().clamp_min(1e-8)
            update_rms = applied_update.detach().float().square().mean().sqrt()
            self.latest_diagnostics = {
                'dense_scales': torch.tanh(self.dense_residual_scales).detach(),
                'branch_weights': branch_weights.detach(),
                'channel_gain_mean': channel_gain.detach().float().mean(),
                'update_ratio': update_rms / input_rms,
            }

        return x + applied_update


def _make_full_resolution_head(in_channels, out_channels, output_bias=0.0):
    """Decode encoder-resolution features back to the input STFT grid."""
    head = nn.Sequential(
        nn.Conv2d(in_channels, in_channels * 4, 1, 1, 0, bias=False),
        nn.PixelShuffle(2),
        nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(1, 3),
            stride=(2, 1),
            padding=(0, 1),
            groups=in_channels,
            bias=False
        ),
        nn.InstanceNorm2d(in_channels, affine=True),
        nn.PReLU(in_channels),
        nn.Conv2d(in_channels, out_channels, (1, 1))
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.constant_(head[-1].bias, output_bias)
    return head


class MambaSEUNet(nn.Module):
    """
    Harmonic-prior suppression-generation Mamba speech enhancement model.

    Stage 1 combines a center complex mask with neighboring-frame complex
    filtering for strong coarse suppression. A differentiable F0 template bank
    extracts a soft harmonic prior. Stage 2 generates separate harmonic and
    aperiodic complex residuals under that prior, while suppression features
    provide one-way bottleneck context.
    """

    def __init__(self, cfg):
        super(MambaSEUNet, self).__init__()
        self.cfg = cfg
        self.num_tscblocks = cfg['model_cfg'].get('num_tfmamba', 4)
        self.num_mid_pairs = int(cfg['model_cfg'].get('num_mid_pairs', 2))
        self.num_mid_pairs = max(1, min(4, self.num_mid_pairs))

        # Both towers are full-width by default in the high-capacity generator.
        mag_base = cfg['model_cfg']['hid_feature']
        restore_width_ratio = float(cfg['model_cfg'].get('restoration_width_ratio', 1.0))
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
        self.dense_bridge_width_ratio = float(
            cfg['model_cfg'].get('dense_bridge_width_ratio', 0.5)
        )
        if not 0.0 < self.dense_bridge_width_ratio <= 2.0:
            raise ValueError('dense_bridge_width_ratio must be in (0, 2].')

        # --- 1. 初始化输入配置 ---
        mag_cfg = deepcopy(cfg)
        mag_cfg['model_cfg']['input_channel'] = 2
        mag_cfg['model_cfg']['hid_feature'] = mag_base

        restore_cfg = deepcopy(cfg)
        restore_cfg['model_cfg']['input_channel'] = 6
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

        bridge_ratio = self.dense_bridge_width_ratio
        rng_state = torch.random.get_rng_state()
        self.dense_bridges = nn.ModuleDict({
            'encoder_level1': ResidualDenseBridge(
                restore_dim[0], [mag_dim[0], mag_dim[0], mag_dim[0]], bridge_ratio
            ),
            'encoder_level2': ResidualDenseBridge(
                restore_dim[1], [mag_dim[1], mag_dim[1]], bridge_ratio
            ),
            'middle': ResidualDenseBridge(
                restore_dim[2],
                [mag_dim[2]] * (self.num_mid_pairs + 1),
                bridge_ratio
            ),
            'decoder_level2': ResidualDenseBridge(
                restore_dim[1], [mag_dim[1], mag_dim[1]], bridge_ratio
            ),
            'decoder_level1': ResidualDenseBridge(
                restore_dim[0], [mag_dim[0], mag_dim[0], mag_dim[0]], bridge_ratio
            ),
            'output': ResidualDenseBridge(
                restore_dim[0], [mag_dim[0]], bridge_ratio
            ),
        })
        torch.random.set_rng_state(rng_state)

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
        output_channels = int(cfg['model_cfg']['output_channel'])
        if output_channels != 1:
            raise ValueError('The harmonic generator currently supports one output channel.')

        # Neighboring-frame complex filters augment the center complex mask.
        self.deep_filter_offsets = tuple(
            int(offset) for offset in cfg['model_cfg'].get(
                'deep_filter_offsets', [-2, -1, 1, 2]
            )
        )
        if not self.deep_filter_offsets or 0 in self.deep_filter_offsets:
            raise ValueError('deep_filter_offsets must contain non-zero frame offsets.')
        if len(set(self.deep_filter_offsets)) != len(self.deep_filter_offsets):
            raise ValueError('deep_filter_offsets must be unique.')
        num_side_filters = len(self.deep_filter_offsets)
        self.deep_filter_gate_bias = float(
            cfg['model_cfg'].get('deep_filter_gate_bias', -2.0)
        )
        self.deep_filter_coeff_decoder = _make_full_resolution_head(
            mag_dim[0], num_side_filters * 2
        )
        self.deep_filter_gate_decoder = _make_full_resolution_head(
            mag_dim[0], num_side_filters, self.deep_filter_gate_bias
        )

        # Fixed differentiable harmonic analysis. Buffers are regenerated from
        # config, so checkpoints stay independent of the template resolution.
        self.pitch_candidates = int(cfg['model_cfg'].get('pitch_candidates', 64))
        self.pitch_min_f0 = float(cfg['model_cfg'].get('pitch_min_f0', 60.0))
        self.pitch_max_f0 = float(cfg['model_cfg'].get('pitch_max_f0', 500.0))
        self.pitch_temperature = float(cfg['model_cfg'].get('pitch_temperature', 0.1))
        self.pitch_smoothing_bins = int(cfg['model_cfg'].get('pitch_smoothing_bins', 9))
        self.harmonic_bandwidth_bins = float(
            cfg['model_cfg'].get('harmonic_bandwidth_bins', 0.75)
        )
        if self.pitch_candidates < 2:
            raise ValueError('pitch_candidates must be at least 2.')
        if not 0.0 < self.pitch_min_f0 < self.pitch_max_f0:
            raise ValueError('Expected 0 < pitch_min_f0 < pitch_max_f0.')
        if self.pitch_temperature <= 0.0:
            raise ValueError('pitch_temperature must be positive.')
        if self.pitch_smoothing_bins < 3 or self.pitch_smoothing_bins % 2 == 0:
            raise ValueError('pitch_smoothing_bins must be an odd integer >= 3.')
        if self.harmonic_bandwidth_bins <= 0.0:
            raise ValueError('harmonic_bandwidth_bins must be positive.')

        candidate_f0s, pitch_templates, harmonic_templates = self._build_harmonic_templates()
        self.register_buffer('candidate_f0s', candidate_f0s, persistent=False)
        self.register_buffer('pitch_templates', pitch_templates, persistent=False)
        self.register_buffer('harmonic_templates', harmonic_templates, persistent=False)

        self.voicing_gate_bias = float(cfg['model_cfg'].get('voicing_gate_bias', -1.0))
        self.voicing_confidence_power = float(
            cfg['model_cfg'].get('voicing_confidence_power', 0.5)
        )
        if not 0.0 < self.voicing_confidence_power <= 1.0:
            raise ValueError('voicing_confidence_power must be in (0, 1].')
        self.voicing_head = nn.Conv2d(mag_dim[0], 1, 1)
        nn.init.zeros_(self.voicing_head.weight)
        nn.init.constant_(self.voicing_head.bias, self.voicing_gate_bias)

        # Separate residual generators make the source prior explicit. Their
        # zero initialization preserves the Stage-1 output at startup.
        self.harmonic_residual_decoder = _make_full_resolution_head(
            restore_dim[0], output_channels * 2
        )
        self.aperiodic_residual_decoder = _make_full_resolution_head(
            restore_dim[0], output_channels * 2
        )
        self.restoration_gate = _make_full_resolution_head(
            restore_dim[0], 2, self.complex_residual_gate_bias
        )
        self.latest_aux = {}

        strip_kernel = int(cfg['model_cfg'].get('local_channel_strip_kernel', 7))
        initial_scale = float(
            cfg['model_cfg'].get('local_channel_initial_scale', 0.05)
        )
        dense_initial_scale = float(
            cfg['model_cfg'].get('local_channel_dense_initial_scale', 0.0)
        )
        self.local_channel_refiners = nn.ModuleDict({
            'mag_encoder_level1': MultiScaleLocalChannelRefiner(
                mag_dim[0], strip_kernel, initial_scale, dense_initial_scale
            ),
            'mag_encoder_level2': MultiScaleLocalChannelRefiner(
                mag_dim[1], strip_kernel, initial_scale, dense_initial_scale
            ),
            'mag_middle': MultiScaleLocalChannelRefiner(
                mag_dim[2], strip_kernel, initial_scale, dense_initial_scale
            ),
            'mag_decoder_level2': MultiScaleLocalChannelRefiner(
                mag_dim[1], strip_kernel, initial_scale, dense_initial_scale
            ),
            'mag_decoder_level1': MultiScaleLocalChannelRefiner(
                mag_dim[0], strip_kernel, initial_scale, dense_initial_scale
            ),
            'mag_refinement': MultiScaleLocalChannelRefiner(
                mag_dim[0], strip_kernel, initial_scale, dense_initial_scale
            ),
            'restore_encoder_level1': MultiScaleLocalChannelRefiner(
                restore_dim[0], strip_kernel, initial_scale, dense_initial_scale
            ),
            'restore_encoder_level2': MultiScaleLocalChannelRefiner(
                restore_dim[1], strip_kernel, initial_scale, dense_initial_scale
            ),
            'restore_middle': MultiScaleLocalChannelRefiner(
                restore_dim[2], strip_kernel, initial_scale, dense_initial_scale
            ),
            'restore_decoder_level2': MultiScaleLocalChannelRefiner(
                restore_dim[1], strip_kernel, initial_scale, dense_initial_scale
            ),
            'restore_decoder_level1': MultiScaleLocalChannelRefiner(
                restore_dim[0], strip_kernel, initial_scale, dense_initial_scale
            ),
            'restore_refinement': MultiScaleLocalChannelRefiner(
                restore_dim[0], strip_kernel, initial_scale, dense_initial_scale
            ),
        })

    def _build_harmonic_templates(self):
        n_fft = int(self.cfg['stft_cfg']['n_fft'])
        sample_rate = float(self.cfg['stft_cfg']['sampling_rate'])
        num_bins = n_fft // 2 + 1
        nyquist = sample_rate / 2.0
        bin_hz = sample_rate / n_fft
        bandwidth_hz = self.harmonic_bandwidth_bins * bin_hz

        candidate_f0s = torch.logspace(
            math.log10(self.pitch_min_f0),
            math.log10(self.pitch_max_f0),
            self.pitch_candidates,
            dtype=torch.float32
        )
        frequencies = torch.arange(num_bins, dtype=torch.float32) * bin_hz
        harmonic_templates = []
        pitch_templates = []
        for f0 in candidate_f0s.tolist():
            num_harmonics = max(1, int(nyquist // f0))
            harmonic_indices = torch.arange(
                1, num_harmonics + 1, dtype=torch.float32
            )
            harmonic_frequencies = harmonic_indices * f0
            distances = (
                frequencies.unsqueeze(0) - harmonic_frequencies.unsqueeze(1)
            ) / bandwidth_hz
            harmonic_weights = harmonic_indices.rsqrt().unsqueeze(1)
            template = (
                torch.exp(-0.5 * distances.square()) * harmonic_weights
            ).sum(dim=0)
            template[0] = 0.0
            template = template / template.amax().clamp_min(1e-8)
            harmonic_templates.append(template)
            pitch_templates.append(template / torch.linalg.vector_norm(template).clamp_min(1e-8))

        return (
            candidate_f0s,
            torch.stack(pitch_templates, dim=0),
            torch.stack(harmonic_templates, dim=0),
        )

    @staticmethod
    def _shift_time(spectrum, offset):
        if offset < 0:
            amount = -offset
            return F.pad(spectrum[:, :, :-amount, :], (0, 0, amount, 0))
        return F.pad(spectrum[:, :, offset:, :], (0, 0, 0, offset))

    def harmonic_analysis(self, magnitude):
        """Return soft F0 posterior, harmonic occupancy, and pitch confidence."""
        if magnitude.ndim == 3:
            magnitude = rearrange(magnitude, 'b f t -> b 1 t f')
        elif magnitude.ndim != 4 or magnitude.shape[1] != 1:
            raise ValueError('Expected magnitude with shape [B, F, T] or [B, 1, T, F].')
        if magnitude.shape[-1] != self.pitch_templates.shape[-1]:
            raise ValueError(
                'Magnitude/template frequency dimensions differ: '
                f'{magnitude.shape[-1]} vs {self.pitch_templates.shape[-1]}.'
            )

        log_magnitude = torch.log1p(magnitude.clamp_min(0.0)).squeeze(1)
        batch_size, frames, freq_bins = log_magnitude.shape
        flattened = log_magnitude.reshape(batch_size * frames, 1, freq_bins)
        smooth_envelope = F.avg_pool1d(
            flattened,
            kernel_size=self.pitch_smoothing_bins,
            stride=1,
            padding=self.pitch_smoothing_bins // 2
        ).reshape(batch_size, frames, freq_bins)
        salience = F.relu(log_magnitude - smooth_envelope)
        salience = F.normalize(salience, dim=-1, p=2, eps=1e-8)

        pitch_logits = torch.einsum(
            'btf,pf->btp', salience, self.pitch_templates
        ) / self.pitch_temperature
        pitch_posterior = torch.softmax(pitch_logits, dim=-1)
        harmonic_prior = torch.einsum(
            'btp,pf->btf', pitch_posterior, self.harmonic_templates
        ).unsqueeze(1)
        entropy = -(
            pitch_posterior * torch.log(pitch_posterior.clamp_min(1e-8))
        ).sum(dim=-1)
        confidence = (
            1.0 - entropy / math.log(self.pitch_candidates)
        ).clamp(0.0, 1.0)
        return pitch_posterior, harmonic_prior, confidence

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
        mag_x1 = self.local_channel_refiners['mag_encoder_level1'](mag_x1)
        mag_skip1 = mag_x1

        mag_x2 = self.mag_down1_2(mag_x1)
        mag_copy2 = mag_x2
        mag_x2 = self.mag_patch_embed_encoder_level2(mag_x2)
        for block in self.mag_TSMamba2_encoder:
            mag_x2 = block(mag_x2)
        mag_x2 = mag_copy2 + mag_x2
        mag_x2 = self.local_channel_refiners['mag_encoder_level2'](mag_x2)
        mag_skip2 = mag_x2

        mag_x3 = self.mag_down2_3(mag_x2)
        mag_x3 = self.mag_patch_embed_middle(mag_x3)
        suppress_middle_features = [mag_x3]
        for fm_block, tm_block in zip(self.mag_FM_middle, self.mag_TM_middle):
            mag_x3 = fm_block(mag_x3)
            mag_x3 = tm_block(mag_x3)
            suppress_middle_features.append(mag_x3)
        mag_x3 = self.local_channel_refiners['mag_middle'](mag_x3)
        suppress_middle_features[-1] = mag_x3
        suppress_bottleneck = mag_x3

        # Stage 1 decoder and coarse complex-spectrum reconstruction.
        mag_y2 = self.mag_up3_2(mag_x3)
        mag_y2 = self.mag_concat_level2(torch.cat([mag_y2, mag_skip2], dim=1))
        mag_y2_copy = mag_y2
        mag_y2 = self.mag_patch_embed_decoder_level2(mag_y2)
        for block in self.mag_TSMamba2_decoder:
            mag_y2 = block(mag_y2)
        mag_y2 = mag_y2_copy + mag_y2
        mag_y2 = self.local_channel_refiners['mag_decoder_level2'](mag_y2)

        # Decode to the original encoder resolution.
        mag_y1 = self.mag_up2_1(mag_y2)
        mag_y1 = self.mag_concat_level1(torch.cat([mag_y1, mag_skip1], dim=1))
        mag_y1_copy = mag_y1
        mag_y1 = self.mag_patch_embed_decoder_level1(mag_y1)
        for block in self.mag_TSMamba1_decoder:
            mag_y1 = block(mag_y1)
        mag_y1 = mag_y1_copy + mag_y1
        mag_y1 = self.local_channel_refiners['mag_decoder_level1'](mag_y1)

        # Refine Stage 1 features before coarse reconstruction.
        mag_copy_ref = mag_y1
        mag_y1 = self.mag_patch_embed_refinement(mag_y1)
        for block in self.mag_refinement:
            mag_y1 = block(mag_y1)
        mag_refined = self.local_channel_refiners['mag_refinement'](
            mag_y1 + mag_copy_ref
        )
        mag_final = self.mag_output(mag_refined) + mag_skip1

        # Coarse signal reconstruction.
        mag_mask = self.mask_decoder(self.mag_to_mask_proj(mag_final))
        if not torch.isfinite(mag_mask).all():
            raise RuntimeError('mag_mask contains NaN/Inf')
        center_mag_4d = mag_mask * noisy_mag_4d

        # Predict a unit complex rotation and apply it on the noisy phase unit vector.
        rot_vec = self.coarse_phase_decoder(mag_final)
        rot_vec = F.normalize(rot_vec, dim=1, p=2, eps=self.phase_eps)
        delta_cos, delta_sin = torch.chunk(rot_vec, 2, dim=1)

        noisy_cos = torch.cos(noisy_pha_4d)
        noisy_sin = torch.sin(noisy_pha_4d)
        noisy_real_4d = noisy_mag_4d * noisy_cos
        noisy_imag_4d = noisy_mag_4d * noisy_sin

        coarse_cos = noisy_cos * delta_cos - noisy_sin * delta_sin
        coarse_sin = noisy_sin * delta_cos + noisy_cos * delta_sin

        center_real_4d = center_mag_4d * coarse_cos
        center_imag_4d = center_mag_4d * coarse_sin

        # Complex deep filtering uses neighboring noisy frames as learnable
        # cancellation/reconstruction taps around the center estimate.
        deep_filter_coefficients = torch.tanh(
            self.deep_filter_coeff_decoder(mag_final)
        )
        deep_filter_gates = torch.sigmoid(
            self.deep_filter_gate_decoder(mag_final)
        )
        coeff_real, coeff_imag = torch.chunk(
            deep_filter_coefficients, 2, dim=1
        )
        side_real_4d = torch.zeros_like(center_real_4d)
        side_imag_4d = torch.zeros_like(center_imag_4d)
        for index, offset in enumerate(self.deep_filter_offsets):
            shifted_real = self._shift_time(noisy_real_4d, offset)
            shifted_imag = self._shift_time(noisy_imag_4d, offset)
            tap_real = coeff_real[:, index:index + 1]
            tap_imag = coeff_imag[:, index:index + 1]
            tap_gate = deep_filter_gates[:, index:index + 1]
            side_real_4d = side_real_4d + tap_gate * (
                tap_real * shifted_real - tap_imag * shifted_imag
            )
            side_imag_4d = side_imag_4d + tap_gate * (
                tap_real * shifted_imag + tap_imag * shifted_real
            )

        coarse_real_4d = center_real_4d + side_real_4d
        coarse_imag_4d = center_imag_4d + side_imag_4d
        coarse_mag_4d = torch.sqrt(torch.clamp(
            coarse_real_4d.square() + coarse_imag_4d.square(), min=1e-12
        ))
        if not torch.isfinite(coarse_real_4d).all() or not torch.isfinite(coarse_imag_4d).all():
            raise RuntimeError('Coarse complex spectrum contains NaN/Inf')

        # The soft source prior remains differentiable with respect to Stage 1.
        pitch_posterior, raw_harmonic_prior, pitch_confidence = self.harmonic_analysis(
            coarse_mag_4d
        )
        voicing = torch.sigmoid(
            self.voicing_head(mag_final).mean(dim=-1, keepdim=True)
        )
        # Train voicing only against its source-analysis teacher. Isolating the
        # restoration gradient prevents a trivial all-aperiodic bypass.
        conditioning_voicing = voicing.detach()
        harmonic_prior = raw_harmonic_prior * conditioning_voicing
        voicing_map = conditioning_voicing.expand(
            -1, -1, -1, noisy_mag_4d.shape[-1]
        )

        # Stage 2 restores details from the original/coarse spectra and source prior.
        restore_in = torch.cat(
            [
                noisy_real_4d,
                noisy_imag_4d,
                coarse_real_4d,
                coarse_imag_4d,
                harmonic_prior,
                voicing_map,
            ],
            dim=1
        )

        restore_x1 = self.restore_encoder(restore_in)
        restore_copy1 = restore_x1
        restore_x1 = self.restore_patch_embed_encoder_level1(restore_x1)
        for block in self.restore_TMamba1_encoder:
            restore_x1 = block(restore_x1)
        restore_x1 = restore_copy1 + restore_x1
        restore_x1 = self.local_channel_refiners[
            'restore_encoder_level1'
        ](restore_x1)
        restore_x1 = self.dense_bridges['encoder_level1'](
            restore_x1, mag_skip1, mag_y1, mag_final
        )
        restore_skip1 = restore_x1

        restore_x2 = self.restore_down1_2(restore_x1)
        restore_copy2 = restore_x2
        restore_x2 = self.restore_patch_embed_encoder_level2(restore_x2)
        for block in self.restore_TMamba2_encoder:
            restore_x2 = block(restore_x2)
        restore_x2 = restore_copy2 + restore_x2
        restore_x2 = self.local_channel_refiners[
            'restore_encoder_level2'
        ](restore_x2)
        restore_x2 = self.dense_bridges['encoder_level2'](
            restore_x2, mag_skip2, mag_y2
        )
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
        restore_x3 = self.dense_bridges['middle'](
            restore_x3, *suppress_middle_features
        )
        for tm_block, fm_block in zip(self.restore_TM_middle, self.restore_FM_middle):
            restore_x3 = tm_block(restore_x3)
            restore_x3 = fm_block(restore_x3)
        restore_x3 = self.local_channel_refiners['restore_middle'](restore_x3)

        restore_y2 = self.restore_up3_2(restore_x3)
        restore_y2 = self.restore_concat_level2(
            torch.cat([restore_y2, restore_skip2], dim=1)
        )
        restore_y2_copy = restore_y2
        restore_y2 = self.restore_patch_embed_decoder_level2(restore_y2)
        for block in self.restore_TMamba2_decoder:
            restore_y2 = block(restore_y2)
        restore_y2 = restore_y2_copy + restore_y2
        restore_y2 = self.local_channel_refiners[
            'restore_decoder_level2'
        ](restore_y2)
        restore_y2 = self.dense_bridges['decoder_level2'](
            restore_y2, mag_skip2, mag_y2
        )

        restore_y1 = self.restore_up2_1(restore_y2)
        restore_y1 = self.restore_concat_level1(
            torch.cat([restore_y1, restore_skip1], dim=1)
        )
        restore_y1_copy = restore_y1
        restore_y1 = self.restore_patch_embed_decoder_level1(restore_y1)
        for block in self.restore_TMamba1_decoder:
            restore_y1 = block(restore_y1)
        restore_y1 = restore_y1_copy + restore_y1
        restore_y1 = self.local_channel_refiners[
            'restore_decoder_level1'
        ](restore_y1)
        restore_y1 = self.dense_bridges['decoder_level1'](
            restore_y1, mag_skip1, mag_y1, mag_final
        )

        restore_copy_ref = restore_y1
        restore_y1 = self.restore_patch_embed_refinement(restore_y1)
        for block in self.restore_refinement:
            restore_y1 = block(restore_y1)
        restore_y1 = self.local_channel_refiners['restore_refinement'](
            restore_y1 + restore_copy_ref
        )
        restore_final = self.restore_output(restore_y1) + restore_skip1
        restore_final = self.dense_bridges['output'](restore_final, mag_final)

        harmonic_residual = torch.tanh(
            self.harmonic_residual_decoder(restore_final)
        )
        aperiodic_residual = torch.tanh(
            self.aperiodic_residual_decoder(restore_final)
        )
        if not torch.isfinite(harmonic_residual).all():
            raise RuntimeError('harmonic_residual contains NaN/Inf')
        if not torch.isfinite(aperiodic_residual).all():
            raise RuntimeError('aperiodic_residual contains NaN/Inf')
        restoration_gates = torch.sigmoid(self.restoration_gate(restore_final))
        if not torch.isfinite(restoration_gates).all():
            raise RuntimeError('restoration_gates contains NaN/Inf')

        harmonic_real, harmonic_imag = torch.chunk(harmonic_residual, 2, dim=1)
        aperiodic_real, aperiodic_imag = torch.chunk(aperiodic_residual, 2, dim=1)
        harmonic_gate, aperiodic_gate = torch.chunk(restoration_gates, 2, dim=1)
        reference_mag_4d = 0.5 * (noisy_mag_4d + coarse_mag_4d)
        harmonic_support = harmonic_prior.clamp(0.0, 1.0)
        aperiodic_support = 1.0 - harmonic_support

        harmonic_real_4d = (
            harmonic_gate * harmonic_real * reference_mag_4d * harmonic_support
        )
        harmonic_imag_4d = (
            harmonic_gate * harmonic_imag * reference_mag_4d * harmonic_support
        )
        aperiodic_real_4d = (
            aperiodic_gate * aperiodic_real * reference_mag_4d * aperiodic_support
        )
        aperiodic_imag_4d = (
            aperiodic_gate * aperiodic_imag * reference_mag_4d * aperiodic_support
        )
        applied_real_4d = harmonic_real_4d + aperiodic_real_4d
        applied_imag_4d = harmonic_imag_4d + aperiodic_imag_4d

        coarse_real = rearrange(coarse_real_4d.squeeze(1), 'b t f -> b f t')
        coarse_imag = rearrange(coarse_imag_4d.squeeze(1), 'b t f -> b f t')
        applied_real = rearrange(applied_real_4d.squeeze(1), 'b t f -> b f t')
        applied_imag = rearrange(applied_imag_4d.squeeze(1), 'b t f -> b f t')
        harmonic_applied_real = rearrange(
            harmonic_real_4d.squeeze(1), 'b t f -> b f t'
        )
        harmonic_applied_imag = rearrange(
            harmonic_imag_4d.squeeze(1), 'b t f -> b f t'
        )
        aperiodic_applied_real = rearrange(
            aperiodic_real_4d.squeeze(1), 'b t f -> b f t'
        )
        aperiodic_applied_imag = rearrange(
            aperiodic_imag_4d.squeeze(1), 'b t f -> b f t'
        )
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

        dense_bridge_scales = torch.stack([
            torch.tanh(bridge.residual_scale)
            for bridge in self.dense_bridges.values()
        ])
        transition_residual_scales = torch.stack([
            torch.tanh(module.residual_scale)
            for module in (
                self.mag_down1_2,
                self.mag_down2_3,
                self.mag_up3_2,
                self.mag_up2_1,
                self.restore_down1_2,
                self.restore_down2_3,
                self.restore_up3_2,
                self.restore_up2_1,
            )
        ])
        local_channel_scales = torch.stack([
            torch.tanh(refiner.residual_scale)
            for refiner in self.local_channel_refiners.values()
        ]).detach()
        local_channel_dense_scales = torch.stack([
            refiner.latest_diagnostics['dense_scales']
            for refiner in self.local_channel_refiners.values()
        ])
        local_channel_branch_weights = torch.stack([
            refiner.latest_diagnostics['branch_weights']
            for refiner in self.local_channel_refiners.values()
        ])
        local_channel_channel_gain = torch.stack([
            refiner.latest_diagnostics['channel_gain_mean']
            for refiner in self.local_channel_refiners.values()
        ])
        local_channel_update_ratio = torch.stack([
            refiner.latest_diagnostics['update_ratio']
            for refiner in self.local_channel_refiners.values()
        ])
        suppression_slice = slice(0, 6)
        restoration_slice = slice(6, 12)
        self.latest_aux = {
            'coarse_complex': torch.stack((coarse_real, coarse_imag), dim=-1),
            'deep_filter_coefficients': deep_filter_coefficients,
            'deep_filter_gates': deep_filter_gates,
            'pitch_posterior': pitch_posterior,
            'pitch_confidence': pitch_confidence,
            'raw_harmonic_prior': raw_harmonic_prior,
            'harmonic_prior': harmonic_prior,
            'voicing': voicing,
            'harmonic_residual': harmonic_residual,
            'aperiodic_residual': aperiodic_residual,
            'restoration_gate': restoration_gates,
            'harmonic_residual_applied': torch.stack(
                (harmonic_applied_real, harmonic_applied_imag), dim=-1
            ),
            'aperiodic_residual_applied': torch.stack(
                (aperiodic_applied_real, aperiodic_applied_imag), dim=-1
            ),
            'complex_residual_applied': torch.stack((applied_real, applied_imag), dim=-1),
            'suppression_context_scale': context_scale,
            'dense_bridge_scales': dense_bridge_scales,
            'transition_residual_scales': transition_residual_scales,
            'local_channel_scales': local_channel_scales,
            'local_channel_dense_scales': local_channel_dense_scales,
            'local_channel_branch_weights': local_channel_branch_weights,
            'local_channel_channel_gain': local_channel_channel_gain,
            'local_channel_update_ratio': local_channel_update_ratio,
            'local_channel_suppression_scale_mean': (
                local_channel_scales[suppression_slice].abs().mean()
            ),
            'local_channel_restoration_scale_mean': (
                local_channel_scales[restoration_slice].abs().mean()
            ),
            'local_channel_suppression_update_ratio_mean': (
                local_channel_update_ratio[suppression_slice].mean()
            ),
            'local_channel_restoration_update_ratio_mean': (
                local_channel_update_ratio[restoration_slice].mean()
            ),
        }
        return denoised_mag, pred_pha, denoised_com
