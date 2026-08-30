"""Evidence-conditioned asymmetric polar refinement for Residual-Dense MPDM-Net."""

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .cross import SS2D_cross_new, SelectiveScan, eca_layer
from .mamba_block import FMambaBlock, TMambaBlock


def _scale_parameter(initial_value):
    if not -1.0 < initial_value < 1.0:
        raise ValueError('Residual scale initialization must be in (-1, 1).')
    return nn.Parameter(torch.tensor(math.atanh(float(initial_value))))


def _build_aligned_scan_sequences(features):
    """Create row/column forward and reverse sequences without losing layout."""
    batch, channels, height, width = features.shape
    length = height * width
    row_and_column = torch.stack((
        features.reshape(batch, channels, length),
        features.transpose(2, 3).contiguous().reshape(batch, channels, length),
    ), dim=1)
    return torch.cat((row_and_column, row_and_column.flip(-1)), dim=1)


def _merge_aligned_scan_outputs(outputs, height, width):
    """Invert every scan permutation before combining the four directions."""
    batch, directions, channels, length = outputs.shape
    if directions != 4 or length != height * width:
        raise ValueError(
            f'Expected [B, 4, C, {height * width}], got {tuple(outputs.shape)}.'
        )
    reverse = outputs[:, 2:4].flip(-1)
    column = outputs[:, 1].reshape(
        batch, channels, width, height
    ).transpose(2, 3).contiguous()
    reverse_column = reverse[:, 1].reshape(
        batch, channels, width, height
    ).transpose(2, 3).contiguous()
    return (
        outputs[:, 0].reshape(batch, channels, height, width)
        + reverse[:, 0].reshape(batch, channels, height, width)
        + column
        + reverse_column
    )


def _aligned_selective_scan_branch(module, features):
    """Run one branch and restore every directional result to the TF grid."""
    batch, _, height, width = features.shape
    length = height * width
    sequences = _build_aligned_scan_sequences(features)
    projected = torch.einsum(
        'b k d l, k c d -> b k c l', sequences, module.x_proj_weight
    )
    delta, state_b, state_c = torch.split(
        projected, [module.dt_rank, module.d_state, module.d_state], dim=2
    )
    delta = torch.einsum(
        'b k r l, k d r -> b k d l', delta, module.dt_projs_weight
    )
    sequences = sequences.reshape(batch, -1, length).float()
    delta = delta.contiguous().reshape(batch, -1, length).float()
    state_b = state_b.contiguous().float()
    state_c = state_c.contiguous().float()
    state_a = -torch.exp(module.A_logs.float())
    skip = module.Ds.float()
    delta_bias = module.dt_projs_bias.reshape(-1).float()
    scanned = SelectiveScan.apply(
        sequences,
        delta,
        state_a,
        state_b,
        state_c,
        skip,
        delta_bias,
        True,
        1,
    ).reshape(batch, 4, -1, length)
    merged = _merge_aligned_scan_outputs(scanned, height, width)
    merged = merged.permute(0, 2, 3, 1).contiguous()
    return module.out_norm(merged).to(features.dtype)


class _AlignedSS2DCross(SS2D_cross_new):
    """SS2D cross core with explicit inverse mapping for all scan directions."""

    def forward_corev2(
        self, x1, x2, nrows=-1, channel_first=False, step_size=1
    ):
        if not channel_first:
            x1 = x1.permute(0, 3, 1, 2).contiguous()
            x2 = x2.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x1 = self.in_rank(x1)
            x2 = self.in_rank(x2)
        y1 = _aligned_selective_scan_branch(self, x1)
        y2 = _aligned_selective_scan_branch(self, x2)
        if self.ssm_low_rank:
            y1 = self.out_rank(y1)
            y2 = self.out_rank(y2)
        return y1, y2


class _ChannelLayerNorm(nn.Module):
    """LayerNorm over channels while preserving a channels-first TF layout."""

    def __init__(self, channels):
        super().__init__()
        self.channels = int(channels)
        self.norm = nn.LayerNorm(self.channels)

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(
                f'Expected [B, {self.channels}, H, W], got {tuple(features.shape)}.'
            )
        return self.norm(
            features.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2).contiguous()


class _BoundedChannelLayerScale(nn.Module):
    """Per-channel residual scale with a hard, inspectable magnitude budget."""

    def __init__(self, channels, initial_scale, max_scale):
        super().__init__()
        if not 0.0 <= abs(initial_scale) < max_scale:
            raise ValueError('LayerScale initialization must be inside its budget.')
        self.channels = int(channels)
        self.max_scale = float(max_scale)
        initial_logit = math.atanh(float(initial_scale) / self.max_scale)
        self.logit = nn.Parameter(torch.full((self.channels,), initial_logit))
        self.logit._no_weight_decay = True

    def values(self):
        return self.max_scale * torch.tanh(self.logit)

    def forward(self, residual):
        if residual.ndim != 4 or residual.shape[1] != self.channels:
            raise ValueError(
                f'Expected [B, {self.channels}, H, W], got {tuple(residual.shape)}.'
            )
        return self.values().view(1, -1, 1, 1) * residual


def _init_small(module, std=1e-3, bias=0.0):
    nn.init.normal_(module.weight, mean=0.0, std=std)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)


class _AxisPath(nn.Module):
    """Axis-specialized residual model used after the stage transition."""

    def __init__(self, cfg, channels, frequency_first):
        super().__init__()
        first = FMambaBlock if frequency_first else TMambaBlock
        second = TMambaBlock if frequency_first else FMambaBlock
        self.pre_norm = _ChannelLayerNorm(channels)
        self.axis_blocks = nn.ModuleList((first(cfg, channels), second(cfg, channels)))
        self.residual_scale = _scale_parameter(0.10)

    def forward(self, features):
        residual = self.pre_norm(features)
        for block in self.axis_blocks:
            residual = block(residual)
        return features + torch.tanh(self.residual_scale) * residual


class _DownTransition(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                channels, channels, 3, stride=2, padding=1,
                groups=channels, bias=False,
            ),
            nn.Conv2d(channels, channels, 1, bias=False),
            _ChannelLayerNorm(channels),
            nn.GELU(),
        )

    def forward(self, features):
        return self.body(features)


class _UpSkipTransition(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            channels, channels, 3, stride=2, padding=1, bias=False
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            _ChannelLayerNorm(channels),
            nn.GELU(),
        )
        self.layer_scale = _BoundedChannelLayerScale(
            channels, initial_scale=0.10, max_scale=1.0
        )

    def forward(self, features, skip):
        features = self.up(features, output_size=skip.shape)
        if features.shape != skip.shape:
            raise RuntimeError(
                f'Full-resolution skip mismatch: {tuple(features.shape)} != '
                f'{tuple(skip.shape)}.'
            )
        fused = self.fuse(torch.cat((features, skip), dim=1))
        return skip + self.layer_scale(fused)


class _LatentInjection(nn.Module):
    """Project a generator latent and inject it through a small learnable gate."""

    def __init__(self, in_channels, out_channels, initial_scale=0.05):
        super().__init__()
        self.in_channels = int(in_channels)
        self.projection = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = _ChannelLayerNorm(out_channels)
        self.scale = _scale_parameter(initial_scale)

    def forward(self, target, latent):
        if latent.ndim != 4 or latent.shape[1] != self.in_channels:
            raise ValueError(
                f'Latent must have shape [B, {self.in_channels}, H, W], got '
                f'{tuple(latent.shape)}.'
            )
        if latent.shape[0] != target.shape[0]:
            raise ValueError('Latent and target batch dimensions differ.')
        update = self.projection(latent)
        if update.shape[2:] != target.shape[2:]:
            update = F.interpolate(
                update, size=target.shape[2:], mode='bilinear', align_corners=False
            )
        update = self.norm(update)
        return target + torch.tanh(self.scale) * update


class _ProjectedCrossInteraction(nn.Module):
    """Device-safe Cross/VSS interaction with explicit bidirectional exchange."""

    def __init__(
        self,
        mag_channels,
        phase_channels,
        common_channels,
        d_state,
    ):
        super().__init__()
        self.common_channels = int(common_channels)
        self.mag_projection = nn.Conv2d(mag_channels, common_channels, 1, bias=False)
        self.phase_projection = nn.Conv2d(phase_channels, common_channels, 1, bias=False)
        self.mag_local = nn.Conv2d(
            common_channels, common_channels, 3, padding=1,
            groups=common_channels, bias=False
        )
        self.phase_local = nn.Conv2d(
            common_channels, common_channels, 3, padding=1,
            groups=common_channels, bias=False
        )
        self.mag_norm = nn.LayerNorm(common_channels)
        self.phase_norm = nn.LayerNorm(common_channels)
        self.cross = _AlignedSS2DCross(
            d_model=common_channels,
            d_state=d_state,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            d_conv=3,
            dropout=0.0,
        )
        self.mag_eca = eca_layer(common_channels)
        self.phase_eca = eca_layer(common_channels)
        self.mag_calibration = _ChannelLayerNorm(common_channels)
        self.phase_calibration = _ChannelLayerNorm(common_channels)
        self.mag_back_projection = nn.Conv2d(
            common_channels, mag_channels, 1, bias=False
        )
        self.phase_back_projection = nn.Conv2d(
            common_channels, phase_channels, 1, bias=False
        )
        self.mag_layer_scale = _BoundedChannelLayerScale(
            mag_channels, initial_scale=0.03, max_scale=0.25
        )
        self.phase_layer_scale = _BoundedChannelLayerScale(
            phase_channels, initial_scale=0.03, max_scale=0.25
        )

    def forward(self, mag_features, phase_features):
        mag_common = self.mag_projection(mag_features)
        phase_common = self.phase_projection(phase_features)
        mag_common = mag_common + self.mag_local(mag_common)
        phase_common = phase_common + self.phase_local(phase_common)
        mag_scan, phase_scan = self.cross(
            self.mag_norm(mag_common.permute(0, 2, 3, 1)),
            self.phase_norm(phase_common.permute(0, 2, 3, 1)),
        )
        mag_scan = mag_scan.permute(0, 3, 1, 2)
        phase_scan = phase_scan.permute(0, 3, 1, 2)
        mag_update = self.mag_back_projection(
            self.mag_calibration(self.mag_eca(phase_scan))
        )
        phase_update = self.phase_back_projection(
            self.phase_calibration(self.phase_eca(mag_scan))
        )
        return (
            mag_features + self.mag_layer_scale(mag_update),
            phase_features + self.phase_layer_scale(phase_update),
        )


class _CompressedMagnitudeDenseBridge(nn.Module):
    """Fuse Stage-2 context before Stage-3 compressed-domain modeling."""

    def __init__(self, channels):
        super().__init__()
        self.channels = int(channels)
        self.hidden_channels = max(8, self.channels // 4)
        self.fusion = nn.Sequential(
            _ChannelLayerNorm(self.channels * 2),
            nn.Conv2d(self.channels * 2, self.hidden_channels, 1, bias=False),
            nn.PReLU(self.hidden_channels),
            nn.Conv2d(
                self.hidden_channels,
                self.hidden_channels,
                3,
                padding=1,
                groups=self.hidden_channels,
                bias=False,
            ),
            _ChannelLayerNorm(self.hidden_channels),
            nn.PReLU(self.hidden_channels),
            nn.Conv2d(self.hidden_channels, self.channels, 1, bias=False),
        )
        self.residual_scale = _scale_parameter(0.05)

    def forward(self, target, context):
        expected_channels = self.channels
        for name, value in (('target', target), ('context', context)):
            if value.ndim != 4 or value.shape[1] != expected_channels:
                raise ValueError(
                    f'{name} must have shape [B, {expected_channels}, H, W], '
                    f'got {tuple(value.shape)}.'
                )
        if target.shape != context.shape:
            raise ValueError(
                f'Compressed dense bridge shapes differ: {tuple(target.shape)} '
                f'vs {tuple(context.shape)}.'
            )
        fused = self.fusion(torch.cat((target, context), dim=1))
        if fused.shape != target.shape:
            raise RuntimeError(
                f'Compressed dense bridge output shape mismatch: '
                f'{tuple(fused.shape)} != {tuple(target.shape)}.'
            )
        return target + torch.tanh(self.residual_scale) * fused


class _PairedStage(nn.Module):
    """One persistent-resolution stage in the asymmetric backend."""

    def __init__(
        self, cfg, mag_channels, phase_channels, mode, common_channels=None,
        compressed_dense_bridge_enabled=False,
    ):
        super().__init__()
        if mode not in ('full', 'down', 'half', 'up'):
            raise ValueError(f'Unsupported stage mode: {mode}.')
        self.mode = mode
        self.common_channels = common_channels
        self.mag_path = _AxisPath(cfg, mag_channels, frequency_first=True)
        self.phase_path = _AxisPath(cfg, phase_channels, frequency_first=False)
        self.mag_down = _DownTransition(mag_channels) if mode == 'down' else None
        self.phase_down = _DownTransition(phase_channels) if mode == 'down' else None
        self.mag_up = _UpSkipTransition(mag_channels) if mode == 'up' else None
        self.phase_up = _UpSkipTransition(phase_channels) if mode == 'up' else None
        self.interaction = None
        if common_channels is not None:
            self.interaction = _ProjectedCrossInteraction(
                mag_channels,
                phase_channels,
                common_channels,
                d_state=int(cfg['model_cfg']['d_state']),
            )
        self.interaction_position = (
            'none' if self.interaction is None
            else 'compressed' if mode in ('down', 'half')
            else 'full_resolution'
        )
        self.compressed_mag_dense_bridge = None
        if compressed_dense_bridge_enabled:
            if mode != 'half':
                raise ValueError(
                    'The dense bridge requires the half-resolution Stage-3 path.'
                )
            rng_state = torch.random.get_rng_state()
            try:
                self.compressed_mag_dense_bridge = (
                    _CompressedMagnitudeDenseBridge(mag_channels)
                )
            finally:
                torch.random.set_rng_state(rng_state)

    def forward(
        self, mag_features, phase_features, mag_dense_context=None,
        mag_skip=None, phase_skip=None,
    ):
        if self.mode == 'down':
            mag_features = self.mag_down(mag_features)
            phase_features = self.phase_down(phase_features)
        elif self.mode == 'up':
            if mag_skip is None or phase_skip is None:
                raise ValueError('Stage-4 upsampling requires both Stage-1 skips.')
            mag_features = self.mag_up(mag_features, mag_skip)
            phase_features = self.phase_up(phase_features, phase_skip)
        if self.compressed_mag_dense_bridge is not None:
            if mag_dense_context is None:
                raise ValueError('Stage-3 dense bridge requires Stage-2 context.')
            mag_features = self.compressed_mag_dense_bridge(
                mag_features, mag_dense_context
            )
        mag_features = self.mag_path(mag_features)
        phase_features = self.phase_path(phase_features)
        if self.interaction is not None:
            mag_features, phase_features = self.interaction(
                mag_features, phase_features
            )
        return mag_features, phase_features


class AsymmetricPolarZipRefine(nn.Module):
    """Evidence-conditioned four-stage 80/40-channel mag/phase backend."""

    compression_ratios = (1, 2, 2, 1)
    stage_modes = ('full', 'down', 'half', 'up')
    evidence_channels = 20

    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg['model_cfg']
        self.mag_channels = int(
            model_cfg.get('asymmetric_polar_zip_refine_mag_channels', 80)
        )
        self.phase_channels = int(
            model_cfg.get('asymmetric_polar_zip_refine_phase_channels', 40)
        )
        self.stage_common_channels = tuple(
            model_cfg.get(
                'asymmetric_polar_zip_refine_stage_common_channels',
                [0, 64, 64, 40],
            )
        )
        self.refiner_expand = int(
            model_cfg.get('asymmetric_polar_zip_refine_expand', 2)
        )
        self.eps = float(
            model_cfg.get('asymmetric_polar_zip_refine_eps', 1e-6)
        )
        self.phase_eps = float(model_cfg.get('phase_eps', 1e-3))
        self.delta_limit = float(
            model_cfg.get('asymmetric_polar_zip_refine_delta_limit', 1.0)
        )
        self.additive_limit = float(
            model_cfg.get(
                'asymmetric_polar_zip_refine_additive_mag_scale',
                model_cfg.get('asymmetric_polar_zip_refine_additive_limit', 0.5),
            )
        )
        self.magnitude_floor = float(
            model_cfg.get(
                'asymmetric_polar_zip_refine_reference_floor',
                model_cfg.get('asymmetric_polar_zip_refine_magnitude_floor', 1e-4),
            )
        )
        self.phase_delta_limit = float(
            model_cfg.get('asymmetric_polar_zip_refine_phase_delta_limit', 1.0)
        )
        self.ri_residual_ratio = float(
            model_cfg.get('asymmetric_polar_zip_refine_ri_residual_ratio', 0.25)
        )
        self.noisy_reference_absorption = float(
            model_cfg.get(
                'asymmetric_polar_zip_refine_noisy_reference_absorption', 0.25
            )
        )
        self.context_initial_scale = float(
            model_cfg.get('asymmetric_polar_zip_refine_context_scale', 0.05)
        )
        self.persistent_backbone = bool(
            model_cfg.get('asymmetric_polar_zip_refine_persistent_backbone', True)
        )
        self.demand_gate_bias = float(
            model_cfg.get('asymmetric_polar_zip_refine_demand_gate_bias', -1.5)
        )
        self.activation_checkpointing = bool(
            model_cfg.get(
                'asymmetric_polar_zip_refine_activation_checkpointing', False
            )
        )
        self.compressed_dense_bridge_enabled = bool(
            model_cfg.get(
                'asymmetric_polar_zip_refine_compressed_dense_bridge_enabled',
                False,
            )
        )
        if self.mag_channels <= 0 or self.phase_channels <= 0:
            raise ValueError('Magnitude and phase refiner widths must be positive.')
        if len(self.stage_common_channels) != len(self.stage_modes):
            raise ValueError('One interaction width is required for each refiner stage.')
        if any(int(width) < 0 for width in self.stage_common_channels):
            raise ValueError('Refiner interaction widths must be non-negative.')
        if self.refiner_expand <= 0:
            raise ValueError('The refiner-specific Mamba expand must be positive.')
        if self.eps <= 0.0:
            raise ValueError('asymmetric_polar_zip_refine_eps must be positive.')
        if self.phase_eps <= 0.0:
            raise ValueError('phase_eps must be positive.')
        if not 0.0 < self.delta_limit <= 8.0:
            raise ValueError('delta_limit must be in (0, 8].')
        if self.additive_limit <= 0.0 or self.magnitude_floor <= 0.0:
            raise ValueError('Magnitude additive limit and floor must be positive.')
        if not 0.0 < self.phase_delta_limit <= math.pi:
            raise ValueError('phase_delta_limit must be in (0, pi].')
        if not 0.0 < self.ri_residual_ratio <= 0.25:
            raise ValueError(
                'The fixed RI residual ratio must be in (0, 0.25].'
            )
        if not 0.0 <= self.noisy_reference_absorption <= 0.25:
            raise ValueError('Noisy reference absorption must be in [0, 0.25].')
        if not 0.0 < self.context_initial_scale < 1.0:
            raise ValueError('Refiner context scale must be in (0, 1).')
        if not self.persistent_backbone:
            raise ValueError(
                'The approved refiner requires persistent full-half-half-full stages.'
            )

        refiner_cfg = deepcopy(cfg)
        refiner_cfg['model_cfg']['expand'] = self.refiner_expand
        self.core_cfg = refiner_cfg

        mag_base = int(model_cfg['hid_feature'])
        restore_base = max(
            1, int(round(
                mag_base * float(model_cfg.get('restoration_width_ratio', 1.0))
            ))
        )
        self.evidence_specs = {
            'mag_final': (mag_base, 'full'),
            'restore_final': (restore_base, 'full'),
            'suppress_bottleneck': (mag_base * 3, 'bottleneck'),
            'restore_bottleneck': (restore_base * 3, 'bottleneck'),
        }

        self.mag_stem = nn.Sequential(
            nn.Conv2d(self.evidence_channels, self.mag_channels, 3, padding=1),
            _ChannelLayerNorm(self.mag_channels),
            nn.GELU(),
        )
        self.phase_stem = nn.Sequential(
            nn.Conv2d(self.evidence_channels, self.phase_channels, 3, padding=1),
            _ChannelLayerNorm(self.phase_channels),
            nn.GELU(),
        )
        self.full_context_injections = nn.ModuleDict({
            'mag_from_mag': _LatentInjection(
                mag_base, self.mag_channels, self.context_initial_scale
            ),
            'mag_from_restore': _LatentInjection(
                restore_base, self.mag_channels, self.context_initial_scale
            ),
            'phase_from_mag': _LatentInjection(
                mag_base, self.phase_channels, self.context_initial_scale
            ),
            'phase_from_restore': _LatentInjection(
                restore_base, self.phase_channels, self.context_initial_scale
            ),
        })
        self.bottleneck_context_injections = nn.ModuleDict({
            'mag_from_suppress': _LatentInjection(
                mag_base * 3, self.mag_channels, self.context_initial_scale
            ),
            'mag_from_restore': _LatentInjection(
                restore_base * 3, self.mag_channels, self.context_initial_scale
            ),
            'phase_from_suppress': _LatentInjection(
                mag_base * 3, self.phase_channels, self.context_initial_scale
            ),
            'phase_from_restore': _LatentInjection(
                restore_base * 3, self.phase_channels, self.context_initial_scale
            ),
        })
        self.paired_stages = nn.ModuleList([
            _PairedStage(
                refiner_cfg,
                self.mag_channels,
                self.phase_channels,
                mode,
                common_channels or None,
                compressed_dense_bridge_enabled=(
                    self.compressed_dense_bridge_enabled and stage_index == 2
                ),
            )
            for stage_index, (mode, common_channels) in enumerate(zip(
                self.stage_modes, self.stage_common_channels
            ))
        ])
        self.delta_log_mag_head = nn.Conv2d(self.mag_channels, 1, 1)
        self.delta_add_mag_head = nn.Conv2d(self.mag_channels, 1, 1)
        self.phase_delta_head = nn.Conv2d(self.phase_channels, 1, 1)
        self.mag_demand_head = nn.Conv2d(self.mag_channels, 1, 1)
        self.phase_demand_head = nn.Conv2d(self.phase_channels, 1, 1)
        for head in (
            self.delta_log_mag_head, self.delta_add_mag_head,
            self.phase_delta_head,
        ):
            _init_small(head)
        _init_small(self.mag_demand_head, bias=self.demand_gate_bias)
        _init_small(self.phase_demand_head, bias=self.demand_gate_bias)

        self.ri_feature_fusion = nn.Sequential(
            nn.Conv2d(
                self.mag_channels + self.phase_channels,
                self.phase_channels,
                1,
                bias=False,
            ),
            _ChannelLayerNorm(self.phase_channels),
            nn.GELU(),
        )
        ri_input_channels = self.phase_channels + 8
        self.ri_residual_head = nn.Sequential(
            _ChannelLayerNorm(ri_input_channels),
            nn.Conv2d(
                ri_input_channels,
                ri_input_channels,
                3,
                padding=1,
                groups=ri_input_channels,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(ri_input_channels, 2, 1),
        )
        self.ri_demand_head = nn.Conv2d(ri_input_channels, 1, 1)
        _init_small(self.ri_residual_head[-1])
        _init_small(self.ri_demand_head, bias=self.demand_gate_bias)

    @property
    def outer_mag_gate(self):
        return self.mag_demand_head.bias

    @property
    def outer_phase_gate(self):
        return self.phase_demand_head.bias

    @property
    def ri_residual_gate(self):
        return self.ri_demand_head

    @property
    def rotation_head(self):
        return self.phase_delta_head

    @property
    def compressed_mag_dense_bridge(self):
        return self.paired_stages[2].compressed_mag_dense_bridge

    def _run_stage(self, stage, *arguments):
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(stage, *arguments, use_reentrant=False)
        return stage(*arguments)

    @staticmethod
    def _complex_magnitude(value):
        return torch.linalg.vector_norm(value, dim=1, keepdim=True)

    def _validate_evidence(self, noisy_complex, evidence):
        if not isinstance(evidence, dict):
            raise TypeError('evidence must be a dict of generator tensors.')
        required = {
            'coarse_complex', 'base_minus_coarse', 'harmonic_prior',
            'voicing_map', 'restoration_gates', 'mag_final', 'restore_final',
            'suppress_bottleneck', 'restore_bottleneck',
        }
        missing = sorted(required.difference(evidence))
        extra = sorted(set(evidence).difference(required))
        if missing or extra:
            raise ValueError(
                f'Evidence keys differ; missing={missing}, unexpected={extra}.'
            )
        batch, _, frames, bins = noisy_complex.shape
        full_shapes = {
            'coarse_complex': (batch, 2, frames, bins),
            'base_minus_coarse': (batch, 2, frames, bins),
            'harmonic_prior': (batch, 1, frames, bins),
            'voicing_map': (batch, 1, frames, bins),
            'restoration_gates': (batch, 2, frames, bins),
        }
        for name, expected_shape in full_shapes.items():
            value = evidence[name]
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f'{name} must have shape {expected_shape}, got {tuple(value.shape)}.'
                )
        encoded_bins = (bins + 1) // 2
        latent_shapes = {
            'mag_final': (batch, self.evidence_specs['mag_final'][0], frames, encoded_bins),
            'restore_final': (
                batch, self.evidence_specs['restore_final'][0], frames, encoded_bins
            ),
            'suppress_bottleneck': (
                batch, self.evidence_specs['suppress_bottleneck'][0],
                frames // 4, encoded_bins // 4,
            ),
            'restore_bottleneck': (
                batch, self.evidence_specs['restore_bottleneck'][0],
                frames // 4, encoded_bins // 4,
            ),
        }
        for name, expected_shape in latent_shapes.items():
            value = evidence[name]
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f'{name} must have shape {expected_shape}, got {tuple(value.shape)}.'
                )
        for name in required:
            value = evidence[name]
            if value.device != noisy_complex.device:
                raise ValueError(f'{name} must match noisy_complex device.')
            if not torch.isfinite(value).all():
                raise RuntimeError(f'{name} contains NaN/Inf.')

    def build_evidence_input(self, noisy_complex, base_complex, evidence):
        for name, value in (
            ('noisy_complex', noisy_complex), ('base_complex', base_complex)
        ):
            if value.ndim != 4 or value.shape[1] != 2:
                raise ValueError(f'{name} must have shape [B, 2, T, F].')
        if noisy_complex.shape != base_complex.shape:
            raise ValueError(
                f'Complex input shapes differ: {tuple(noisy_complex.shape)} vs '
                f'{tuple(base_complex.shape)}'
            )
        self._validate_evidence(noisy_complex, evidence)
        coarse_complex = evidence['coarse_complex']
        noisy_mag = self._complex_magnitude(noisy_complex)
        base_mag = self._complex_magnitude(base_complex)
        coarse_mag = self._complex_magnitude(coarse_complex)
        safe_noisy_unit = noisy_complex / noisy_mag.clamp_min(self.eps)
        identity_unit = torch.zeros_like(noisy_complex)
        identity_unit[:, :1] = 1.0
        noisy_unit = torch.where(noisy_mag > self.phase_eps, safe_noisy_unit, identity_unit)
        base_unit = torch.where(
            base_mag > self.phase_eps,
            base_complex / base_mag.clamp_min(self.eps),
            noisy_unit,
        )
        noisy_real, noisy_imag = torch.chunk(noisy_unit, 2, dim=1)
        base_real, base_imag = torch.chunk(base_unit, 2, dim=1)
        relative_phase = torch.cat((
            noisy_real * base_real + noisy_imag * base_imag,
            noisy_imag * base_real - noisy_real * base_imag,
        ), dim=1)
        maps = torch.cat((
            noisy_complex,
            base_complex,
            noisy_complex - base_complex,
            coarse_complex,
            evidence['base_minus_coarse'],
            torch.log1p(noisy_mag),
            torch.log1p(base_mag),
            torch.log1p(coarse_mag),
            torch.log(
                (base_mag + self.eps) / (noisy_mag + self.eps)
            ).clamp(-8.0, 8.0),
            relative_phase,
            evidence['harmonic_prior'],
            evidence['voicing_map'],
            evidence['restoration_gates'],
        ), dim=1)
        if maps.shape[1] != self.evidence_channels:
            raise RuntimeError(
                f'Expected {self.evidence_channels} evidence maps, got {maps.shape[1]}.'
            )
        return maps

    def forward(self, noisy_complex, base_complex, evidence):
        maps = self.build_evidence_input(noisy_complex, base_complex, evidence)
        if not torch.isfinite(maps).all():
            raise RuntimeError('Asymmetric polar refiner input contains NaN/Inf.')

        mag_features = self.mag_stem(maps)
        phase_features = self.phase_stem(maps)
        mag_features = self.full_context_injections['mag_from_mag'](
            mag_features, evidence['mag_final']
        )
        mag_features = self.full_context_injections['mag_from_restore'](
            mag_features, evidence['restore_final']
        )
        phase_features = self.full_context_injections['phase_from_mag'](
            phase_features, evidence['mag_final']
        )
        phase_features = self.full_context_injections['phase_from_restore'](
            phase_features, evidence['restore_final']
        )

        mag_features, phase_features = self._run_stage(
            self.paired_stages[0], mag_features, phase_features
        )
        mag_stage1_skip, phase_stage1_skip = mag_features, phase_features
        mag_features, phase_features = self._run_stage(
            self.paired_stages[1], mag_features, phase_features
        )
        stage2_dense_context = mag_features
        mag_features = self.bottleneck_context_injections['mag_from_suppress'](
            mag_features, evidence['suppress_bottleneck']
        )
        mag_features = self.bottleneck_context_injections['mag_from_restore'](
            mag_features, evidence['restore_bottleneck']
        )
        phase_features = self.bottleneck_context_injections['phase_from_suppress'](
            phase_features, evidence['suppress_bottleneck']
        )
        phase_features = self.bottleneck_context_injections['phase_from_restore'](
            phase_features, evidence['restore_bottleneck']
        )
        if self.compressed_dense_bridge_enabled:
            mag_features, phase_features = self._run_stage(
                self.paired_stages[2], mag_features, phase_features,
                stage2_dense_context,
            )
        else:
            mag_features, phase_features = self._run_stage(
                self.paired_stages[2], mag_features, phase_features
            )
        mag_features, phase_features = self._run_stage(
            self.paired_stages[3], mag_features, phase_features,
            None, mag_stage1_skip, phase_stage1_skip,
        )

        mag_demand = torch.sigmoid(self.mag_demand_head(mag_features))
        phase_demand = torch.sigmoid(self.phase_demand_head(phase_features))
        raw_mag_multiplicative = self.delta_limit * torch.tanh(
            self.delta_log_mag_head(mag_features)
        )
        noisy_mag = self._complex_magnitude(noisy_complex)
        base_mag = self._complex_magnitude(base_complex)
        coarse_mag = self._complex_magnitude(evidence['coarse_complex'])
        utterance_floor = self.magnitude_floor * noisy_mag.mean(
            dim=(2, 3), keepdim=True
        ).clamp_min(self.eps)
        trusted_magnitude_reference = torch.maximum(base_mag, coarse_mag) + utterance_floor
        magnitude_reference = trusted_magnitude_reference + (
            self.noisy_reference_absorption
            * torch.relu(noisy_mag - trusted_magnitude_reference)
        )
        raw_mag_additive = self.additive_limit * magnitude_reference * torch.tanh(
            self.delta_add_mag_head(mag_features)
        )
        applied_mag_multiplicative = mag_demand * raw_mag_multiplicative
        applied_mag_additive = mag_demand * raw_mag_additive
        corrected_mag = (
            base_mag * torch.exp(applied_mag_multiplicative)
            + applied_mag_additive
        ).clamp_min(0.0)

        raw_phase_delta = self.phase_delta_limit * torch.tanh(
            self.phase_delta_head(phase_features)
        )
        applied_phase_delta = phase_demand * raw_phase_delta
        noisy_unit = noisy_complex / noisy_mag.clamp_min(self.eps)
        identity_unit = torch.zeros_like(noisy_complex)
        identity_unit[:, :1] = 1.0
        noisy_unit = torch.where(noisy_mag > self.phase_eps, noisy_unit, identity_unit)
        base_unit = torch.where(
            base_mag > self.phase_eps,
            base_complex / base_mag.clamp_min(self.eps),
            noisy_unit,
        )
        base_real, base_imag = torch.chunk(base_unit, 2, dim=1)
        delta_cos = torch.cos(applied_phase_delta)
        delta_sin = torch.sin(applied_phase_delta)
        polar_unit = torch.cat((
            base_real * delta_cos - base_imag * delta_sin,
            base_real * delta_sin + base_imag * delta_cos,
        ), dim=1)
        polar_complex = corrected_mag * polar_unit

        ri_features = self.ri_feature_fusion(
            torch.cat((mag_features, phase_features), dim=1)
        )
        ri_input = torch.cat((
            ri_features,
            noisy_complex,
            base_complex,
            polar_complex,
            noisy_complex - polar_complex,
        ), dim=1)
        ri_raw = torch.tanh(self.ri_residual_head(ri_input))
        ri_demand = torch.sigmoid(self.ri_demand_head(ri_input))
        ri_residual_ratio = polar_complex.new_tensor(self.ri_residual_ratio)
        applied_ri_residual = (
            ri_residual_ratio * magnitude_reference * ri_demand * ri_raw
        )
        refined_complex = polar_complex + applied_ri_residual

        finite_outputs = {
            'raw_mag_multiplicative': raw_mag_multiplicative,
            'applied_mag_multiplicative': applied_mag_multiplicative,
            'raw_mag_additive': raw_mag_additive,
            'applied_mag_additive': applied_mag_additive,
            'corrected_mag': corrected_mag,
            'raw_phase_delta': raw_phase_delta,
            'applied_phase_delta': applied_phase_delta,
            'polar_complex': polar_complex,
            'mag_demand': mag_demand,
            'phase_demand': phase_demand,
            'ri_raw': ri_raw,
            'ri_demand': ri_demand,
            'applied_ri_residual': applied_ri_residual,
            'refined_complex': refined_complex,
        }
        for name, value in finite_outputs.items():
            if not torch.isfinite(value).all():
                raise RuntimeError(f'Asymmetric polar {name} contains NaN/Inf.')

        aux = {
            'base_complex': base_complex,
            'coarse_complex': evidence['coarse_complex'],
            'raw_mag_multiplicative': raw_mag_multiplicative,
            'applied_mag_multiplicative': applied_mag_multiplicative,
            'raw_mag_additive': raw_mag_additive,
            'applied_mag_additive': applied_mag_additive,
            'corrected_magnitude': corrected_mag,
            'trusted_magnitude_reference': trusted_magnitude_reference,
            'magnitude_reference': magnitude_reference,
            'raw_phase_delta': raw_phase_delta,
            'applied_phase_delta': applied_phase_delta,
            'polar_complex': polar_complex,
            'mag_demand_gate': mag_demand,
            'phase_demand_gate': phase_demand,
            'ri_demand_gate': ri_demand,
            'ri_residual_raw': ri_raw,
            'ri_residual_applied': applied_ri_residual,
            'ri_residual_ratio': ri_residual_ratio,
            'refined_complex': refined_complex,
            'asymmetric_mag_stage_scales': torch.stack([
                torch.tanh(stage.mag_path.residual_scale)
                for stage in self.paired_stages
            ]),
            'asymmetric_phase_stage_scales': torch.stack([
                torch.tanh(stage.phase_path.residual_scale)
                for stage in self.paired_stages
            ]),
            'asymmetric_interaction_mag_scales': torch.stack([
                stage.interaction.mag_layer_scale.values()
                for stage in self.paired_stages
                if stage.interaction is not None
            ]),
            'asymmetric_interaction_phase_scales': torch.stack([
                stage.interaction.phase_layer_scale.values()
                for stage in self.paired_stages
                if stage.interaction is not None
            ]),
            'asymmetric_upskip_mag_scale': (
                self.paired_stages[3].mag_up.layer_scale.values()
            ),
            'asymmetric_upskip_phase_scale': (
                self.paired_stages[3].phase_up.layer_scale.values()
            ),
            'asymmetric_dense_bridge_scale': (
                torch.tanh(
                    self.compressed_mag_dense_bridge.residual_scale
                )
                if self.compressed_mag_dense_bridge is not None
                else base_complex.new_zeros(())
            ),
            'asymmetric_context_scales': torch.stack([
                torch.tanh(module.scale)
                for module in (
                    *self.full_context_injections.values(),
                    *self.bottleneck_context_injections.values(),
                )
            ]),
        }
        return refined_complex, aux
