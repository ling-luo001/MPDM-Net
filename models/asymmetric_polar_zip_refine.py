"""Asymmetric magnitude/phase polar refinement for Residual-Dense MPDM-Net.

The parent generator and its S0 construction stay untouched. This optional
post-refiner consumes X and S0 as eight real-valued maps, runs asymmetric
frequency/time towers, then applies an exactly identity-initialized polar
correction followed by the zero-head A/B complex residual path.
"""

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


class _AxisPath(nn.Module):
    """One branch of a paired stage, excluding cross-branch interaction."""

    def __init__(self, cfg, channels, ratio, frequency_first):
        super().__init__()
        if ratio not in (1, 2):
            raise ValueError(f'Unsupported compression ratio: {ratio}')
        self.ratio = ratio
        first = FMambaBlock if frequency_first else TMambaBlock
        second = TMambaBlock if frequency_first else FMambaBlock
        self.pre_norm = nn.GroupNorm(1, channels)
        self.axis_blocks = nn.ModuleList((first(cfg, channels), second(cfg, channels)))
        self.same_resolution = None
        self.down = None
        self.up = None
        if ratio == 2:
            self.same_resolution = nn.Sequential(
                nn.Conv2d(
                    channels, channels, 3, padding=1, groups=channels, bias=False
                ),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.GELU(),
            )
            self.down = nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(1, channels),
                nn.GELU(),
            )
            self.up = nn.ConvTranspose2d(
                channels, channels, 3, stride=2, padding=1, bias=False
            )
        self.residual_scale = _scale_parameter(0.10)

    def encode(self, features):
        residual = self.pre_norm(features)
        same_resolution = None
        if self.ratio == 2:
            same_resolution = self.same_resolution(residual)
            residual = self.down(residual)
        for block in self.axis_blocks:
            residual = block(residual)
        return residual, same_resolution

    def restore(self, source, residual, same_resolution):
        if self.ratio == 2:
            residual = self.up(residual, output_size=source.shape)
            if residual.shape != source.shape:
                raise RuntimeError(
                    f'Zip restoration shape mismatch: {tuple(residual.shape)} != '
                    f'{tuple(source.shape)}'
                )
            residual = residual + same_resolution
        return source + torch.tanh(self.residual_scale) * residual


class _ProjectedCrossInteraction(nn.Module):
    """Device-safe Cross/VSS interaction with explicit bidirectional exchange."""

    def __init__(
        self,
        mag_channels,
        phase_channels,
        common_channels,
        d_state,
        gate_bias,
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
        self.mag_cross_gate = nn.Conv2d(common_channels * 2, common_channels, 1)
        self.phase_cross_gate = nn.Conv2d(common_channels * 2, common_channels, 1)
        nn.init.zeros_(self.mag_cross_gate.weight)
        nn.init.zeros_(self.phase_cross_gate.weight)
        nn.init.constant_(self.mag_cross_gate.bias, gate_bias)
        nn.init.constant_(self.phase_cross_gate.bias, gate_bias)
        self.mag_eca = eca_layer(common_channels)
        self.phase_eca = eca_layer(common_channels)
        self.mag_calibration = nn.GroupNorm(1, common_channels)
        self.phase_calibration = nn.GroupNorm(1, common_channels)
        self.mag_back_projection = nn.Conv2d(
            common_channels, mag_channels, 1, bias=False
        )
        self.phase_back_projection = nn.Conv2d(
            common_channels, phase_channels, 1, bias=False
        )
        self.mag_scale = _scale_parameter(0.05)
        self.phase_scale = _scale_parameter(0.05)

    def forward(self, mag_features, phase_features):
        mag_common = self.mag_projection(mag_features)
        phase_common = self.phase_projection(phase_features)
        mag_common = mag_common + self.mag_local(mag_common)
        phase_common = phase_common + self.phase_local(phase_common)
        mag_cross, phase_cross = self.cross(
            self.mag_norm(mag_common.permute(0, 2, 3, 1)),
            self.phase_norm(phase_common.permute(0, 2, 3, 1)),
        )
        mag_cross = mag_cross.permute(0, 3, 1, 2)
        phase_cross = phase_cross.permute(0, 3, 1, 2)
        joint = torch.cat((mag_cross, phase_cross), dim=1)
        mag_mixed = mag_cross + torch.sigmoid(
            self.mag_cross_gate(joint)
        ) * phase_cross
        phase_mixed = phase_cross + torch.sigmoid(
            self.phase_cross_gate(joint)
        ) * mag_cross
        mag_update = self.mag_back_projection(
            self.mag_calibration(self.mag_eca(mag_mixed))
        )
        phase_update = self.phase_back_projection(
            self.phase_calibration(self.phase_eca(phase_mixed))
        )
        return (
            mag_features + torch.tanh(self.mag_scale) * mag_update,
            phase_features + torch.tanh(self.phase_scale) * phase_update,
        )


class _PairedStage(nn.Module):
    """Run both asymmetric paths and interact at the specified stage position."""

    def __init__(
        self, cfg, mag_channels, phase_channels, ratio, common_channels=None
    ):
        super().__init__()
        self.ratio = ratio
        self.common_channels = common_channels
        self.mag_path = _AxisPath(cfg, mag_channels, ratio, frequency_first=True)
        self.phase_path = _AxisPath(cfg, phase_channels, ratio, frequency_first=False)
        self.interaction = None
        if common_channels is not None:
            self.interaction = _ProjectedCrossInteraction(
                mag_channels,
                phase_channels,
                common_channels,
                d_state=int(cfg['model_cfg']['d_state']),
                gate_bias=float(
                    cfg['model_cfg'].get(
                        'asymmetric_polar_zip_refine_interaction_gate_bias', -2.0
                    )
                ),
            )
        self.interaction_position = (
            'none' if self.interaction is None
            else 'compressed_pre_up' if ratio == 2
            else 'full_resolution'
        )

    def forward(self, mag_features, phase_features):
        mag_residual, mag_same = self.mag_path.encode(mag_features)
        phase_residual, phase_same = self.phase_path.encode(phase_features)
        if self.interaction is not None:
            mag_residual, phase_residual = self.interaction(
                mag_residual, phase_residual
            )
        mag_output = self.mag_path.restore(
            mag_features, mag_residual, mag_same
        )
        phase_output = self.phase_path.restore(
            phase_features, phase_residual, phase_same
        )
        return mag_output, phase_output


class AsymmetricPolarZipRefine(nn.Module):
    """Four paired stages with an 80-channel mag and 40-channel phase tower."""

    compression_ratios = (1, 2, 2, 1)

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
        self.complex_residual_scale = float(
            model_cfg.get('asymmetric_polar_zip_refine_complex_residual_scale', 0.1)
        )
        self.complex_residual_gate_bias = float(
            model_cfg.get(
                'asymmetric_polar_zip_refine_complex_residual_gate_bias', -2.0
            )
        )
        self.activation_checkpointing = bool(
            model_cfg.get(
                'asymmetric_polar_zip_refine_activation_checkpointing', False
            )
        )
        if self.mag_channels != 80 or self.phase_channels != 40:
            raise ValueError('The approved asymmetric widths are magnitude=80, phase=40.')
        if self.stage_common_channels != (0, 64, 64, 40):
            raise ValueError('Approved common widths are [0, 64, 64, 40].')
        if self.refiner_expand != 2:
            raise ValueError('The approved refiner-specific expand is 2.')
        if self.eps <= 0.0:
            raise ValueError('asymmetric_polar_zip_refine_eps must be positive.')
        if self.phase_eps <= 0.0:
            raise ValueError('phase_eps must be positive.')
        if not 0.0 < self.delta_limit <= 8.0:
            raise ValueError('delta_limit must be in (0, 8].')
        if self.complex_residual_scale != 0.1:
            raise ValueError('The approved A/B complex residual scale is 0.1.')

        refiner_cfg = deepcopy(cfg)
        refiner_cfg['model_cfg']['expand'] = self.refiner_expand
        self.core_cfg = refiner_cfg

        self.mag_stem = nn.Sequential(
            nn.Conv2d(8, self.mag_channels, 3, padding=1),
            nn.GroupNorm(1, self.mag_channels),
            nn.GELU(),
        )
        self.phase_stem = nn.Sequential(
            nn.Conv2d(8, self.phase_channels, 3, padding=1),
            nn.GroupNorm(1, self.phase_channels),
            nn.GELU(),
        )
        self.paired_stages = nn.ModuleList([
            _PairedStage(
                refiner_cfg,
                self.mag_channels,
                self.phase_channels,
                ratio,
                common_channels or None,
            )
            for ratio, common_channels in zip(
                self.compression_ratios, self.stage_common_channels
            )
        ])

        self.delta_log_mag_head = nn.Conv2d(self.mag_channels, 1, 1)
        self.rotation_head = nn.Conv2d(self.phase_channels, 2, 1)
        self.outer_mag_gate = nn.Parameter(torch.zeros(()))
        self.outer_phase_gate = nn.Parameter(torch.zeros(()))

        self.ri_residual_head = nn.Sequential(
            nn.GroupNorm(1, self.phase_channels),
            nn.Conv2d(
                self.phase_channels,
                self.phase_channels,
                3,
                padding=1,
                groups=self.phase_channels,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(self.phase_channels, 2, 1),
        )
        self.ri_residual_gate = nn.Conv2d(self.phase_channels, 1, 1)
        nn.init.zeros_(self.ri_residual_head[-1].weight)
        nn.init.zeros_(self.ri_residual_head[-1].bias)
        nn.init.zeros_(self.ri_residual_gate.weight)
        nn.init.constant_(
            self.ri_residual_gate.bias, self.complex_residual_gate_bias
        )

    def _run_stage(self, stage, mag_features, phase_features):
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(
                stage, mag_features, phase_features, use_reentrant=False
            )
        return stage(mag_features, phase_features)

    def build_eight_map_input(self, noisy_complex, base_complex):
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
        noisy_mag = torch.linalg.vector_norm(noisy_complex, dim=1, keepdim=True)
        base_mag = torch.linalg.vector_norm(base_complex, dim=1, keepdim=True)
        maps = torch.cat((
            noisy_complex,
            base_complex,
            noisy_complex - base_complex,
            torch.log1p(noisy_mag),
            torch.log1p(base_mag),
        ), dim=1)
        if maps.shape[1] != 8:
            raise RuntimeError(f'Expected eight refinement maps, got {maps.shape[1]}.')
        return maps

    def forward(self, noisy_complex, base_complex):
        maps = self.build_eight_map_input(noisy_complex, base_complex)
        if not torch.isfinite(maps).all():
            raise RuntimeError('Asymmetric polar refiner input contains NaN/Inf.')

        mag_features = self.mag_stem(maps)
        phase_features = self.phase_stem(maps)
        for stage in self.paired_stages:
            mag_features, phase_features = self._run_stage(
                stage, mag_features, phase_features
            )

        bounded_delta = self.delta_limit * torch.tanh(
            self.delta_log_mag_head(mag_features)
        )
        raw_rotation = self.rotation_head(phase_features)
        identity_rotation = torch.zeros_like(raw_rotation)
        identity_rotation[:, :1] = 1.0
        raw_rotation_norm = torch.linalg.vector_norm(
            raw_rotation, dim=1, keepdim=True
        )
        rotation = torch.where(
            raw_rotation_norm > self.eps,
            raw_rotation / raw_rotation_norm.clamp_min(self.eps),
            identity_rotation,
        )

        mag_gate = torch.tanh(self.outer_mag_gate)
        phase_gate = torch.tanh(self.outer_phase_gate)
        applied_rotation = F.normalize(
            identity_rotation + phase_gate * (rotation - identity_rotation),
            dim=1,
            p=2,
            eps=self.eps,
        )
        base_real, base_imag = torch.chunk(base_complex, 2, dim=1)
        rot_real, rot_imag = torch.chunk(applied_rotation, 2, dim=1)
        phase_complex = torch.cat((
            base_real * rot_real - base_imag * rot_imag,
            base_real * rot_imag + base_imag * rot_real,
        ), dim=1)

        base_mag = torch.linalg.vector_norm(base_complex, dim=1, keepdim=True)
        applied_delta = mag_gate * bounded_delta
        amplification = (base_mag + self.eps) * torch.expm1(applied_delta)
        attenuation = base_mag * torch.expm1(applied_delta)
        delta_mag = torch.where(applied_delta >= 0.0, amplification, attenuation)
        corrected_mag = base_mag + delta_mag
        nonzero_magnitude = base_mag > 0.0
        safe_base_mag = torch.where(
            nonzero_magnitude, base_mag, torch.ones_like(base_mag)
        )
        phase_unit = phase_complex / safe_base_mag
        phase_unit = torch.where(nonzero_magnitude, phase_unit, applied_rotation)
        polar_complex = phase_complex + (corrected_mag - base_mag) * phase_unit

        ri_residual = torch.tanh(self.ri_residual_head(phase_features))
        learned_gate = torch.sigmoid(self.ri_residual_gate(phase_features))
        noisy_mag = torch.linalg.vector_norm(noisy_complex, dim=1, keepdim=True)
        energy_gate = noisy_mag / noisy_mag.amax(
            dim=(2, 3), keepdim=True
        ).clamp_min(self.phase_eps)
        energy_gate = energy_gate.clamp(0.0, 1.0)
        residual_gate = learned_gate * energy_gate
        applied_ri_residual = (
            self.complex_residual_scale * noisy_mag * residual_gate * ri_residual
        )
        refined_complex = polar_complex + applied_ri_residual

        if not torch.isfinite(refined_complex).all():
            raise RuntimeError('Asymmetric polar refiner output contains NaN/Inf.')

        aux = {
            'base_complex': base_complex,
            'delta_log_mag': bounded_delta,
            'applied_delta_log_mag': applied_delta,
            'applied_delta_magnitude': delta_mag,
            'corrected_magnitude': corrected_mag,
            'rotation': rotation,
            'applied_rotation': applied_rotation,
            'outer_mag_gate': mag_gate,
            'outer_phase_gate': phase_gate,
            'ri_residual': ri_residual,
            'ri_residual_learned_gate': learned_gate,
            'ri_residual_energy_gate': energy_gate,
            'ri_residual_gate': residual_gate,
            'ri_residual_applied': applied_ri_residual,
        }
        return refined_complex, aux
