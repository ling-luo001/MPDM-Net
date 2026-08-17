"""Independent magnitude/phase complex-spectrum refinement for RD-ZipRefine-MP.

The module consumes the noisy complex spectrum X and the existing generator
output S0.  It never changes either upstream generator stage: it predicts a
bounded log-magnitude correction and a unit complex phase rotation which are
applied after S0 under exactly zero-initialized outer gates.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .mamba_block import FMambaBlock, TMambaBlock


def _scale_parameter(initial_value):
    if not -1.0 < initial_value < 1.0:
        raise ValueError('Residual scale initialization must be in (-1, 1).')
    return nn.Parameter(torch.tensor(math.atanh(float(initial_value))))


class _AxisRefinementStage(nn.Module):
    """One asymmetric axis-modeling stage with an optional learned TF zip path."""

    def __init__(self, cfg, channels, ratio, frequency_first):
        super().__init__()
        if ratio not in (1, 2):
            raise ValueError(f'Unsupported compression ratio: {ratio}')
        self.ratio = ratio
        first = FMambaBlock if frequency_first else TMambaBlock
        second = TMambaBlock if frequency_first else FMambaBlock
        self.axis_blocks = nn.ModuleList((first(cfg, channels), second(cfg, channels)))
        self.pre_norm = nn.GroupNorm(1, channels)
        self.same_resolution = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GELU(),
        ) if ratio == 2 else None
        if ratio == 2:
            self.down = nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(1, channels),
                nn.GELU(),
            )
            self.up = nn.ConvTranspose2d(
                channels, channels, 3, stride=2, padding=1, bias=False
            )
        else:
            self.down = None
            self.up = None
        self.residual_scale = _scale_parameter(0.10)

    def forward(self, x):
        residual = self.pre_norm(x)
        if self.ratio == 2:
            same_resolution = self.same_resolution(residual)
            residual = self.down(residual)
        else:
            same_resolution = None
        for block in self.axis_blocks:
            residual = block(residual)
        if self.ratio == 2:
            residual = self.up(residual, output_size=x.shape)
            if residual.shape != x.shape:
                raise RuntimeError(
                    f'Zip restoration shape mismatch: {tuple(residual.shape)} != {tuple(x.shape)}'
                )
            residual = residual + same_resolution
        return x + torch.tanh(self.residual_scale) * residual


class _BidirectionalGatedInteraction(nn.Module):
    """Gated phase-to-magnitude and magnitude-to-phase feature exchange."""

    def __init__(self, channels):
        super().__init__()
        self.phase_to_mag = nn.Conv2d(channels, channels, 1, bias=False)
        self.mag_to_phase = nn.Conv2d(channels, channels, 1, bias=False)
        self.mag_gate = nn.Conv2d(channels * 2, channels, 1)
        self.phase_gate = nn.Conv2d(channels * 2, channels, 1)
        self.mag_scale = _scale_parameter(0.05)
        self.phase_scale = _scale_parameter(0.05)

    def forward(self, mag, phase):
        joint = torch.cat((mag, phase), dim=1)
        mag_update = torch.sigmoid(self.mag_gate(joint)) * self.phase_to_mag(phase)
        phase_update = torch.sigmoid(self.phase_gate(joint)) * self.mag_to_phase(mag)
        return (
            mag + torch.tanh(self.mag_scale) * mag_update,
            phase + torch.tanh(self.phase_scale) * phase_update,
        )


class ZipRefineMP(nn.Module):
    """Four-stage asymmetric multi-resolution complex-spectrum refiner."""

    compression_ratios = (1, 2, 2, 1)

    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg['model_cfg']
        self.channels = int(model_cfg.get('zip_refine_mp_channels', 112))
        self.eps = float(model_cfg.get('zip_refine_mp_eps', 1e-6))
        self.delta_limit = float(model_cfg.get('zip_refine_mp_delta_limit', 1.0))
        self.activation_checkpointing = bool(
            model_cfg.get('zip_refine_mp_activation_checkpointing', False)
        )
        if self.channels <= 0:
            raise ValueError('zip_refine_mp_channels must be positive.')
        if self.eps <= 0.0:
            raise ValueError('zip_refine_mp_eps must be positive.')
        if not 0.0 < self.delta_limit <= 8.0:
            raise ValueError('zip_refine_mp_delta_limit must be in (0, 8].')

        # Both branches see the exact same eight-map evidence, but use separate
        # projections and opposite axis priorities.
        self.mag_stem = nn.Sequential(
            nn.Conv2d(8, self.channels, 3, padding=1),
            nn.GroupNorm(1, self.channels),
            nn.GELU(),
        )
        self.phase_stem = nn.Sequential(
            nn.Conv2d(8, self.channels, 3, padding=1),
            nn.GroupNorm(1, self.channels),
            nn.GELU(),
        )
        self.mag_stages = nn.ModuleList([
            _AxisRefinementStage(cfg, self.channels, ratio, frequency_first=True)
            for ratio in self.compression_ratios
        ])
        self.phase_stages = nn.ModuleList([
            _AxisRefinementStage(cfg, self.channels, ratio, frequency_first=False)
            for ratio in self.compression_ratios
        ])
        self.interactions = nn.ModuleList([
            _BidirectionalGatedInteraction(self.channels)
            for _ in self.compression_ratios
        ])
        self.delta_log_mag_head = nn.Conv2d(self.channels, 1, 1)
        self.rotation_head = nn.Conv2d(self.channels, 2, 1)

        # These are the only exactly-zero scales. Internal residual scales stay
        # small but nonzero so a tiny outer gate immediately exposes gradients.
        self.outer_mag_gate = nn.Parameter(torch.zeros(()))
        self.outer_phase_gate = nn.Parameter(torch.zeros(()))

    def _run_stage(self, stage, features):
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(stage, features, use_reentrant=False)
        return stage(features)

    def build_eight_map_input(self, noisy_complex, base_complex):
        for name, value in (('noisy_complex', noisy_complex), ('base_complex', base_complex)):
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
            raise RuntimeError('RD-ZipRefine-MP input contains NaN/Inf.')

        mag_features = self.mag_stem(maps)
        phase_features = self.phase_stem(maps)
        for mag_stage, phase_stage, interaction in zip(
                self.mag_stages, self.phase_stages, self.interactions):
            mag_features = self._run_stage(mag_stage, mag_features)
            phase_features = self._run_stage(phase_stage, phase_features)
            mag_features, phase_features = interaction(mag_features, phase_features)

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
        phase_real = base_real * rot_real - base_imag * rot_imag
        phase_imag = base_real * rot_imag + base_imag * rot_real

        base_mag = torch.linalg.vector_norm(base_complex, dim=1, keepdim=True)
        applied_delta = mag_gate * bounded_delta
        # Positive deltas can synthesize from the magnitude floor. Negative
        # deltas attenuate the existing magnitude and therefore cannot cross zero.
        # Both branches are exactly additive-zero at a zero outer gate.
        amplification = (base_mag + self.eps) * torch.expm1(applied_delta)
        attenuation = base_mag * torch.expm1(applied_delta)
        delta_mag = torch.where(applied_delta >= 0.0, amplification, attenuation)
        corrected_mag = base_mag + delta_mag
        phase_complex = torch.cat((phase_real, phase_imag), dim=1)
        nonzero_magnitude = base_mag > 0.0
        safe_base_mag = torch.where(
            nonzero_magnitude, base_mag, torch.ones_like(base_mag)
        )
        phase_unit = phase_complex / safe_base_mag
        phase_unit = torch.where(nonzero_magnitude, phase_unit, applied_rotation)
        magnitude_update = corrected_mag - base_mag
        refined_complex = phase_complex + magnitude_update * phase_unit

        if not torch.isfinite(refined_complex).all():
            raise RuntimeError('RD-ZipRefine-MP output contains NaN/Inf.')

        stage_scales = torch.stack((
            torch.stack([torch.tanh(stage.residual_scale) for stage in self.mag_stages]),
            torch.stack([torch.tanh(stage.residual_scale) for stage in self.phase_stages]),
        ))
        interaction_scales = torch.stack([
            torch.stack((torch.tanh(block.mag_scale), torch.tanh(block.phase_scale)))
            for block in self.interactions
        ])
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
            'stage_scales': stage_scales,
            'interaction_scales': interaction_scales,
        }
        return refined_complex, aux
