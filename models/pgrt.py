"""Phase-gradient residual transport building blocks for PGRT Gate 0."""

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorPair = Tuple[torch.Tensor, torch.Tensor]


def _wrapped_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the principal wrapped phase difference ``a - b``."""
    delta = a - b
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def _centered_wrapped_gradient(
    phase: torch.Tensor,
    dim: int,
    expected_increment: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Use one-sided boundary and averaged adjacent wrapped differences."""
    size = phase.shape[dim]
    if size == 1:
        return torch.zeros_like(phase)

    current = phase.narrow(dim, 1, size - 1)
    previous = phase.narrow(dim, 0, size - 1)
    if expected_increment is None:
        expected_increment = 0.0
    increments = _wrapped_difference(current, previous + expected_increment)
    first = increments.narrow(dim, 0, 1)
    last = increments.narrow(dim, size - 2, 1)
    if size == 2:
        return torch.cat((first, last), dim=dim)

    left = increments.narrow(dim, 0, size - 2)
    right = increments.narrow(dim, 1, size - 2)
    middle = left + 0.5 * _wrapped_difference(right, left)
    middle = torch.atan2(torch.sin(middle), torch.cos(middle))
    return torch.cat((first, middle, last), dim=dim)


def compute_analytic_phase_field(
    noisy_magnitude: torch.Tensor,
    noisy_phase: torch.Tensor,
    target_size: Sequence[int],
    max_offset: float = 1.0,
    eps: float = 1e-6,
    n_fft: Optional[int] = None,
    hop_size: Optional[int] = None,
    return_diagnostics: bool = False,
) -> Union[TensorPair, Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
    """Compute bounded phase-gradient offsets and energy reliability.

    Inputs use the model boundary layout ``[B, F, T]``. Outputs use feature-map
    layout ``[B, C, T', F']``. Offset channel 0 is time displacement and channel
    1 is frequency displacement. When ``n_fft`` and ``hop_size`` are supplied,
    the deterministic STFT carrier plane is removed before interpreting the
    residual spectral and temporal phase slopes. A residual phase change of pi
    maps to one maximum-offset cell.
    """
    if noisy_magnitude.ndim != 3 or noisy_phase.ndim != 3:
        raise ValueError("noisy magnitude and phase must have shape [B, F, T]")
    if noisy_magnitude.shape != noisy_phase.shape:
        raise ValueError("noisy magnitude and phase must have identical shapes")
    if len(target_size) != 2:
        raise ValueError("target_size must be (T, F)")
    if max_offset <= 0.0:
        raise ValueError("max_offset must be positive")
    if (n_fft is None) != (hop_size is None):
        raise ValueError("n_fft and hop_size must be supplied together")
    if n_fft is not None and (int(n_fft) <= 0 or int(hop_size) <= 0):
        raise ValueError("n_fft and hop_size must be positive")

    target_t, target_f = int(target_size[0]), int(target_size[1])
    if target_t <= 0 or target_f <= 0:
        raise ValueError("target dimensions must be positive")

    phase_tf = noisy_phase.transpose(1, 2)
    expected_t = None
    expected_f = None
    if n_fft is not None:
        carrier_step = 2.0 * math.pi * float(hop_size) / float(n_fft)
        frequency = torch.arange(
            phase_tf.shape[2], dtype=phase_tf.dtype, device=phase_tf.device
        ).view(1, 1, -1)
        frame = torch.arange(
            phase_tf.shape[1], dtype=phase_tf.dtype, device=phase_tf.device
        ).view(1, -1, 1)
        expected_t = torch.remainder(
            carrier_step * frequency + math.pi, 2.0 * math.pi
        ) - math.pi
        expected_f = torch.remainder(
            carrier_step * frame + math.pi, 2.0 * math.pi
        ) - math.pi

    temporal_residual = _centered_wrapped_gradient(
        phase_tf, dim=1, expected_increment=expected_t
    )
    spectral_residual = _centered_wrapped_gradient(
        phase_tf, dim=2, expected_increment=expected_f
    )
    offsets = torch.stack((-spectral_residual, temporal_residual), dim=1)
    offsets = (offsets / math.pi).clamp(-1.0, 1.0) * max_offset

    magnitude_tf = noisy_magnitude.transpose(1, 2)
    energy = magnitude_tf.square()
    reference_energy = energy.mean(dim=(1, 2), keepdim=True)
    confidence = energy / (energy + reference_energy + eps)
    confidence = confidence.unsqueeze(1).clamp(0.0, 1.0)

    if offsets.shape[-2:] != (target_t, target_f):
        offsets = F.interpolate(
            offsets,
            size=(target_t, target_f),
            mode="bilinear",
            align_corners=False,
        )
        confidence = F.interpolate(
            confidence,
            size=(target_t, target_f),
            mode="bilinear",
            align_corners=False,
        )

    offsets = offsets.clamp(-max_offset, max_offset)
    confidence = confidence.clamp(0.0, 1.0)
    if not return_diagnostics:
        return offsets, confidence

    diagnostics = {
        "temporal_phase_residual": temporal_residual,
        "spectral_phase_residual": spectral_residual,
        "offset_abs_max": offsets.detach().abs().amax(),
        "confidence_mean": confidence.detach().mean(),
    }
    return offsets, confidence, diagnostics


class AnalyticPhaseGradientField(nn.Module):
    """Module wrapper around :func:`compute_analytic_phase_field`."""

    def __init__(
        self,
        max_offset: float = 1.0,
        eps: float = 1e-6,
        n_fft: Optional[int] = None,
        hop_size: Optional[int] = None,
    ):
        super().__init__()
        self.max_offset = float(max_offset)
        self.eps = float(eps)
        self.n_fft = n_fft
        self.hop_size = hop_size

    def forward(
        self,
        noisy_magnitude: torch.Tensor,
        noisy_phase: torch.Tensor,
        target_size: Sequence[int],
        return_diagnostics: bool = False,
    ):
        return compute_analytic_phase_field(
            noisy_magnitude,
            noisy_phase,
            target_size,
            max_offset=self.max_offset,
            eps=self.eps,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            return_diagnostics=return_diagnostics,
        )


class ConservativeSoftSplat2D(nn.Module):
    """Conservative four-neighbor bilinear scatter with a paired adjoint.

    Features have shape ``[B, C, H, W]`` and offsets have shape
    ``[B, 2, H, W]`` in ``(dy, dx)`` order. Destination coordinates are
    clamped before the bilinear stencil is formed, so boundary mass is retained.
    """

    @staticmethod
    def _validate(features: torch.Tensor, offsets: torch.Tensor) -> None:
        if features.ndim != 4:
            raise ValueError("features must have shape [B, C, H, W]")
        expected = (features.shape[0], 2, features.shape[2], features.shape[3])
        if offsets.shape != expected:
            raise ValueError("offsets must have shape [B, 2, H, W]")
        if not offsets.is_floating_point():
            raise TypeError("offsets must be floating point")

    @staticmethod
    def _stencil(offsets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = offsets.shape
        dtype = offsets.dtype
        device = offsets.device
        base_y = torch.arange(height, dtype=dtype, device=device).view(1, height, 1)
        base_x = torch.arange(width, dtype=dtype, device=device).view(1, 1, width)
        target_y = (base_y + offsets[:, 0]).clamp(0.0, float(height - 1))
        target_x = (base_x + offsets[:, 1]).clamp(0.0, float(width - 1))

        y0 = torch.floor(target_y)
        x0 = torch.floor(target_x)
        y1 = (y0 + 1.0).clamp(max=float(height - 1))
        x1 = (x0 + 1.0).clamp(max=float(width - 1))
        wy = target_y - y0
        wx = target_x - x0

        weights = torch.stack(
            (
                (1.0 - wy) * (1.0 - wx),
                (1.0 - wy) * wx,
                wy * (1.0 - wx),
                wy * wx,
            ),
            dim=1,
        ).reshape(batch, 4, -1)
        indices = torch.stack(
            (
                y0 * width + x0,
                y0 * width + x1,
                y1 * width + x0,
                y1 * width + x1,
            ),
            dim=1,
        ).reshape(batch, 4, -1).long()
        return indices, weights

    def source_weights(self, offsets: torch.Tensor) -> torch.Tensor:
        """Return the four source-normalized bilinear weights."""
        if offsets.ndim != 4 or offsets.shape[1] != 2:
            raise ValueError("offsets must have shape [B, 2, H, W]")
        _, weights = self._stencil(offsets)
        return weights.reshape(offsets.shape[0], 4, offsets.shape[2], offsets.shape[3])

    def forward(self, features: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Apply conservative soft splatting."""
        self._validate(features, offsets)
        indices, weights = self._stencil(offsets)
        batch, channels, height, width = features.shape
        source = features.reshape(batch, channels, -1)
        output = torch.zeros_like(source)
        for neighbor in range(4):
            index = indices[:, neighbor].unsqueeze(1).expand(-1, channels, -1)
            contribution = source * weights[:, neighbor].unsqueeze(1)
            output.scatter_add_(2, index, contribution)
        return output.reshape(batch, channels, height, width)

    def adjoint(self, features: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Apply the exact transpose of :meth:`forward` for fixed offsets."""
        self._validate(features, offsets)
        indices, weights = self._stencil(offsets)
        batch, channels, height, width = features.shape
        target = features.reshape(batch, channels, -1)
        output = torch.zeros_like(target)
        for neighbor in range(4):
            index = indices[:, neighbor].unsqueeze(1).expand(-1, channels, -1)
            gathered = torch.gather(target, 2, index)
            output = output + gathered * weights[:, neighbor].unsqueeze(1)
        return output.reshape(batch, channels, height, width)

    gather = adjoint


class SharedPhaseFieldPredictor(nn.Module):
    """Shared lightweight predictor for phase-field residual and reliability."""

    def __init__(
        self,
        channels: int = 48,
        hidden: int = 24,
        max_offset: float = 1.0,
        residual_bound: float = 0.25,
    ):
        super().__init__()
        if channels <= 0 or hidden <= 0 or hidden > 24:
            raise ValueError("channels must be positive and hidden must be in [1, 24]")
        if max_offset <= 0.0 or residual_bound < 0.0:
            raise ValueError("offset bounds must be non-negative and max_offset positive")
        self.max_offset = float(max_offset)
        self.residual_bound = float(residual_bound)
        input_channels = 2 * channels + 3
        self.input_projection = nn.Conv2d(input_channels, hidden, kernel_size=1)
        self.norm = nn.GroupNorm(1, hidden)
        self.spatial = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=3,
            padding=1,
            groups=hidden,
        )
        self.field_head = nn.Conv2d(hidden, 3, kernel_size=1)

    def forward(
        self,
        mag_features: torch.Tensor,
        phase_features: torch.Tensor,
        analytic_offsets: torch.Tensor,
        analytic_confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mag_features.shape != phase_features.shape or mag_features.ndim != 4:
            raise ValueError("mag and phase features must share shape [B, C, H, W]")
        expected_offsets = (mag_features.shape[0], 2) + mag_features.shape[-2:]
        expected_confidence = (mag_features.shape[0], 1) + mag_features.shape[-2:]
        if analytic_offsets.shape != expected_offsets:
            raise ValueError("analytic_offsets do not match the feature grid")
        if analytic_confidence.shape != expected_confidence:
            raise ValueError("analytic_confidence does not match the feature grid")

        normalized_mag = F.group_norm(mag_features, 1)
        normalized_phase = F.group_norm(phase_features, 1)
        inputs = torch.cat(
            (normalized_mag, normalized_phase, analytic_offsets, analytic_confidence),
            dim=1,
        )
        hidden = F.silu(self.norm(self.input_projection(inputs)))
        hidden = F.silu(hidden + self.spatial(hidden))
        prediction = self.field_head(hidden)

        residual = self.residual_bound * torch.tanh(prediction[:, :2])
        offsets = analytic_offsets + residual
        offsets = offsets.clamp(-self.max_offset, self.max_offset)

        base_confidence = analytic_confidence.clamp(0.0, 1.0)
        safe_confidence = base_confidence.clamp(1e-6, 1.0 - 1e-6)
        confidence = torch.sigmoid(torch.logit(safe_confidence) + prediction[:, 2:3])
        confidence = torch.where(base_confidence <= 0.0, base_confidence, confidence)
        confidence = torch.where(base_confidence >= 1.0, base_confidence, confidence)
        return offsets, confidence


class CrossTowerInteractionRefiner(nn.Module):
    """Lightweight shared refiner operating on transported tower features."""

    def __init__(self, channels: int = 48, hidden: int = 24):
        super().__init__()
        if channels <= 0 or hidden <= 0 or hidden > 24:
            raise ValueError("channels must be positive and hidden must be in [1, 24]")
        self.input_projection = nn.Conv2d(2 * channels + 1, hidden, kernel_size=1)
        self.norm = nn.GroupNorm(1, hidden)
        self.spatial = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=3,
            padding=1,
            groups=hidden,
        )
        self.channel_mix = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.mag_output = nn.Conv2d(hidden, channels, kernel_size=1)
        self.phase_output = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(
        self,
        transported_mag: torch.Tensor,
        transported_phase: torch.Tensor,
        reliability: torch.Tensor,
    ) -> TensorPair:
        inputs = torch.cat((transported_mag, transported_phase, reliability), dim=1)
        hidden = F.silu(self.norm(self.input_projection(inputs)))
        hidden = F.silu(hidden + self.spatial(hidden))
        hidden = F.silu(self.channel_mix(hidden))
        mag_residual = self.mag_output(hidden) * reliability
        phase_residual = self.phase_output(hidden) * reliability
        return mag_residual, phase_residual


class PGRTInteraction(nn.Module):
    """Inject a bounded transported interaction residual into base fusion outputs."""

    def __init__(
        self,
        channels: int = 48,
        num_stages: int = 6,
        hidden: int = 24,
        max_offset: float = 1.0,
        offset_residual_bound: float = 0.25,
        injection_bound: float = 0.25,
        eps: float = 1e-6,
        n_fft: Optional[int] = None,
        hop_size: Optional[int] = None,
    ):
        super().__init__()
        if num_stages <= 0:
            raise ValueError("num_stages must be positive")
        self.channels = int(channels)
        self.num_stages = int(num_stages)
        self.eps = float(eps)
        if injection_bound <= 0.0:
            raise ValueError("injection_bound must be positive")
        self.injection_bound = float(injection_bound)
        self.analytic_field = AnalyticPhaseGradientField(
            max_offset=max_offset,
            eps=eps,
            n_fft=n_fft,
            hop_size=hop_size,
        )
        self.field_predictor = SharedPhaseFieldPredictor(
            channels,
            hidden,
            max_offset,
            residual_bound=offset_residual_bound,
        )
        self.transport = ConservativeSoftSplat2D()
        self.refiner = CrossTowerInteractionRefiner(channels, hidden)
        self.stage_branch_scales = nn.Parameter(torch.zeros(num_stages, 2))
        self.last_diagnostics: Dict[str, torch.Tensor] = {}

    def reset_diagnostics(self) -> None:
        self.last_diagnostics = {}

    def forward(
        self,
        mag_features: torch.Tensor,
        phase_features: torch.Tensor,
        analytic_offsets: torch.Tensor,
        analytic_confidence: torch.Tensor,
        stage_index: int,
        base_outputs: TensorPair,
        return_diagnostics: bool = False,
    ):
        if not isinstance(stage_index, int) or not 0 <= stage_index < self.num_stages:
            raise IndexError("stage_index is outside the configured stage range")
        if mag_features.shape != phase_features.shape or mag_features.ndim != 4:
            raise ValueError("mag and phase features must share shape [B, C, H, W]")
        if mag_features.shape[1] != self.channels:
            raise ValueError("feature channels do not match the configured channels")
        if not isinstance(base_outputs, (tuple, list)) or len(base_outputs) != 2:
            raise ValueError("base_outputs must be a (mag, phase) pair")
        base_mag, base_phase = base_outputs
        if base_mag.shape != mag_features.shape or base_phase.shape != phase_features.shape:
            raise ValueError("base outputs must match the bottleneck feature shapes")

        offsets, confidence = self.field_predictor(
            mag_features,
            phase_features,
            analytic_offsets,
            analytic_confidence,
        )
        transported_mag = self.transport(mag_features, offsets)
        transported_phase = self.transport(phase_features, offsets)
        occupancy = self.transport(torch.ones_like(confidence), offsets)
        transported_confidence = self.transport(confidence, offsets)
        transported_confidence = (
            transported_confidence / occupancy.clamp_min(self.eps)
        ).clamp(0.0, 1.0)

        refined_mag, refined_phase = self.refiner(
            transported_mag,
            transported_phase,
            transported_confidence,
        )
        mag_residual = self.transport.adjoint(refined_mag, offsets)
        phase_residual = self.transport.adjoint(refined_phase, offsets)
        scales = self.injection_bound * torch.tanh(
            self.stage_branch_scales[stage_index]
        )
        output_mag = base_mag + scales[0] * mag_residual
        output_phase = base_phase + scales[1] * phase_residual

        prefix = "stage_{:02d}".format(stage_index)
        self.last_diagnostics.update({
            prefix + "/offset_abs_mean": offsets.detach().abs().mean(),
            prefix + "/confidence_mean": confidence.detach().mean(),
            prefix + "/mag_scale": scales[0].detach(),
            prefix + "/phase_scale": scales[1].detach(),
            prefix + "/mag_residual_rms": mag_residual.detach().square().mean().sqrt(),
            prefix + "/phase_residual_rms": phase_residual.detach().square().mean().sqrt(),
        })

        if not return_diagnostics:
            return output_mag, output_phase

        weights = self.transport.source_weights(offsets)
        diagnostics = {
            "offsets": offsets.detach(),
            "confidence": confidence.detach(),
            "transported_confidence": transported_confidence.detach(),
            "branch_scales": scales.detach(),
            "source_weight_error": (weights.sum(dim=1) - 1.0).detach().abs().amax(),
            "offset_abs_max": offsets.detach().abs().amax(),
            "mag_residual_rms": mag_residual.detach().square().mean().sqrt(),
            "phase_residual_rms": phase_residual.detach().square().mean().sqrt(),
        }
        return output_mag, output_phase, diagnostics
