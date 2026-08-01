"""Invertible wavelet-domain interaction for magnitude and phase features."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def haar_dwt2(x):
    """Apply an orthonormal 2D Haar transform without discarding samples."""
    if x.ndim != 4:
        raise ValueError(f"Expected a 4D tensor, got shape {tuple(x.shape)}")

    height, width = x.shape[-2:]
    pad_height = height % 2
    pad_width = width % 2
    if pad_height or pad_width:
        x = F.pad(x, (0, pad_width, 0, pad_height), mode="replicate")

    even_time_even_freq = x[..., 0::2, 0::2]
    even_time_odd_freq = x[..., 0::2, 1::2]
    odd_time_even_freq = x[..., 1::2, 0::2]
    odd_time_odd_freq = x[..., 1::2, 1::2]

    low = (
        even_time_even_freq
        + even_time_odd_freq
        + odd_time_even_freq
        + odd_time_odd_freq
    ) / 2.0
    time_detail = (
        -even_time_even_freq
        - even_time_odd_freq
        + odd_time_even_freq
        + odd_time_odd_freq
    ) / 2.0
    frequency_detail = (
        -even_time_even_freq
        + even_time_odd_freq
        - odd_time_even_freq
        + odd_time_odd_freq
    ) / 2.0
    joint_detail = (
        even_time_even_freq
        - even_time_odd_freq
        - odd_time_even_freq
        + odd_time_odd_freq
    ) / 2.0

    return (low, time_detail, frequency_detail, joint_detail), (height, width)


def haar_idwt2(bands, output_size):
    """Invert :func:`haar_dwt2` and crop only transform-time padding."""
    low, time_detail, frequency_detail, joint_detail = bands
    even_time_even_freq = (
        low - time_detail - frequency_detail + joint_detail
    ) / 2.0
    even_time_odd_freq = (
        low - time_detail + frequency_detail - joint_detail
    ) / 2.0
    odd_time_even_freq = (
        low + time_detail - frequency_detail - joint_detail
    ) / 2.0
    odd_time_odd_freq = (
        low + time_detail + frequency_detail + joint_detail
    ) / 2.0

    top = torch.stack(
        (even_time_even_freq, even_time_odd_freq), dim=-1
    ).flatten(-2)
    bottom = torch.stack(
        (odd_time_even_freq, odd_time_odd_freq), dim=-1
    ).flatten(-2)
    reconstructed = torch.stack((top, bottom), dim=-2).flatten(-3, -2)
    height, width = output_size
    return reconstructed[..., :height, :width]


class DirectionalSubbandExchange(nn.Module):
    """Exchange aligned magnitude/phase features within one wavelet band."""

    def __init__(
        self,
        channels,
        kernel_size,
        magnitude_update_limit=1.0,
        phase_update_limit=0.5,
    ):
        super().__init__()
        if len(kernel_size) != 2 or any(size % 2 == 0 for size in kernel_size):
            raise ValueError("kernel_size must contain two positive odd values")

        padding = tuple(size // 2 for size in kernel_size)
        self.magnitude_update_limit = float(magnitude_update_limit)
        self.phase_update_limit = float(phase_update_limit)
        self.input_norm = nn.GroupNorm(1, channels * 2)
        self.input_projection = nn.Conv2d(channels * 2, channels * 2, 1)
        self.directional_mixer = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
        )
        self.output_projection = nn.Conv2d(channels, channels * 2, 1)
        self.output_norm = nn.GroupNorm(1, channels * 2)

        # ReZero-style bounded scales make the new cross update conservative.
        self.raw_magnitude_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.raw_phase_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def _scales(self):
        magnitude_scale = self.magnitude_update_limit * torch.tanh(
            self.raw_magnitude_scale
        )
        phase_scale = self.phase_update_limit * torch.tanh(self.raw_phase_scale)
        return magnitude_scale, phase_scale

    def forward(self, magnitude, phase):
        if magnitude.shape != phase.shape:
            raise ValueError(
                "Magnitude and phase features must be aligned, got "
                f"{tuple(magnitude.shape)} and {tuple(phase.shape)}"
            )

        fused = self.input_norm(torch.cat((magnitude, phase), dim=1))
        value, gate = self.input_projection(fused).chunk(2, dim=1)
        value = self.directional_mixer(value)
        value = F.gelu(value) * torch.sigmoid(gate)
        magnitude_update, phase_update = self.output_norm(
            self.output_projection(value)
        ).chunk(2, dim=1)
        magnitude_scale, phase_scale = self._scales()
        return (
            magnitude + magnitude_scale * magnitude_update,
            phase + phase_scale * phase_update,
        )

    def scale_summary(self):
        magnitude_scale, phase_scale = self._scales()
        return {
            "magnitude": magnitude_scale.detach().abs().mean().item(),
            "phase": phase_scale.detach().abs().mean().item(),
        }


class ResidualDenseDirectionalLayer(nn.Module):
    """Compress a dense feature history, then refine it directionally."""

    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        padding = tuple(size // 2 for size in kernel_size)
        self.input_norm = nn.GroupNorm(1, input_channels)
        self.compress = nn.Conv2d(input_channels, hidden_channels, 1, bias=False)
        self.directional = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_channels,
            bias=False,
        )
        self.output_norm = nn.GroupNorm(1, hidden_channels)

    def forward(self, dense_history):
        shortcut = self.compress(self.input_norm(dense_history))
        update = F.gelu(self.output_norm(self.directional(shortcut)))
        return shortcut + update


class ResidualDenseSubbandAdapter(nn.Module):
    """Add a bounded residual-dense update with optional coarse context."""

    def __init__(
        self,
        channels,
        kernel_size,
        depth=3,
        width_ratio=0.5,
        magnitude_update_limit=0.5,
        phase_update_limit=0.25,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least one")
        if not 0.0 < width_ratio <= 2.0:
            raise ValueError("width_ratio must be in (0, 2]")

        self.channels = int(channels)
        self.magnitude_update_limit = float(magnitude_update_limit)
        self.phase_update_limit = float(phase_update_limit)
        hidden_channels = max(4, int(round(channels * width_ratio)))
        self.input_norm = nn.GroupNorm(1, channels * 4)
        self.stem = nn.Conv2d(channels * 4, hidden_channels, 1, bias=False)
        self.layers = nn.ModuleList(
            [
                ResidualDenseDirectionalLayer(
                    input_channels=hidden_channels * (index + 1),
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                )
                for index in range(depth)
            ]
        )
        dense_channels = hidden_channels * (depth + 1)
        self.output_norm = nn.GroupNorm(1, dense_channels)
        self.output_projection = nn.Conv2d(
            dense_channels, channels * 2, 1, bias=False
        )
        self.raw_magnitude_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.raw_phase_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def _scales(self):
        magnitude_scale = self.magnitude_update_limit * torch.tanh(
            self.raw_magnitude_scale
        )
        phase_scale = self.phase_update_limit * torch.tanh(self.raw_phase_scale)
        return magnitude_scale, phase_scale

    def forward(
        self,
        magnitude,
        phase,
        context_magnitude=None,
        context_phase=None,
    ):
        if magnitude.shape != phase.shape:
            raise ValueError("Magnitude and phase adapter inputs must be aligned")
        if context_magnitude is None and context_phase is None:
            context_magnitude = torch.zeros_like(magnitude)
            context_phase = torch.zeros_like(phase)
        elif context_magnitude is None or context_phase is None:
            raise ValueError("Both coarse context tensors must be provided together")
        if context_magnitude.shape != magnitude.shape or context_phase.shape != phase.shape:
            raise ValueError("Coarse context must match the current subband shape")

        adapter_input = self.input_norm(
            torch.cat(
                (magnitude, phase, context_magnitude, context_phase), dim=1
            )
        )
        features = [F.gelu(self.stem(adapter_input))]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        magnitude_update, phase_update = self.output_projection(
            self.output_norm(torch.cat(features, dim=1))
        ).chunk(2, dim=1)
        magnitude_scale, phase_scale = self._scales()
        return (
            magnitude + magnitude_scale * magnitude_update,
            phase + phase_scale * phase_update,
        )

    def scale_summary(self):
        magnitude_scale, phase_scale = self._scales()
        return {
            "dense_magnitude": magnitude_scale.detach().abs().mean().item(),
            "dense_phase": phase_scale.detach().abs().mean().item(),
        }


class ResidualDenseDirectionalExchange(nn.Module):
    """Preserve the base exchange and append a zero-start dense adapter."""

    def __init__(
        self,
        channels,
        kernel_size,
        magnitude_update_limit=1.0,
        phase_update_limit=0.5,
        dense_depth=3,
        dense_width_ratio=0.5,
        dense_magnitude_update_limit=0.5,
        dense_phase_update_limit=0.25,
    ):
        super().__init__()
        self.base_exchange = DirectionalSubbandExchange(
            channels=channels,
            kernel_size=kernel_size,
            magnitude_update_limit=magnitude_update_limit,
            phase_update_limit=phase_update_limit,
        )

        # Do not perturb initialization of later baseline modules.
        rng_state = torch.random.get_rng_state()
        self.dense_adapter = ResidualDenseSubbandAdapter(
            channels=channels,
            kernel_size=kernel_size,
            depth=dense_depth,
            width_ratio=dense_width_ratio,
            magnitude_update_limit=dense_magnitude_update_limit,
            phase_update_limit=dense_phase_update_limit,
        )
        torch.random.set_rng_state(rng_state)

    def forward(
        self,
        magnitude,
        phase,
        context_magnitude=None,
        context_phase=None,
    ):
        magnitude, phase = self.base_exchange(magnitude, phase)
        return self.dense_adapter(
            magnitude,
            phase,
            context_magnitude=context_magnitude,
            context_phase=context_phase,
        )

    def scale_summary(self):
        return {
            **self.base_exchange.scale_summary(),
            **self.dense_adapter.scale_summary(),
        }


class WaveletSubbandCrossFusion(nn.Module):
    """Multi-level, lossless subband-selective magnitude-phase fusion."""

    def __init__(
        self,
        channels,
        levels=1,
        magnitude_update_limit=1.0,
        phase_update_limit=0.5,
        dense_depth=3,
        dense_width_ratio=0.5,
        dense_magnitude_update_limit=0.5,
        dense_phase_update_limit=0.25,
    ):
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be at least one")
        self.levels = int(levels)

        def exchange(kernel_size):
            return ResidualDenseDirectionalExchange(
                channels=channels,
                kernel_size=kernel_size,
                magnitude_update_limit=magnitude_update_limit,
                phase_update_limit=phase_update_limit,
                dense_depth=dense_depth,
                dense_width_ratio=dense_width_ratio,
                dense_magnitude_update_limit=dense_magnitude_update_limit,
                dense_phase_update_limit=dense_phase_update_limit,
            )

        self.detail_exchanges = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "time": exchange((5, 1)),
                        "frequency": exchange((1, 5)),
                        "joint": exchange((3, 3)),
                    }
                )
                for _ in range(self.levels)
            ]
        )
        self.low_exchange = exchange((3, 3))

    def forward(self, magnitude, phase):
        if magnitude.shape != phase.shape:
            raise ValueError(
                "Magnitude and phase fusion inputs must have identical shapes"
            )

        details = []
        current_magnitude = magnitude
        current_phase = phase
        for exchanges in self.detail_exchanges:
            magnitude_bands, output_size = haar_dwt2(current_magnitude)
            phase_bands, phase_output_size = haar_dwt2(current_phase)
            if output_size != phase_output_size:
                raise RuntimeError("Wavelet branch sizes diverged unexpectedly")

            magnitude_low, magnitude_time, magnitude_frequency, magnitude_joint = (
                magnitude_bands
            )
            phase_low, phase_time, phase_frequency, phase_joint = phase_bands
            details.append(
                (
                    magnitude_time,
                    phase_time,
                    magnitude_frequency,
                    phase_frequency,
                    magnitude_joint,
                    phase_joint,
                    output_size,
                )
            )
            current_magnitude = magnitude_low
            current_phase = phase_low

        current_magnitude, current_phase = self.low_exchange(
            current_magnitude, current_phase
        )

        for level_index in reversed(range(len(details))):
            detail = details[level_index]
            exchanges = self.detail_exchanges[level_index]
            (
                magnitude_time,
                phase_time,
                magnitude_frequency,
                phase_frequency,
                magnitude_joint,
                phase_joint,
                output_size,
            ) = detail
            context_magnitude = current_magnitude
            context_phase = current_phase
            magnitude_time, phase_time = exchanges["time"](
                magnitude_time,
                phase_time,
                context_magnitude,
                context_phase,
            )
            magnitude_frequency, phase_frequency = exchanges["frequency"](
                magnitude_frequency,
                phase_frequency,
                context_magnitude,
                context_phase,
            )
            magnitude_joint, phase_joint = exchanges["joint"](
                magnitude_joint,
                phase_joint,
                context_magnitude,
                context_phase,
            )
            current_magnitude = haar_idwt2(
                (
                    current_magnitude,
                    magnitude_time,
                    magnitude_frequency,
                    magnitude_joint,
                ),
                output_size,
            )
            current_phase = haar_idwt2(
                (
                    current_phase,
                    phase_time,
                    phase_frequency,
                    phase_joint,
                ),
                output_size,
            )

        return current_magnitude, current_phase

    def scale_summary(self):
        summaries = [self.low_exchange.scale_summary()]
        for exchanges in self.detail_exchanges:
            summaries.extend(exchange.scale_summary() for exchange in exchanges.values())
        return {
            key: sum(summary[key] for summary in summaries) / len(summaries)
            for key in summaries[0]
        }
