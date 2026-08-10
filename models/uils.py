import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def diagonal_kalman_rts(
    observations,
    transition,
    process_noise,
    observation_noise,
    initial_covariance=1.0,
    gain_limit=1.0,
    force_fp32=True,
    return_diagnostics=True,
):
    """Run diagonal Kalman filtering and bounded RTS smoothing over time."""
    if observations.ndim != 4:
        raise ValueError('Expected observations with shape [B, D, T, F].')
    if observations.shape != transition.shape:
        raise ValueError('Kalman parameter shapes must match observations.')
    if process_noise.shape != observations.shape:
        raise ValueError('Process-noise shape must match observations.')
    if observation_noise.shape != observations.shape:
        raise ValueError('Observation-noise shape must match observations.')
    if observations.shape[2] < 1:
        raise ValueError('Time dimension must be non-empty.')
    if gain_limit <= 0.0:
        raise ValueError('gain_limit must be positive.')

    dtype = torch.float32 if force_fp32 else observations.dtype
    y = observations.to(dtype=dtype).permute(2, 0, 3, 1).contiguous()
    a = transition.to(dtype=dtype).permute(2, 0, 3, 1).contiguous()
    q = process_noise.to(dtype=dtype).permute(2, 0, 3, 1).contiguous()
    r = observation_noise.to(dtype=dtype).permute(2, 0, 3, 1).contiguous()

    state = torch.zeros_like(y[0])
    covariance = torch.full_like(y[0], float(initial_covariance))
    predicted_states = []
    predicted_covariances = []
    filtered_states = []
    filtered_covariances = []
    kalman_gains = [] if return_diagnostics else None

    for time_index in range(y.shape[0]):
        predicted_state = a[time_index] * state
        predicted_covariance = (
            a[time_index].square() * covariance + q[time_index]
        )
        innovation_covariance = predicted_covariance + r[time_index]
        kalman_gain = predicted_covariance / innovation_covariance.clamp_min(1e-8)
        state = predicted_state + kalman_gain * (y[time_index] - predicted_state)
        one_minus_gain = 1.0 - kalman_gain
        covariance = (
            one_minus_gain.square() * predicted_covariance
            + kalman_gain.square() * r[time_index]
        ).clamp_min(0.0)

        predicted_states.append(predicted_state)
        predicted_covariances.append(predicted_covariance)
        filtered_states.append(state)
        filtered_covariances.append(covariance)
        if return_diagnostics:
            kalman_gains.append(kalman_gain)

    smoothed_states = [None] * y.shape[0]
    rts_gains = (
        [torch.zeros_like(state) for _ in range(y.shape[0])]
        if return_diagnostics else None
    )
    smoothed_states[-1] = filtered_states[-1]
    for time_index in range(y.shape[0] - 2, -1, -1):
        gain = (
            filtered_covariances[time_index]
            * a[time_index + 1]
            / predicted_covariances[time_index + 1].clamp_min(1e-8)
        )
        gain = gain.clamp(min=-gain_limit, max=gain_limit)
        if return_diagnostics:
            rts_gains[time_index] = gain
        smoothed_states[time_index] = filtered_states[time_index] + gain * (
            smoothed_states[time_index + 1]
            - predicted_states[time_index + 1]
        )

    def restore_layout(values):
        return torch.stack(values, dim=0).permute(1, 3, 0, 2).contiguous()

    diagnostics = {}
    if return_diagnostics:
        diagnostics = {
            'filtered': restore_layout(filtered_states),
            'P': restore_layout(filtered_covariances),
            'K': restore_layout(kalman_gains),
            'G': restore_layout(rts_gains),
        }
    return restore_layout(smoothed_states), diagnostics


class UILS(nn.Module):
    """Uncertainty-informed latent smoothing at the dual-tower bottleneck."""

    def __init__(
        self,
        mag_channels=48,
        pha_channels=24,
        mag_state_dim=8,
        pha_state_dim=4,
        controller_hidden=24,
        a_limit=0.98,
        noise_floor=1e-4,
        noise_ceiling=10.0,
        rts_gain_limit=1.0,
    ):
        super().__init__()
        if (mag_channels, pha_channels) != (48, 24):
            raise ValueError('UILS v1 requires 48 magnitude and 24 phase channels.')
        if (mag_state_dim, pha_state_dim) != (8, 4):
            raise ValueError('UILS v1 requires state dimensions 8 and 4.')
        if not 0.0 < a_limit <= 0.98:
            raise ValueError('a_limit must be in (0, 0.98].')
        if not 0.0 < noise_floor < noise_ceiling <= 10.0:
            raise ValueError('Invalid process/observation noise bounds.')

        self.mag_channels = mag_channels
        self.pha_channels = pha_channels
        self.mag_state_dim = mag_state_dim
        self.pha_state_dim = pha_state_dim
        self.state_dim = mag_state_dim + pha_state_dim
        self.a_limit = float(a_limit)
        self.noise_floor = float(noise_floor)
        self.noise_ceiling = float(noise_ceiling)
        self.rts_gain_limit = float(rts_gain_limit)

        self.mag_to_state = nn.Conv2d(mag_channels, mag_state_dim, 1)
        self.pha_to_state = nn.Conv2d(pha_channels, pha_state_dim, 1)
        self.controller = nn.Sequential(
            nn.Conv2d(self.state_dim, controller_hidden, 1),
            nn.SiLU(),
            nn.Conv2d(controller_hidden, self.state_dim * 3, 1),
        )
        self.state_to_mag = nn.Conv2d(mag_state_dim, mag_channels, 1)
        self.state_to_pha = nn.Conv2d(pha_state_dim, pha_channels, 1)
        self.mag_gate = nn.Parameter(torch.zeros(()))
        self.pha_gate = nn.Parameter(torch.zeros(()))
        self.latest_diagnostics = {}
        self._initialize_controller_bias()

    def _initialize_controller_bias(self):
        final = self.controller[-1]
        initial_a = math.atanh(0.8 / self.a_limit)

        def inverse_softplus(value):
            return math.log(math.expm1(value - self.noise_floor))

        with torch.no_grad():
            final.bias[:self.state_dim].fill_(initial_a)
            final.bias[self.state_dim:2 * self.state_dim].fill_(
                inverse_softplus(0.05)
            )
            final.bias[2 * self.state_dim:].fill_(inverse_softplus(0.20))

    def _bounded_parameters(self, latent):
        raw_a, raw_q, raw_r = self.controller(latent).chunk(3, dim=1)
        transition = self.a_limit * torch.tanh(raw_a)
        process_noise = (F.softplus(raw_q) + self.noise_floor).clamp(
            max=self.noise_ceiling
        )
        observation_noise = (F.softplus(raw_r) + self.noise_floor).clamp(
            max=self.noise_ceiling
        )
        return transition, process_noise, observation_noise

    @staticmethod
    def _summary(name, value):
        detached = value.detach()
        return {
            f'{name}_finite': bool(torch.isfinite(detached).all().item()),
            f'{name}_min': float(detached.min().item()),
            f'{name}_max': float(detached.max().item()),
        }

    def forward(self, mag_features, pha_features, enabled=True, return_diagnostics=False):
        if mag_features.ndim != 4 or pha_features.ndim != 4:
            raise ValueError('UILS inputs must have shape [B, C, T, F].')
        if mag_features.shape[0] != pha_features.shape[0]:
            raise ValueError('UILS inputs must share batch size.')
        if mag_features.shape[2:] != pha_features.shape[2:]:
            raise ValueError('UILS inputs must share time-frequency shape.')
        if mag_features.shape[1] != self.mag_channels:
            raise ValueError('Unexpected magnitude channel count.')
        if pha_features.shape[1] != self.pha_channels:
            raise ValueError('Unexpected phase channel count.')
        if not enabled:
            result = (mag_features, pha_features)
            return (*result, {}) if return_diagnostics else result

        mag_latent = self.mag_to_state(mag_features)
        pha_latent = self.pha_to_state(pha_features)
        observations = torch.cat([mag_latent, pha_latent], dim=1)
        transition, process_noise, observation_noise = self._bounded_parameters(
            observations
        )
        smoothed, filter_diagnostics = diagonal_kalman_rts(
            observations,
            transition,
            process_noise,
            observation_noise,
            gain_limit=self.rts_gain_limit,
            force_fp32=True,
            return_diagnostics=return_diagnostics,
        )
        innovation = smoothed - observations.float()
        mag_innovation, pha_innovation = torch.split(
            innovation, [self.mag_state_dim, self.pha_state_dim], dim=1
        )
        mag_update = self.state_to_mag(mag_innovation.to(mag_features.dtype))
        pha_update = self.state_to_pha(pha_innovation.to(pha_features.dtype))
        mag_output = mag_features + (
            0.1 * torch.tanh(self.mag_gate) * torch.tanh(mag_update)
        )
        pha_output = pha_features + (
            0.1 * torch.tanh(self.pha_gate) * torch.tanh(pha_update)
        )

        if return_diagnostics:
            diagnostics = {
                'a': transition,
                'Q': process_noise,
                'R': observation_noise,
                **filter_diagnostics,
            }
            summary = {}
            for name in ('a', 'Q', 'R', 'P', 'K', 'G'):
                summary.update(self._summary(name, diagnostics[name]))
            self.latest_diagnostics = summary
            detached = {name: value.detach() for name, value in diagnostics.items()}
            return mag_output, pha_output, detached
        return mag_output, pha_output
