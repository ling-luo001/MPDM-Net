"""Multi-resolution complex correction (MRCC) for MPDM-Net.

The module refines an already estimated native-resolution complex spectrum.  It
keeps all learned proposal processing in real/imaginary channels and uses
complex tensors only at the torch STFT/ISTFT API boundary.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DepthwiseResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.depthwise = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.norm1 = nn.GroupNorm(1, width)
        self.norm2 = nn.GroupNorm(1, width)

    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.silu(self.norm1(self.depthwise(x)))
        x = self.norm2(self.pointwise(x))
        x = x * (1.0 + scale) + shift
        return residual + F.silu(x)


class SharedComplexResidualProposer(nn.Module):
    """Shared convolutional proposer with learned resolution conditioning."""

    def __init__(self, width: int = 64, depth: int = 4, condition_dim: int = 240):
        super().__init__()
        if width < 8 or depth < 1 or condition_dim < 8:
            raise ValueError("MRCC proposer width/depth/condition_dim are too small")
        self.input_projection = nn.Conv2d(6, width, 3, padding=1)
        self.input_norm = nn.GroupNorm(1, width)
        self.blocks = nn.ModuleList([_DepthwiseResidualBlock(width) for _ in range(depth)])
        self.resolution_embedding = nn.Embedding(2, condition_dim)
        self.conditioner = nn.Sequential(
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, width * 2),
        )
        self.output = nn.Conv2d(width, 3, 1)

    def forward(self, features: torch.Tensor, resolution_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        condition_index = torch.tensor(resolution_index, device=features.device)
        scale, shift = self.conditioner(self.resolution_embedding(condition_index)).chunk(2)
        scale = 0.1 * torch.tanh(scale).view(1, -1, 1, 1)
        shift = 0.1 * torch.tanh(shift).view(1, -1, 1, 1)

        x = F.silu(self.input_norm(self.input_projection(features)))
        for block in self.blocks:
            x = block(x, scale, shift)
        output = self.output(x)
        return output[:, :2], output[:, 2:3]


class MRCCRefiner(nn.Module):
    """Post-estimator multi-resolution complex consensus correction."""

    ITERATIONS = 2

    def __init__(self, cfg: Dict):
        super().__init__()
        model_cfg = cfg["model_cfg"]
        stft_cfg = cfg["stft_cfg"]
        self.enabled = bool(model_cfg.get("mrcc_enabled", False))
        self.compress_factor = float(model_cfg.get("compress_factor", 0.3))
        self.native_resolution = (
            int(stft_cfg.get("n_fft", 510)),
            int(stft_cfg.get("win_size", 510)),
            int(stft_cfg.get("hop_size", 120)),
        )
        configured = model_cfg.get("mrcc_aux_resolutions", [[128, 128, 32], [256, 256, 64]])
        self.aux_resolutions = tuple(tuple(int(v) for v in resolution) for resolution in configured)
        if len(self.aux_resolutions) != 2 or any(len(resolution) != 3 for resolution in self.aux_resolutions):
            raise ValueError("MRCC v1 requires exactly two [n_fft, win, hop] auxiliary resolutions")

        self.correction_bound = float(model_cfg.get("mrcc_correction_bound", 0.1))
        self.reliability_floor = float(model_cfg.get("mrcc_reliability_floor", 0.05))
        self.reliability_ceiling = float(model_cfg.get("mrcc_reliability_ceiling", 0.95))
        self.consensus_damping = float(model_cfg.get("mrcc_consensus_damping", 0.05))
        self.consensus_step = float(model_cfg.get("mrcc_consensus_step", 0.5))
        self.eps = float(model_cfg.get("mrcc_eps", 1.0e-8))
        self.phase_eps = float(model_cfg.get("phase_eps", 1.0e-3))
        if not 0.0 < self.correction_bound <= 1.0:
            raise ValueError("mrcc_correction_bound must be in (0, 1]")
        if not 0.0 < self.reliability_floor < self.reliability_ceiling < 1.0:
            raise ValueError("MRCC reliability bounds must satisfy 0 < floor < ceiling < 1")
        if self.consensus_damping <= 0.0:
            raise ValueError("mrcc_consensus_damping must be positive")
        if not 0.0 < self.consensus_step <= 1.0:
            raise ValueError("mrcc_consensus_step must be in (0, 1]")
        if self.eps <= 0.0 or self.phase_eps <= 0.0:
            raise ValueError("MRCC epsilon values must be positive")

        self.proposer = SharedComplexResidualProposer(
            width=int(model_cfg.get("mrcc_proposer_width", 64)),
            depth=int(model_cfg.get("mrcc_proposer_depth", 4)),
            condition_dim=int(model_cfg.get("mrcc_condition_dim", 240)),
        )
        self.correction_gains = nn.Parameter(torch.zeros(2))
        self.last_diagnostics: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _assert_finite(name: str, value: torch.Tensor) -> None:
        if not torch.isfinite(value).all():
            raise RuntimeError(f"MRCC {name} contains NaN/Inf")

    @staticmethod
    def _window(win_length: int, reference: torch.Tensor) -> torch.Tensor:
        return torch.hann_window(win_length, device=reference.device, dtype=reference.dtype)

    def _stft_ri(self, waveform: torch.Tensor, resolution: Sequence[int]) -> torch.Tensor:
        n_fft, win_length, hop_length = resolution
        spectrum = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=self._window(win_length, waveform),
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
        return torch.stack((spectrum.real, spectrum.imag), dim=1)

    def _istft_ri(self, spectrum_ri: torch.Tensor, resolution: Sequence[int], length: int) -> torch.Tensor:
        n_fft, win_length, hop_length = resolution
        spectrum = torch.complex(spectrum_ri[:, 0], spectrum_ri[:, 1])
        return torch.istft(
            spectrum,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=self._window(win_length, spectrum_ri),
            center=True,
            normalized=False,
            length=length,
        )

    def _linear_ri(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        linear_magnitude = magnitude.clamp_min(0.0).pow(1.0 / self.compress_factor)
        return torch.stack((linear_magnitude * torch.cos(phase), linear_magnitude * torch.sin(phase)), dim=1)

    def _compressed_outputs(self, spectrum_ri: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real, imag = spectrum_ri[:, 0], spectrum_ri[:, 1]
        energy = real.square() + imag.square()
        linear_magnitude = torch.sqrt(energy.clamp_min(self.eps * self.eps))
        magnitude = linear_magnitude.pow(self.compress_factor)
        phase_real = torch.where(
            linear_magnitude.detach() < self.phase_eps,
            torch.full_like(real, self.phase_eps),
            real,
        )
        phase = torch.atan2(imag, phase_real)
        compressed_complex = torch.stack((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=-1)
        self._assert_finite("corrected magnitude", magnitude)
        self._assert_finite("corrected phase", phase)
        self._assert_finite("corrected compressed complex spectrum", compressed_complex)
        return magnitude, phase, compressed_complex

    def _propose(
        self,
        noisy_aux: torch.Tensor,
        baseline_aux: torch.Tensor,
        resolution_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = torch.cat((noisy_aux, baseline_aux, baseline_aux - noisy_aux), dim=1)
        raw_correction, reliability_logits = self.proposer(features, resolution_index)
        direction = torch.tanh(raw_correction)
        direction_norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
        direction = direction / direction_norm.clamp_min(1.0)
        noisy_magnitude = torch.linalg.vector_norm(noisy_aux, dim=1, keepdim=True)
        proposed = direction * (self.correction_bound * (noisy_magnitude + self.eps))
        gain = torch.tanh(self.correction_gains[resolution_index]).view(1, 1, 1, 1)
        correction = proposed * gain
        reliability = self.reliability_floor + (
            self.reliability_ceiling - self.reliability_floor
        ) * torch.sigmoid(reliability_logits)
        self._assert_finite("proposed correction", correction)
        self._assert_finite("reliability", reliability)
        proposed_ratio = torch.linalg.vector_norm(proposed, dim=1) / (
            self.correction_bound * (noisy_magnitude.squeeze(1) + self.eps)
        )
        return correction, reliability, proposed, proposed_ratio

    def _analyze_and_objective(
        self,
        waveform_correction: torch.Tensor,
        proposals: Sequence[torch.Tensor],
        reliabilities: Sequence[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        analyses: List[torch.Tensor] = []
        objective = self.consensus_damping * waveform_correction.square().mean(dim=-1)
        for proposal, reliability, resolution in zip(
            proposals, reliabilities, self.aux_resolutions
        ):
            analysis = self._stft_ri(waveform_correction, resolution)
            if analysis.shape != proposal.shape or reliability.shape != proposal[:, :1].shape:
                raise RuntimeError(
                    "MRCC proposal/reliability shape does not match its auxiliary STFT lattice"
                )
            residual_energy = (analysis - proposal).square().sum(dim=1)
            objective = objective + (reliability.squeeze(1) * residual_energy).mean(dim=(1, 2))
            analyses.append(analysis)
        return analyses, objective

    def _consensus(
        self,
        proposals: Sequence[torch.Tensor],
        reliabilities: Sequence[torch.Tensor],
        waveform_length: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        if len(proposals) != 2 or len(reliabilities) != 2:
            raise ValueError("MRCC v1 consensus requires exactly two proposal/reliability lattices")
        waveform_correction = proposals[0].new_zeros((proposals[0].shape[0], waveform_length))
        analyses, objective = self._analyze_and_objective(
            waveform_correction, proposals, reliabilities
        )
        objectives = [objective.mean()]
        acceptance_rates: List[torch.Tensor] = []

        for _ in range(self.ITERATIONS):
            update = -self.consensus_damping * waveform_correction
            preconditioner = waveform_correction.new_full(
                (waveform_correction.shape[0], 1), self.consensus_damping
            )
            for proposal, reliability, analysis, resolution in zip(
                proposals, reliabilities, analyses, self.aux_resolutions
            ):
                weighted_residual = reliability * (proposal - analysis)
                update = update + self._istft_ri(
                    weighted_residual, resolution, waveform_length
                )
                preconditioner = preconditioner + reliability.amax(
                    dim=(1, 2, 3), keepdim=False
                ).unsqueeze(1)

            trial = waveform_correction + self.consensus_step * update / preconditioner.clamp_min(
                self.eps
            )
            trial_analyses, trial_objective = self._analyze_and_objective(
                trial, proposals, reliabilities
            )
            # A detached, per-example accept/no-op decision keeps the selected
            # branch differentiable and guarantees the measured TF objective
            # cannot increase. There is no line search or data-dependent step growth.
            accept = (trial_objective <= objective).detach()
            waveform_correction = torch.where(
                accept.unsqueeze(1), trial, waveform_correction
            )
            analyses = [
                torch.where(accept.view(-1, 1, 1, 1), trial_value, current_value)
                for trial_value, current_value in zip(trial_analyses, analyses)
            ]
            objective = torch.where(accept, trial_objective, objective)
            objectives.append(objective.mean())
            acceptance_rates.append(accept.to(waveform_correction.dtype).mean())
        return waveform_correction, objectives, acceptance_rates

    def forward(
        self,
        noisy_magnitude: torch.Tensor,
        noisy_phase: torch.Tensor,
        baseline_magnitude: torch.Tensor,
        baseline_phase: torch.Tensor,
        baseline_complex: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.enabled:
            self.last_diagnostics = {}
            return baseline_magnitude, baseline_phase, baseline_complex

        for name, value in (
            ("noisy magnitude", noisy_magnitude),
            ("noisy phase", noisy_phase),
            ("baseline magnitude", baseline_magnitude),
            ("baseline phase", baseline_phase),
        ):
            self._assert_finite(name, value)

        noisy_native = self._linear_ri(noisy_magnitude, noisy_phase)
        baseline_native = self._linear_ri(baseline_magnitude, baseline_phase)
        waveform_length = self.native_resolution[2] * (noisy_magnitude.shape[-1] - 1)
        noisy_waveform = self._istft_ri(noisy_native, self.native_resolution, waveform_length)
        baseline_waveform = self._istft_ri(baseline_native, self.native_resolution, waveform_length)

        corrections: List[torch.Tensor] = []
        reliabilities: List[torch.Tensor] = []
        proposed_corrections: List[torch.Tensor] = []
        proposed_ratios: List[torch.Tensor] = []
        for resolution_index, resolution in enumerate(self.aux_resolutions):
            noisy_aux = self._stft_ri(noisy_waveform, resolution)
            baseline_aux = self._stft_ri(baseline_waveform, resolution)
            correction, reliability, proposed, proposed_ratio = self._propose(
                noisy_aux, baseline_aux, resolution_index
            )
            corrections.append(correction)
            reliabilities.append(reliability)
            proposed_corrections.append(proposed)
            proposed_ratios.append(proposed_ratio)

        waveform_correction, objectives, acceptance_rates = self._consensus(
            corrections, reliabilities, waveform_length
        )
        self._assert_finite("consensus waveform correction", waveform_correction)
        native_correction = self._stft_ri(waveform_correction, self.native_resolution)
        if native_correction.shape != baseline_native.shape:
            raise RuntimeError(
                f"MRCC native correction shape {tuple(native_correction.shape)} does not match "
                f"baseline {tuple(baseline_native.shape)}"
            )
        corrected_native = baseline_native + native_correction
        corrected_outputs = self._compressed_outputs(corrected_native)
        reference_outputs = self._compressed_outputs(baseline_native)
        baseline_outputs = (baseline_magnitude, baseline_phase, baseline_complex)
        # Residualizing in the repository output domain preserves both values
        # and the baseline Jacobian while the correction gains are zero.
        outputs = tuple(
            baseline + (corrected - reference)
            for baseline, corrected, reference in zip(
                baseline_outputs, corrected_outputs, reference_outputs
            )
        )

        correction_ratios = []
        for proposed in proposed_corrections:
            proposed_norm = torch.linalg.vector_norm(proposed, dim=1)
            correction_ratios.append(proposed_norm.amax())
        self.last_diagnostics = {
            "gain_128": torch.tanh(self.correction_gains[0]).detach(),
            "gain_256": torch.tanh(self.correction_gains[1]).detach(),
            "reliability_min": torch.stack([value.amin() for value in reliabilities]).amin().detach(),
            "reliability_mean": torch.stack([value.mean() for value in reliabilities]).mean().detach(),
            "reliability_max": torch.stack([value.amax() for value in reliabilities]).amax().detach(),
            "proposed_correction_max": torch.stack(correction_ratios).amax().detach(),
            "proposed_correction_ratio_max": torch.stack(
                [value.amax() for value in proposed_ratios]
            ).amax().detach(),
            "objective_initial": objectives[0].detach(),
            "objective_iter1": objectives[1].detach(),
            "objective_iter2": objectives[2].detach(),
            "consensus_accept_iter1": acceptance_rates[0].detach(),
            "consensus_accept_iter2": acceptance_rates[1].detach(),
        }
        return outputs
