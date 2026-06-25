# Experiment C: Magnitude-Weighted Phase Loss

Base branch: `exp-B-energy-gated-complex-residual`

New branch: `exp-C-mag-weighted-phase-loss`

## Motivation

Phase is unreliable in very low-energy time-frequency bins. Treating all bins
equally can force the model to chase phase targets that have little perceptual
meaning and weak speech structure. This experiment keeps the energy-gated
complex residual from Experiment B and changes only the phase loss.

## Code Changes

- Extended `models/loss.py::phase_losses` with an optional `mag_weight`
  argument.
- When `mag_weight` is omitted, phase loss is unchanged.
- When enabled, clean magnitude is normalized per sample and used to weight:
  - instantaneous phase loss,
  - group-delay loss,
  - instantaneous angular-frequency loss.
- Updated `train.py` so training and validation both use weighted phase loss
  when `training_cfg.phase_weighted_loss` is true.

## Config Changes

- `training_cfg.phase_weighted_loss: true`
- `training_cfg.phase_weight_power: 0.5`
- `training_cfg.phase_weight_floor: 0.1`

The floor avoids ignoring low-energy bins completely, while the power controls
how aggressively the loss focuses on high-energy regions.

## Expected Comparison

Compare this branch against Experiment B.

If this branch improves PESQ or phase-related validation behavior, it supports
the claim that phase supervision should be magnitude-aware rather than uniform
over all time-frequency bins.
