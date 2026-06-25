# Experiment B: Energy-Gated Complex Residual

Base branch: `main-codex-mask-0`

New branch: `exp-B-energy-gated-complex-residual`

## Motivation

The previous `main_codex_mask` branch adds a bounded complex residual after
magnitude masking and phase rotation. That residual is useful, but it is applied
uniformly across all time-frequency bins. Low-energy bins have unreliable phase
and are more likely to be noise-dominant, so unconstrained residual correction
can hurt stability.

## Code Changes

- Added a learnable residual gate in `models/generator.py`.
- Multiplied the learnable gate by an energy gate derived from normalized noisy
  magnitude.
- Kept the complex residual head zero-initialized.
- Initialized the gate output bias to `-2.0`, so the branch starts
  conservatively and learns stronger corrections only where useful.
- Stored `complex_residual` and `complex_residual_gate` in `generator.latest_aux`
  for later regularization and debugging.

## Config Changes

- `model_cfg.complex_residual_scale: 0.1`
- `model_cfg.complex_residual_gate_bias: -2.0`

## Expected Comparison

Compare this branch against `main_codex_mask`.

If this branch improves PESQ or reduces late-training oscillation, it supports
the claim that complex residual correction should be energy-aware rather than
globally applied.
