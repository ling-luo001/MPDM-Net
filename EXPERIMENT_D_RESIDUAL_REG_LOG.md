# Experiment D: Complex Residual Regularization

Base branch: `exp-C-mag-weighted-phase-loss`

New branch: `exp-D-residual-regularized`

## Motivation

Experiments B and C make the residual branch more structured, but the residual
can still become too dominant if training discovers that direct real/imaginary
correction is an easy shortcut. This experiment adds a small penalty on the
actual applied complex residual so it stays a fine correction term.

## Code Changes

- `models/generator.py` now stores `complex_residual_applied` in
  `generator.latest_aux`.
- `train.py` adds `complex_residual_regularization(generator)`.
- The regularizer penalizes the magnitude of the applied real/imaginary
  residual.
- Training stdout and TensorBoard now include `Res Loss` /
  `Training/Complex Residual Loss`.

## Config Changes

- `training_cfg.loss.complex_residual_reg: 0.01`

## Expected Comparison

Compare this branch against Experiment C.

If validation PESQ becomes more stable or late-training degradation is reduced,
it supports the claim that the residual branch should be constrained as a local
refinement path rather than a free reconstruction path.
