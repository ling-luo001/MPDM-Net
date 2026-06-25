# Experiment E: EMA Validation and Best Checkpoint

Base branch: `exp-D-residual-regularized`

New branch: `exp-E-ema-best-checkpoint`

## Motivation

Previous full runs show that validation PESQ can peak in the middle of training
and then decline. Using the final checkpoint can under-report the method. This
experiment adds exponential moving average (EMA) tracking and automatic best
checkpoint saving.

## Code Changes

- Added EMA state tracking for the generator in `train.py`.
- EMA is updated after every generator optimizer step.
- Regular checkpoints still save:
  - `g_XXXXXXXX.pth`
  - `do_XXXXXXXX.pth`
- When EMA is enabled, regular checkpoint intervals also save:
  - `ema_g_XXXXXXXX.pth`
- Validation temporarily loads EMA generator weights, evaluates PESQ, then
  restores the raw training weights.
- Whenever validation PESQ improves, the branch saves:
  - `best_g.pth`

`best_g.pth` contains EMA weights when EMA is enabled, plus metadata for step,
epoch, PESQ, and EMA decay.

## Config Changes

- `training_cfg.use_ema: true`
- `training_cfg.ema_decay: 0.999`

## Expected Comparison

Compare this branch against Experiment D using the best validation checkpoint,
not the final checkpoint.

This branch is mainly a training-stability and reporting experiment. It should
be used to select the final model for full-test evaluation if EMA validation is
better than raw checkpoint validation.
