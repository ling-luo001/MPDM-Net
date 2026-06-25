# Change Summary
  最高测试 PESQ 是：

  PESQ = 3.465935945511
  step = 366000
  tag  = Validation/PESQ Score

• 原程序这个日志目录里的最高测试 PESQ 是：

  PESQ = 3.471664667130
  step = 584000
  tag  = Validation/PESQ Score


This branch keeps the original dual-tower input design. Both the magnitude tower
and phase tower still receive the raw `(noisy_mag, noisy_pha)` input pair.

The main change is at the generator output stage. After the existing magnitude
mask and phase-rotation path produce a base complex spectrum, the model predicts
a small complex residual from the final high-level phase feature `pha_fused`.
The residual is `tanh` bounded, scaled by the noisy magnitude, and added to the
base real/imaginary spectrum before recomputing enhanced magnitude, phase, and
complex output.

The residual head is zero-initialized, so training starts from behavior close to
the previous output path and learns the residual gradually.

To improve early training stability, low-energy bins use a small `phase_eps`
floor before `atan2`, avoiding singular gradients around `(real, imag) = (0, 0)`.
Training also clips generator and discriminator gradients with `max_grad_norm`
and enables non-finite gradient checks.

New config values:

- `training_cfg.max_grad_norm: 5.0`
- `model_cfg.phase_eps: 0.001`
