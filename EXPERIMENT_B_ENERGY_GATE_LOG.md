• 已经推送到云端了。现在 4 个实验分支远端都存在：

  origin/exp-B-energy-gated-complex-residual  dde5998
  origin/exp-C-mag-weighted-phase-loss        a07513a
  origin/exp-D-residual-regularized           c18452b
  origin/exp-E-ema-best-checkpoint            3d058bb

  刚才补推了 D 和 E：

  exp-D-residual-regularized -> origin/exp-D-residual-regularized
  exp-E-ema-best-checkpoint  -> origin/exp-E-ema-best-checkpoint

  B 和 C 在我检查时已经有远端 tracking 分支。当前本地分支停在 exp-C-mag-weighted-phase-loss，工作区仍只
  有未跟踪的实验输出/缓存，没有代码未提交。

─ Worked for 1m 02s ───────────────────────────────────────────────────────────────────────────────────
Token usage: total=314,306 input=297,638 (+ 2,759,552 cached) output=16,668 (reasoning 3,043)
To continue this session, run codex resume 019efe4c-f4c5-7f70-8bc8-81592fa03449


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
