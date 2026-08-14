# RD-ZipRefine-MP Gate 0 design

## Scope and parent

- Worktree: `G:\MPDM_Net\main_project\.worktrees\rd-ziprefine-mp`
- Parent HEAD: `148a4fe9bdf15f100a3a5fb85159730eb650e7f4`
- Scope: local Gate 0 only. No training, weight loading, checkpoint access, commit,
  push, checkout, or worktree operation is part of this implementation.
- The existing Residual-Dense/Scheme3 Stage 1 and Stage 2 paths remain intact.
  RD-ZipRefine-MP is optional post-processing of their final complex spectrum S0.

## Mechanism

`ZipRefineMP` receives `X` and `S0` in `[B,2,T,F]` real/imag format and forms
exactly eight maps:

1. `X.real`, `X.imag`
2. `S0.real`, `S0.imag`
3. `(X-S0).real`, `(X-S0).imag`
4. `log1p(|X|)`, `log1p(|S0|)`

Separate 8-to-112 stems feed asymmetric magnitude and phase branches. Each has
four stages with compression ratios `[1,2,2,1]`. Magnitude stages run frequency
then time Mamba; phase stages run time then frequency Mamba. Ratio-2 stages use
a learned stride-2 adjacent TF convolution, axis modeling at the compressed
resolution, and learned transposed-convolution restoration with an explicit
output size. A separate same-resolution residual path is retained, and restored
shape mismatch is a hard error.

After every stage, two separate sigmoid gates exchange phase-to-magnitude and
magnitude-to-phase features. Stage residual scales initialize to 0.10 and
interaction scales to 0.05, represented through `tanh`-bounded parameters.

The magnitude head predicts
`bounded_delta = tanh(head(mag_features))`. Positive corrections use the
`|S0| + eps` floor so an empty bin can be restored. Negative corrections scale
only the existing magnitude, so attenuation cannot cross through zero and turn
into a signed magnitude. Both paths use `expm1`, are exactly additive-zero at
the zero outer gate, and keep the exponent bounded to avoid overflow/underflow.

The phase head predicts two channels normalized to a unit complex rotation. Its
outer gate interpolates from the exact identity rotation and renormalizes before
complex multiplication with S0.

Both outer gates are scalar parameters initialized to exactly zero. Therefore
the initial module output and input VJP equal S0 elementwise. The gate derivatives
remain finite and nonzero at zero. Internal scales are deliberately small but
nonzero, so setting the outer gates to a small nonzero value immediately exposes
finite nonzero gradients to internal parameters.

## Integration and reproducibility

`model_cfg.zip_refine_mp_enabled` controls construction and defaults to false
when absent. The default recipe and `train.py` are unchanged. The optional module
is constructed after all baseline generator state, while the CPU RNG state is
saved and restored around construction. With the same seed, all shared state
tensors and the post-construction CPU RNG state therefore match the baseline.

When enabled, `latest_aux` includes:

- `base_complex`
- `delta_log_mag`, `applied_delta_log_mag`, and `applied_delta_magnitude`
- `rotation` and `applied_rotation`
- `outer_mag_gate` and `outer_phase_gate`
- `stage_scales` with shape `[2,4]` (magnitude, phase)
- `interaction_scales` with shape `[4,2]` (to magnitude, to phase)
- `corrected_magnitude`

## Configuration and parameter budget

The dedicated recipe is
`recipes/RD-ZipRefine-MP/RD-ZipRefine-MP.yaml`. It uses experiment/log name
`rd_ziprefine_mp_mini_v1` and dedicated distributed port `29517`. It contains no
resume, pretrained, or weight path. A future run must explicitly pass both the
recipe, independent experiment name, and `--mini`; Gate 0 does not run this
command and the mini gate remains mandatory before any full-data run:

```powershell
python -B train.py --config recipes/RD-ZipRefine-MP/RD-ZipRefine-MP.yaml --exp_name rd_ziprefine_mp_mini_v1 --mini
```

The local stub structural parameter counts are:

- parent generator: 1,961,130
- RD-ZipRefine-MP addition: 7,492,037
- candidate total: 9,453,167

The known real parent construction is approximately 1,963,626 parameters, which
differs from the stub baseline by 2,496. The unavailable native dependency stack
prevents isolating that discrepancy locally, so this document does **not** claim
that the Deform/Mamba dependency stubs preserve the aggregate real count exactly.
Adding the structural refiner count to the known real parent gives a provisional
candidate projection of 9,455,663, still inside 8-10M. The exact native CUDA
candidate count must be remeasured before the mini run.

The width was set to 112 from the structural count. A 128-channel first pass measured
11,612,991 structural parameters and was rejected as outside the approved 8-10M
gate.

## Gate 0 command and limits

Run with bytecode generation disabled:

```powershell
& 'E:\anaconda3\envs\Mamba1\python.exe' -B gate0_rd_ziprefine_mp.py
git diff --check
```

The local environment lacks `selective_scan_cuda`, `triton`, `transformers`, and
a usable torchvision runtime combination. The Gate 0 script therefore uses a
clearly labeled CPU structural stub for unavailable runtime operations. It still
constructs and counts the Mamba parameter tensors represented by this local
dependency-compatible structure, and the stub depends on those tensors so
gradient connectivity is tested. Deformable convolution is replaced only during
this local test by an ordinary convolution. This validates tensor contracts,
module-level and paired-generator exact identity/VJP behavior, gradient
connectivity, RNG isolation, structural parameter-range screening, and a small
generator forward/backward. It is not real selective-scan, deformable-convolution, CUDA,
performance, memory, convergence, or audio-quality validation.
