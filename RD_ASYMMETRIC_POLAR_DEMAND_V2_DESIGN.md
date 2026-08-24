# RD-Asymmetric-Polar-Demand-V2

## Objective

This branch keeps the thesis-level macro structure:

```text
Residual-Dense suppression/restoration parent
    -> S0 and restoration evidence
    -> asymmetric magnitude/phase backend
    -> demand-routed polar correction
    -> serial complex residual correction
```

It is a performance-first integrated candidate. The implementation deliberately
combines mutually supportive changes instead of trying to attribute the gain to
one variable during the first mini run.

Branch: `codex/exp-rd-asymmetric-polar-demand-v2`

Base: `e7631b8` (`codex/exp-rd-asymmetric-polar-anchor-dense`)

## Parent-to-Refiner Evidence

The refiner no longer receives only noisy `X` and final `S0`. The generator
passes nine evidence tensors:

1. coarse complex spectrum;
2. `S0 - coarse` structured-restoration residual;
3. harmonic prior;
4. expanded voicing map;
5. two restoration gates;
6. final suppression feature;
7. final restoration feature;
8. suppression bottleneck;
9. restoration bottleneck.

Together with `X` and `S0`, these form exactly 20 full-resolution evidence maps:
noisy/base/error/coarse/restoration RI maps, three log magnitudes, a bounded log
magnitude ratio, relative-phase cosine/sine, harmonic/voicing cues, and the two
restoration gates. Full and bottleneck parent latents are also injected through
small learnable residual projections.

## Persistent Asymmetric Backend

The old four stages independently executed `down -> model -> up` at Stage 2 and
Stage 3. V2 keeps one compressed state:

```text
Stage 1: full-resolution axis modeling
Stage 2: downsample once, then compressed modeling
Stage 3: fuse Stage-2 dense state before compressed Mamba modeling
Stage 4: upsample once, fuse Stage-1 skip, then full-resolution modeling
```

Magnitude remains frequency-first and phase remains time-first. Projected
bidirectional interactions remain at Stages 2-4, but their gates use small
nonzero initialization so cross-branch learning starts immediately. New
calibration paths use channel-only LayerNorm instead of normalizing an entire TF
map as one GroupNorm group.

## Demand-Routed Serial Correction

The backend predicts separate per-bin demand maps for magnitude, phase, and RI
correction.

Magnitude correction combines two paths:

```text
M_polar = clamp_min(
    M0 * exp(g_mag * delta_log_M)
    + g_mag * delta_add_M,
    0
)
```

The additive path is scaled by a robust reference based on the maximum of noisy,
coarse, and base magnitudes plus an utterance-relative floor. It can therefore
restore a bin even when `M0` is exactly zero.

Phase correction predicts a bounded scalar angle and applies a geodesic complex
rotation:

```text
U_polar = U_reference * exp(j * g_phase * delta_theta)
S_polar = M_polar * U_polar
```

When the base magnitude is too small, the phase unit falls back to the noisy
unit vector. This avoids the near-singular chord interpolation used previously.

The RI path runs after polar correction. It receives fused magnitude/phase
features plus noisy, base, polar, and noisy-minus-polar complex maps. Its scale
is learnable, initialized at `0.1`, and bounded by `0.5`. It uses the robust
reference magnitude directly and no longer suppresses weak bins by an
approximately quadratic noisy-energy gate.

## Gradient and Optimizer Strategy

The old hard anchor is generalized to a scheduled soft VJP. Forward values are
independent of alpha, while all parent evidence gradients are blended as:

```text
parent VJP = alpha * refiner VJP + (1 - alpha) * identity S0 VJP
```

The default schedule moves alpha from `0` to `1` over 20,000 steps. A direct
loss on the unanchored parent `S0` remains active, so the parent still receives a
stable clean target at alpha zero.

AdamW groups are separated into parent, refiner body, and correction heads.
SSM `A_log/D`, biases, normalization parameters, gates, and residual scales use
zero weight decay. The parent and refiner are clipped independently. Optimizer
resume preserves every group's learning-rate scale.

Direct refiner supervision covers parent `S0`, corrected magnitude, applied
phase delta, polar complex output, remaining RI residual, and three demand maps.
These losses prevent small output gates from starving the deep backend during
early optimization.

## Configuration and Data Controls

Recipe:

`recipes/RD-Asymmetric-Polar-Demand-V2/RD-Asymmetric-Polar-Demand-V2.yaml`

The full train/validation manifests, STFT, segment size, batch size, and PCS
setting remain unchanged. Full validation contains 824 paired utterances. Mini
validation remains the existing 165-item subset; no new test split is created.

The current structural count is:

```text
Residual-Dense parent: 1,961,130
V2 total:             3,535,632
Total / parent:       1.803x
```

FLOPs, real peak memory, and training speed remain pending native Linux CUDA
measurement.

## Verification and Later Deployment

Local structural Gate 0:

```bash
python gate0_asymmetric_polar_demand_v2.py --allow-structural-only
```

Required Linux CUDA Gate 0:

```bash
python gate0_asymmetric_polar_demand_v2.py --require-native-cuda
```

Mini launch after native Gate 0:

```bash
python train.py --mini \
  --exp_name rd_asymmetric_polar_demand_v2_mini_seed1234 \
  --max_steps 100000
```

Optional parent-only warm start is supported but must not be used without an
explicitly approved checkpoint:

```bash
python train.py --mini \
  --init_parent_checkpoint /path/to/g_XXXXXXXX.pth \
  --exp_name rd_asymmetric_polar_demand_v2_warm_mini_seed1234 \
  --max_steps 100000
```

The structural Gate uses stubs on the current Windows machine because native
`selective_scan_cuda` is unavailable. It verifies Python/model contracts and
autograd routing, not real selective-scan execution or training acceptance.
