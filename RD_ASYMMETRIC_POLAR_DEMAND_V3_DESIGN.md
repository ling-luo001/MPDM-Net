# RD-Asymmetric-Polar-Demand-V3

## Scope

Demand-V3 keeps the Residual-Dense parent, asymmetric magnitude/phase backend,
four persistent stages, the Stage-3 compressed magnitude dense bridge, and the
serial polar-to-RI correction order from Demand-V2. It changes residual routing,
training schedules, optimizer grouping, and model averaging only.

Branch: `codex/exp-rd-asymmetric-polar-demand-v3`

Base: `7bc49fe`

## Balanced Residual Routing

Each projected interaction performs two scans but routes them only across
branches. The magnitude residual consumes the phase scan; the phase residual
consumes the magnitude scan. There is no additive self-scan term. Each target
uses a per-channel LayerScale initialized to `0.03` and bounded to an absolute
maximum of `0.25`. Channel values are exposed in the refiner diagnostics.

Stage 4 now preserves the Stage-1 skip explicitly:

```text
output = skip + channel_scale * fuse(upsampled, skip)
```

The channel scale starts at `0.10`. Setting it to zero is an exact identity.
No dense path was added; the existing compressed magnitude bridge is unchanged.

## Polar And RI Budgets

The serial polar then RI structure is unchanged. The RI correction has no global
learnable ratio. Its fixed configured maximum is `0.25`, multiplied per bin by
the RI demand and bounded raw RI prediction.

The magnitude reference first trusts parent outputs:

```text
trusted = max(base_magnitude, coarse_magnitude) + utterance_floor
reference = trusted + 0.25 * relu(noisy_magnitude - trusted)
```

Thus noisy magnitude can influence the correction budget without becoming the
reference wholesale.

## Training Policy

V3 computes anchor progress from fractional epoch and ramps alpha from `0` to
`1` over 24 epochs. This gives mini and full runs the same epoch semantics. The
legacy step schedule remains available to recipes that omit the epoch key.

Direct refiner supervision is full strength through epoch 10 and follows a
cosine transition through epoch 30. Afterwards magnitude remains at `0.30`,
phase at `1/3`, and base/polar/RI/demand multipliers are zero. Every multiplier
is logged to TensorBoard.

The parent LR scale is `1.0`; refiner body and final prediction projections are
both `0.75`; epoch LR decay is `0.98`. Multi-dimensional convolution weights,
including gate convolutions, receive weight decay. Biases, normalization, SSM
dynamics, scalar parameters, and channel LayerScale parameters do not.

EMA is enabled with configurable decay `0.999`. Generator checkpoints retain
the raw generator and add `generator_ema`. Validation runs the unchanged 824
member full set or unchanged 165 member mini subset for Raw and EMA separately.
Best-score reporting uses EMA when enabled. No test split is used by training.

## Verification Boundary

Windows Gate0 is structural because native selective-scan CUDA is unavailable:

```bash
python gate0_asymmetric_polar_demand_v3.py --allow-structural-only
```

Native acceptance still requires Linux CUDA:

```bash
python gate0_asymmetric_polar_demand_v3.py --require-native-cuda
```

The checked-in launch script is a direct control only. It was not executed by
this implementation task, and no training or checkpoint reuse was performed.
