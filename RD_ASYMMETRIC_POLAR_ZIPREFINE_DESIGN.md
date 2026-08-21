# RD-Asymmetric-Polar-ZipRefine Design

## Scope and invariant parent path

This experiment is an optional post-refiner on the Residual-Dense generator at
commit `ba669e7`. It does not alter the parent suppression/restoration modules,
the S0 definition, losses, data manifests, validation set, or training
hyperparameters. `models/zip_refine_mp.py` and its recipe remain unchanged.
The new and old refiner switches are mutually exclusive.

The refiner receives noisy spectrum `X` and parent output `S0`, each in real and
imaginary channels `[B, 2, T, F]`. Its input is exactly eight maps:

1. `X.RI`
2. `S0.RI`
3. `(X - S0).RI`
4. `log1p(|X|)`
5. `log1p(|S0|)`

## Asymmetric paired stages

The magnitude tower has 80 channels and applies frequency then time Mamba. The
phase tower has 40 channels and applies time then frequency Mamba. Both use
compression ratios `[1, 2, 2, 1]`. A deep-copied refiner config sets Mamba
`expand=2`; the parent config and all parent blocks retain `expand=4`.

Each paired stage owns both branch paths. Stage 1 has no interaction. Stages 2
and 3 downsample both paths, run axis modeling, interact in a 64-channel common
space, and only then upsample. Stage 4 interacts at full resolution in a
40-channel common space. Each interaction is a distinct module. It uses branch
projection, ordinary depthwise local enhancement, an aligned `SS2D_cross_new`
subclass, independent
ECA and normalization calibration, and explicit bidirectional gated exchange
before branch back-projection. The explicit exchange is required because
`SS2D_cross_new` shares scan parameters but does not itself transfer one branch's
features into the other branch. Cross gates start at `sigmoid(-2)` under zero
weights, and the interaction remains bounded by small learned residual scales.
The local subclass also inverses the reverse and transposed scan layouts before
the four directions are merged; the shared `models/cross.py` remains untouched.
It does not instantiate the legacy `Cross_layer` or
`VSSBlock_Cross_new` classes that include unsafe device assumptions elsewhere
in their dependency path.

Activation checkpointing is active only while training with gradients enabled.
The checkpoint boundary is the entire paired stage and its result is the
`(magnitude_features, phase_features)` tuple.

## Output contract

The main path predicts a bounded log-magnitude delta on `|S0|` and a normalized
unit complex rotation. Its magnitude and phase outer gates are scalar, exactly
zero initialized, so the polar path is exactly S0 at construction.

The final phase feature also drives the inherited A/B complex residual:

- a bounded RI residual whose final convolution is exactly zero initialized;
- a learned gate whose final weight is zero and bias is `-2`;
- noisy-magnitude energy normalization `|X| / max_TF(|X|)`;
- the fixed amplitude factor `0.1 * |X|`.

This residual is added after the polar correction. It has no extra zero outer
gate, so the zero-initialized final RI convolution can receive a gradient on the
first backward pass. The energy gate applies only to this RI residual and never
to the main magnitude delta. A refiner-local `eps=1e-6` protects polar rotation
and magnitude synthesis, while the inherited `phase_eps=1e-3` preserves the
historical A/B energy normalization and parent phase fallback behavior.

## Gate 0 boundary

`gate0_asymmetric_polar_ziprefine.py` performs no training, checkpoint loading,
or checkpoint reuse. It checks configuration isolation, disabled construction,
mutual exclusion, identity/VJP, head and outer-gate gradients, expand isolation,
odd/even shape restoration, interaction placement and widths, checkpoint
parity, finite forward/backward behavior, the `4,525,424` total parameter cap,
the stricter `2 x 1,961,130` same-build immediate-parent comparison, and state/RNG
isolation. If selective-scan or related local dependencies are
unavailable, the script labels and uses structural forward stubs; that result
is not CUDA validation and does not print Gate 0 success. Structural-only runs
must be explicitly acknowledged with `--allow-structural-only`.
`--require-native-cuda` disables fallback and is the required Linux acceptance
mode.
