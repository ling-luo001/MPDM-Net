# RD-Asymmetric-Polar-Anchor-Dense Design

## Scope

This candidate starts from `e448e1b` and changes only the optional asymmetric
polar post-refiner. Its recipe preserves the original data manifests,
training/loss settings, STFT, seed, and validation cadence. It neither loads
nor reuses a parent or refiner checkpoint.

## Mechanism

The one-way anchor is applied at the generator/refiner boundary. For parent
output `S0`, noisy spectrum `X`, and `A = stopgrad(S0)`:

`S_anchor = Refiner(X, A)`

`S = S_anchor + (S0 - A)`

The value of `S` is identical to a joint refiner call at the same weights. Its
VJP to `S0` is exactly identity, while gradients still reach the refiner
parameters through `S_anchor`. Disabling
`asymmetric_polar_zip_refine_oneway_anchor` preserves the `e448e1b` call path.

One magnitude-only dense bridge reuses the Stage-2 post-interaction compressed
state `z2` in Stage 3. After Stage-3 interaction and before upsampling:

`z3_out = z3 + tanh(alpha) * D(cat(z3, z2))`

`alpha` is a zero-initialized scalar. `D` is fixed to
`GN(160) -> Conv1x1(160,20) -> PReLU -> DWConv3x3(20) -> GN(20) -> PReLU ->
Conv1x1(20,80)` with bias-free convolutions, totaling 5,381 parameters including
`alpha`. No phase bridge or other cross-stage bridge is present. Bridge
construction saves and restores CPU RNG state so all shared tensors and the
caller's RNG stream match the legacy configuration at the same seed.

Activation checkpointing still wraps each complete paired stage. In the bridge
configuration, Stages 2 and 3 additionally return/consume the compressed
magnitude state; checkpoint-disabled and checkpoint-enabled paths share the
same stage implementation.

## Diagnostics

Auxiliary state and TensorBoard expose magnitude/phase stage scales,
interaction scales, dense-bridge scale, outer magnitude/phase gates, absolute
applied log-magnitude delta activity with P50/P90/P99, and applied RI residual
activity. Existing tags are unchanged.

## Risks And Paired Experiments

The anchor deliberately removes the refiner-to-parent gradient while preserving
the parent identity route; this may improve stability but can also prevent
useful joint adaptation. The dense bridge may become redundant with Stage-3
features or open too slowly from zero. The first screening experiment therefore
runs the approved combined route with the same manifests, seed policy, steps,
losses, and selection rule as its parent. Report PESQ/STOI/COVL, activation
scales, parameters, FLOPs, latency/RTF, and peak memory. If the combined route
is retained and mechanism attribution is later required for the thesis, the two
single-switch controls can be added without changing this implementation.

Local structural Gate 0 is not native selective-scan CUDA acceptance. The CUDA
run must independently verify real kernels, forward/backward parity, finite
gradients, and memory before training is approved.
