# PGRT-MPDM v1

## Scope

PGRT-MPDM tests one narrow hypothesis: phase gradients can define a more useful
coordinate system for magnitude/phase interaction inside the existing
task-specialized asymmetric MPDM towers.

The experiment starts from `f5a379b`. It does not reuse Scheme 3,
Residual-Dense, TF-LCA, RDHI, MRCC, or their checkpoints.

## Data flow

1. Preserve the original MPDM towers, six middle VSS interactions, global
   fusion, magnitude mask, phase rotation, and output definitions.
2. Remove the deterministic `n_fft`/hop STFT carrier plane, then derive bounded
   time/frequency reassignment offsets from circular residual phase differences
   and a low-energy confidence field at the native STFT lattice.
3. Resize the analytic field to the bottleneck and let a shared phase-guided
   predictor estimate only bounded residual offsets and reliability.
4. Conservatively soft-splat magnitude and phase features to the reassigned
   coordinates. Each source cell distributes to at most four destination cells
   and its weights sum to one, including at boundaries.
5. Compute a lightweight shared cross-tower residual in the reassigned grid and
   apply the exact adjoint transport back to the native bottleneck grid.
6. Add the returned residual beside each original VSS result through bounded,
   zero-initialized per-stage scales.

PGRT does not claim to invent phase gradients, time-frequency reassignment,
deformable sampling, or soft splatting. Its testable contribution is their
bounded, confidence-backed use as an internal cross-tower transport in MPDM.

## Gate 0

All checks must pass before mini training:

- zero-offset transport is an exact identity;
- soft-splat source mass is conserved within `1e-4`;
- the implemented adjoint satisfies the inner-product identity;
- boundary handling and constant/linear phase fields are finite and bounded;
- feature, offset, and module gradients are finite;
- zero PGRT injection preserves the original MPDM output and input Jacobian;
- shared baseline parameters and downstream RNG remain paired when PGRT is
  enabled;
- added parameters are at most 120k;
- measured training forward/backward is at most 1.25x the original MPDM, and
  peak CUDA memory is at most 1.25x.

## Mini gate

- Fresh seed `1234`, 200 epochs, no checkpoint reuse.
- Screening reference: original mini PESQ `3.184735 @ 60k`.
- Inspect fixed-step curves; do not promote a branch from a single noisy peak.
- Continue to controlled P0-P4 ablations only if PGRT gains at least `0.02`
  PESQ by the fixed 100k screening point without material STOI degradation.

The mini recipe rebases the inherited manifests in memory onto the 4090
VoiceBank root. It does not rewrite the original JSON files. Gradient clipping
uses the already configured norm limit of `5.0` and aborts on non-finite norms.

The historical mini reference is only a screening comparator. Final thesis
evidence requires a paired baseline, independent validation, frozen test set,
multiple seeds, and mechanism-specific controls.
