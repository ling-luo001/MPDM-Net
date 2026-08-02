# Multi-Scale Local-Channel Refinement Experiment

## Experiment Identity

- Base branch: `codex/exp-hierarchical-residual-dense`
- Base commit: `404c982985a29b382ee45bade8e483da6ae446ea`
- Parent experiment: `codex/exp-multiscale-local-channel` at `d7348b1`
- Stabilized experiment branch: `codex/exp-multiscale-local-channel-stabilized`
- Primary comparison: residual-dense mini under the same data, optimizer, and seed

## Motivation

The residual-dense model already provides long-range temporal/frequency Mamba
modeling, progressive suppression/restoration, deep filtering, harmonic priors,
and cross-stage feature reuse. It does not explicitly refine the output of each
Mamba stage with local two-dimensional time-frequency operators or channel
selection.

This experiment tests one composite hypothesis: Mamba stage outputs benefit
from a lightweight local-channel adapter that restores nearby spectral detail
and suppresses redundant channels before features move to the next scale.

The design is informed by three established observations:

- MambaDC adds depth-wise convolution after Mamba and reports consistent speech
  enhancement gains over vanilla Mamba.
- MambaIR combines local enhancement and channel attention to address local
  pixel forgetting and channel redundancy in restoration.
- SegNeXt shows that parallel depth-wise square and strip convolutions provide
  efficient multi-scale convolutional attention.

## Module

The stabilized `MultiScaleLocalChannelRefiner` performs:

1. Group normalization and a point-wise channel projection.
2. A depth-wise `3x3` branch, followed by zero-start reuse into the `7x1`
   temporal branch and both branches into the `1x7` frequency branch.
3. Learnable branch weights initialized to `1/sqrt(3)` per branch, exactly
   preserving the original variance-scaled aggregation at step zero.
4. SiLU and the existing point-wise output projection.
5. ECA-style channel attention expressed as `2 * sigmoid(.)`; its convolution
   starts at zero so the channel gain is exactly one.
6. A bounded residual update initialized to `0.05`, matching the original
   approximate effective update of `0.1 * 0.5` without forcing the outer scale
   to compensate for a half-open channel gate.

The three internal reuse scales start at zero. This keeps the initial local
branches equivalent to the original parallel TF-LCA while allowing the
temporal and frequency paths to learn residual reuse. Only six scalar
parameters are added per adapter: three dense scales and three branch logits.
Across 12 adapters this is 72 parameters, below the 128-parameter limit.

The strip branches have explicit speech roles: `7x1` captures short temporal
continuity and `1x7` captures local spectral/formant structure. The residual
path makes the module immediately trainable while limiting initial disruption.

## Placement

One adapter is applied after every stage-level Mamba stack in both towers:

- suppression encoder levels 1 and 2
- suppression bottleneck
- suppression decoder levels 2 and 1
- suppression refinement
- restoration encoder levels 1 and 2
- restoration bottleneck
- restoration decoder levels 2 and 1
- restoration refinement

No adapter is inserted inside individual Mamba blocks. This limits memory and
keeps the intervention interpretable as stage-level local refinement.

## Controlled Variables

The following remain unchanged from the residual-dense baseline:

- magnitude/phase input and complex-spectrum output interfaces
- suppression/restoration topology
- harmonic analysis and deep filtering
- residual-dense bridges and transition shortcuts
- all 12 TF-LCA insertion positions
- generator/discriminator losses
- mini train/validation lists
- batch size, learning rate, seed, and validation interval

## Screening Protocol

- Dataset: repository mini split
- Epoch budget: 200
- Batch size: 2
- Learning rate: `9e-4`
- Validation interval: 2,000 steps
- Initial decision points: 20k, 40k, and 60k steps

The branch is promising if it reaches or exceeds the known mini reference of
approximately `3.18 PESQ` near 60k steps, or shows a clearly stronger trajectory
than the residual-dense mini comparison without instability.

## Diagnostics

All TF-LCA diagnostics are detached before entering `generator.latest_aux`:

- `local_channel_scales`: effective outer residual scale for all 12 adapters
- `local_channel_dense_scales`: three internal reuse scales per adapter
- `local_channel_branch_weights`: `3x3`, temporal, and frequency weights
- `local_channel_channel_gain`: mean centered channel gain per adapter
- `local_channel_update_ratio`: RMS of the applied update divided by input RMS
- suppression/restoration means for the outer scale and update ratio

The same compact summaries are logged to stdout and TensorBoard. They separate
optimization instability from a model that learns to reject a branch or an
entire tower's local update.

## Stability Risks

- The original experiment's outer scale reached about `0.57`; that may reflect
  useful strong local correction rather than compensation for the old gate.
- Dense reuse can blur the intended temporal/frequency specialization.
- Branch weights are not regularized toward equality. If one branch collapses,
  PESQ and update-ratio evidence must decide whether that is useful selection
  or a failed optimization path.
- No extra cross-stage residual or dense bridge is added because the inherited
  `404c982` model already contains six residual-dense bridges and eight
  transition shortcuts.
