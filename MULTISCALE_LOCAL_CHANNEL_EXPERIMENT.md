# Multi-Scale Local-Channel Refinement Experiment

## Experiment Identity

- Base branch: `codex/exp-hierarchical-residual-dense`
- Base commit: `404c982985a29b382ee45bade8e483da6ae446ea`
- Experiment branch: `codex/exp-multiscale-local-channel`
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

`MultiScaleLocalChannelRefiner` performs:

1. Group normalization and a point-wise channel projection.
2. Parallel depth-wise `3x3`, `7x1`, and `1x7` convolutions.
3. Variance-preserving branch aggregation and a point-wise output projection.
4. ECA-style channel attention without channel reduction.
5. A learnable residual connection initialized to an effective scale of `0.1`.

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

The mean absolute effective adapter scale is stored in
`generator.latest_aux['local_channel_scales']` and logged to stdout and
TensorBoard. This distinguishes a failed optimization path from a module that
is active but not useful.
