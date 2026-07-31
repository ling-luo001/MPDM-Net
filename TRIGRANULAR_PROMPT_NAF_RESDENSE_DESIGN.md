# Residual-Dense TriPrompt-NAF Experiment

## 1. Experiment Identity

- Base branch: `codex/exp-trigranular-prompt-naf`
- Base commit: `851981360621f1e10103a857e390d0c0edb826ef`
- Experiment branch: `codex/exp-trigranular-prompt-naf-resdense`
- Task: single-channel speech enhancement
- Training interface: unchanged
- Model input/output interface: unchanged

This is a combined structural optimization of TriPrompt-NAF. It tests whether
stable residual transport and compressed dense feature reuse can improve the
single-tower restoration model without changing its degradation-prompt idea,
loss weights, data, or complex-spectrum reconstruction rule.

## 2. Motivation

The original TriPrompt-NAF already has two residual updates inside each
`AxisNAFBlock`, but information between blocks and scales still travels through
strictly sequential transitions. Encoder detail can be weakened by strided
convolution, decoder skip fusion is a single projection, and the prompt is
estimated only from the deepest feature.

The optimization therefore targets four structural bottlenecks:

1. feature reuse inside each NAF stage;
2. stable information transport across resolution changes;
3. explicit residual preservation at decoder skip fusion;
4. multi-scale evidence for degradation prompts and final detail recovery.

## 3. Modified Data Flow

```text
complex/log-magnitude input
  -> intro
  -> residual-dense encoder stage 1
  -> residual downsample
  -> residual-dense encoder stage 2
  -> residual downsample
  -> gated multi-scale prompt context (enc1 + enc2 + bottleneck)
  -> tri-granular prompt estimator
  -> prompt-modulated residual-dense bottleneck
  -> residual upsample
  -> residual-dense skip fusion with encoder 2
  -> prompt-modulated residual-dense decoder stage 2
  -> residual upsample
  -> residual-dense skip fusion with encoder 1
  -> prompt-modulated residual-dense decoder stage 1
  -> prompt-modulated residual-dense refinement
  -> gated dense output bridge (refine + decoder + encoder + intro)
  -> bounded complex residual head
  -> noisy complex spectrum + predicted complex residual
```

## 4. New Modules

### 4.1 ResidualDenseNAFStage

Every stage retains the original ordered `AxisNAFBlock` path. Before block
`i > 0`, all earlier stage states are concatenated and compressed by an
activation-free gated 1x1 projector. The dense update is multiplied by a
learned `tanh` gate initialized to zero.

The complete stage is written as a long residual:

```text
output = anchor + bounded_gain * (sequential_dense_output - anchor)
```

The gain starts at 1.0 and is restricted to `[0.75, 1.25]` by default. Thus the
new stage exactly follows the original sequential path when dense gates are
zero, while preserving a short gradient route to the stage input.

### 4.2 ResidualDownsample and ResidualUpsample

The original strided convolution and pixel-shuffle paths remain the primary
paths. Each transition receives a zero-gated shortcut:

- downsample shortcut: average pooling followed by 1x1 channel projection;
- upsample shortcut: 1x1 channel projection followed by bilinear interpolation.

These paths preserve low-frequency structure and reduce reliance on a single
learned resampling operator.

### 4.3 ResidualDenseSkipFusion

The original concatenation plus 1x1 projection remains the base decoder fusion.
Two zero-gated additions are provided:

- a direct average of decoder and encoder features;
- a compressed dense update from decoder, encoder, and base-fusion features.

The base behavior is therefore available at initialization, while training can
learn how much identity detail and nonlinear dense context to restore.

### 4.4 MultiScalePromptContext

Encoder level 1 and level 2 are adaptively pooled and projected to bottleneck
width. They are densely fused with the original bottleneck through a zero-gated
residual update before prompt estimation. This lets temporal and spectral noise
profiles use both shallow local evidence and deep global evidence.

### 4.5 DenseOutputBridge

Before the complex residual head, a zero-gated dense bridge combines:

- refined output;
- refinement input;
- level-1 encoder feature;
- intro feature.

This is the final high-resolution detail route. It does not bypass the bounded
complex residual or directly alter waveform reconstruction.

## 5. Stability Strategy

All newly introduced cross-path residual gates start at zero. At initialization:

- transition outputs equal the original learned transition outputs;
- decoder skip outputs equal the original base fusion outputs;
- prompt context equals the original bottleneck;
- output bridge equals the original refined feature;
- dense stage inputs follow the original sequential block order.

Only the bounded stage gain starts active, at exactly 1.0. Existing zero-start
NAF residual scales and the zero-initialized complex residual head are retained.

## 6. Configuration

Two model fields are added:

- `dense_compression: 0.5`: hidden-width ratio in dense projectors;
- `stage_gain_limit: 0.25`: maximum deviation of stage residual gain from 1.0.

The following remain unchanged:

- base width and block counts;
- prompt count and prompt width;
- optimizer and learning-rate schedule;
- all enhancement and auxiliary loss weights;
- mini/full dataset selection;
- STFT and waveform reconstruction.

## 7. Diagnostics

Training reports these values every stdout interval:

- mean absolute dense-connection scale;
- mean stage residual gain;
- mean absolute transition residual scale;
- mean absolute decoder-skip residual scale;
- prompt-context residual scale;
- output-dense residual scale.

The same values are written to TensorBoard. A gate remaining near zero means the
corresponding path was not useful; fast saturation near one indicates a path
that may need tighter bounding in a follow-up experiment.

## 8. Evaluation Protocol

1. Verify syntax, variable-size forward shape, finite outputs, and backward pass.
2. Profile a full-resolution batch on the target GPU.
3. Start a fresh 200-epoch mini run with no checkpoint warm start.
4. Compare against the original TriPrompt-NAF at matched 2k validation steps.
5. Use the established mini reference near 60k as the main go/no-go checkpoint.
6. Run full data only if the mini curve is competitive and all gates remain
   numerically stable.

## 9. Rollback

The original TriPrompt-NAF remains on its own branch and remote training process.
Rollback is therefore branch-level: use `codex/exp-trigranular-prompt-naf` at
commit `8519813`. No checkpoint from this branch is compatible by strict state
dict loading because the structural parameter set is intentionally larger.

## 10. RTX 3090 Deployment Record

- Implementation commit: `6ad02a0`
- Host GPU: NVIDIA GeForce RTX 3090, 24 GB
- Python environment: `/home/lz/anaconda3/envs/mamba`
- PyTorch: `2.4.0+cu118`
- Remote repository: `/home/lz/PycharmProjects/MPDM-Net-trigranular-resdense`
- Mini experiment: `trigranular_prompt_naf_resdense_mini_v1`
- Epoch limit: 200
- Generator parameters: `3,480,618`

Full-resolution profile with `B=2`, `F=256`, and `T=256`:

- forward and backward elapsed time: `0.756 s`;
- peak allocated CUDA memory: `7.862 GiB`;
- peak reserved CUDA memory: `8.154 GiB`;
- output shape: `(2, 256, 256, 2)`.

All 1,743 mini training pairs and 165 mini validation pairs were present on the
target host. A two-step real-data smoke run completed the generator,
discriminator, all losses, backward pass, and optimizer updates.

The first formal validation result was:

- `PESQ 2.5487705714774855 @ 2k`;
- magnitude loss `0.047872947122563014`;
- phase loss `2.592063002875357`.

The original TriPrompt-NAF produced `PESQ 2.4257677287766426 @ 2k` on the same
mini sample lists. The residual-dense version therefore led by approximately
`0.1230 PESQ` at the first matched checkpoint. This is an early convergence
signal, not a final model ranking; later matched checkpoints remain required.
