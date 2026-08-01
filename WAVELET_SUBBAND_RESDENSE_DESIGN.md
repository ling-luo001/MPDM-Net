# Residual-Dense Wavelet Subband Experiment

## Purpose

This branch tests whether the promising wavelet interaction experiment is
limited by optimization and local capacity rather than by its core hypothesis.
It starts from `codex/exp-wavelet-subband-interaction` at commit `61748e8`.

The original wavelet branch remains unchanged and continues its own mini and
full-data runs. This branch is a separate mini experiment.

## Controlled changes

The encoder, Mamba blocks, tower widths, decoders, losses, optimizer, STFT, and
mini sample identities remain unchanged. Three related changes are introduced:

1. **Residual transition shortcuts.** Every tower downsample and upsample gains
   a parallel shape-matched shortcut. Its bounded scalar starts at zero, so the
   transition is exactly the original operation at initialization.
2. **Residual-dense directional adapters.** Each existing wavelet subband
   exchange is preserved. A three-layer dense directional adapter is appended,
   with each layer reading every previous local feature. Its magnitude and
   phase updates use independent bounded channel-wise zero-start scales.
3. **Coarse-to-fine dense context.** The deepest processed LL representation is
   supplied to all three detail adapters at the next synthesis level. The
   reconstructed result then conditions the following finer level. This gives
   time, frequency, and joint detail paths access to accumulated coarse speech
   structure without discarding any wavelet coefficient.

The phase adapter limit is `0.25`, half of the base wavelet phase limit, because
wrapped phase is less tolerant of aggressive residual updates.

## Initialization control

New shortcut and dense modules are constructed while saving and restoring the
CPU random-number-generator state. Consequently:

- all shared baseline parameters retain the same seeded initialization;
- later baseline modules see the same RNG sequence;
- all new externally applied paths are exactly zero at initialization;
- the optimized network initially computes the same function as the original
  wavelet network, up to floating-point reconstruction tolerance.

This isolates the effect of learning the new paths instead of conflating it
with a different random initialization.

## Deployment controls

- Server: RTX 4090 at `192.168.123.155`.
- Dataset: the same mini sample identities as the original 3090 wavelet run,
  with only the absolute server root rewritten.
- Epochs: 200.
- Batch size: 2 unless the measured concurrent memory profile makes it unsafe.
- Distributed port: 13447.
- Validation/checkpoint interval: 2,000 steps.

The experiment should be compared first against the original wavelet curve:
PESQ `2.446@2k`, `2.777@4k`, `2.910@10k`, `2.999@18k`, and `3.057@28k`.
It is retained only if the residual-dense curve improves convergence or later
quality without destabilizing phase loss.
