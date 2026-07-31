# Wavelet Subband Interaction Experiment

## Research question

Can MPDM-Net improve magnitude-phase coordination by replacing generic VSS
cross-fusion with lossless, direction-aware interaction in a multi-resolution
time-frequency basis?

This experiment does not claim magnitude-phase decoupling. Both towers still
receive noisy magnitude and phase, and they exchange information. The precise
claim being tested is **selective coupling**: interaction is constrained by the
time/frequency semantics of aligned wavelet subbands and the phase update is
more conservative than the magnitude update.

## Literature overlap audit

Wavelets themselves are not a new speech-enhancement idea:

- Classical systems use wavelet-domain thresholding, Wiener filtering, NMF, or
  RPCA directly on waveform subbands.
- WA-FSN (Electronics, 2025) replaces part of Adaptive FullSubNet's STFT input
  representation with DWT subband features.
- WTFormer (Interspeech, 2025) uses WTConv in a multi-channel MIMO enhancer to
  preserve spatial cues.
- WTConv (ECCV, 2024) provides an image backbone layer with a large receptive
  field whose parameter growth is logarithmic in receptive-field size.
- SWFormer (2025) combines spatial, wavelet, and Fourier mixing for image
  restoration.
- FFC-SE (Interspeech, 2022) already adapts Fourier global convolution to speech
  enhancement, so a plain Fourier branch would have substantial overlap.

Primary links:

- WA-FSN: https://doi.org/10.3390/electronics14071354
- WTFormer: https://www.isca-archive.org/interspeech_2025/han25c_interspeech.html
- WTConv: https://arxiv.org/abs/2407.05848
- SWFormer: https://arxiv.org/abs/2505.05504
- FFC-SE: https://www.isca-archive.org/interspeech_2022/shchekotov22_interspeech.html
- MP-SENet: https://arxiv.org/abs/2305.13686

The experiment's narrower mechanism was not found in this audit: an internal,
invertible 2D wavelet decomposition of aligned single-channel magnitude- and
phase-tower features, followed by subband-specific bidirectional exchange and
exact synthesis. This is an experimental differentiation, not a definitive
novelty claim; a publication would still require a formal systematic review.

## Frozen variables

The branch starts from commit `f5a379b`, the original dual-tower baseline.
These components remain unchanged:

- magnitude and lightweight phase tower widths;
- encoder, Mamba blocks, skip paths, and decoders;
- magnitude-mask and phase-rotation outputs;
- STFT settings, losses, optimizer, learning rate, and batch size;
- six bottleneck interaction positions and one final interaction position.

The independent variable is the cross-tower interaction operator. All seven
VSS fusion modules are replaced; wavelet fusion is not stacked on top of VSS.

## Proposed operator

For latent features shaped `[B, C, T, F]`, an orthonormal 2D Haar transform
produces four aligned bands at each level:

- `LL`: coarse speech envelope and broad spectral structure;
- `HL`: rapid temporal changes, such as onsets and transients;
- `LH`: local frequency changes, such as harmonic/formant edges;
- `HH`: joint fine detail, where weak speech detail and noise can coexist.

Each band has its own exchange block. Temporal detail uses a `5x1` depthwise
kernel, frequency detail uses `1x5`, joint detail and the deepest low band use
`3x3`. A normalized magnitude-phase concatenation produces a gated shared
feature and separate residual updates for the two towers.

The updates use bounded, channel-wise ReZero scales:

```
M_out = M + tanh(alpha_M) * Delta_M
P_out = P + 0.5 * tanh(alpha_P) * Delta_P
```

Both scales start at zero. Therefore the new operator is exactly identity at
initialization, while gradients can first learn how strongly each channel
should interact. The smaller phase bound limits unstable corrections to the
wrapped phase representation.

The bottleneck uses two wavelet levels; the full-resolution final fusion uses
one. Every coefficient is processed and passed to the inverse Haar transform.
Odd dimensions are padded only for the transform and cropped after exact
reconstruction. This is not lossy pooling or a compute-only frequency packing.

## Why this is orthogonal to the running groups

- It does not add harmonic/F0 generation or a suppression-restoration cascade.
- It does not add prompt tokens, NAF prompting, or tri-granular conditioning.
- It does not add residual-dense paths across the main tower hierarchy.
- It changes where and how the existing magnitude-phase interaction happens.

## Validation plan

1. Verify Haar reconstruction on even and odd shapes with error below `1e-6`.
2. Verify exact identity at zero update scales.
3. Run finite forward/backward checks and inspect all parameter gradients.
4. Profile parameters, peak memory, and step time beside the existing 3090 job.
5. Train the unchanged mini split for 200 epochs with validation every 2k steps.
6. Compare the complete curve against the original mini baseline, whose known
   reference is approximately PESQ 3.18 near 60k steps.

The branch earns a full-data run only if it closes the early gap and reaches or
exceeds the baseline curve without unstable phase loss. A single early PESQ
point is diagnostic evidence, not a final decision.
