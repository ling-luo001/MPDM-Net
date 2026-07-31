# Tri-Granular Degradation-Prompt NAF Speech Enhancement

## 1. Experiment role

This branch is an independent architectural hedge against the progressive
suppression/restoration family. It does not inherit the two-stage data flow,
the F0/harmonic prior, multi-frame deep filtering, dense cross-stage bridges,
or a second correction tower.

The experiment asks a different question:

> Can one deterministic complex-restoration tower adapt its computation to
> utterance-level noise type, frame-level noise variation, and frequency-level
> noise color without noise labels, external pretrained models, or iterative
> generation?

The implemented model is called **TriPrompt-NAF**. The historical Python name
`MambaSEUNet` is retained only as an import alias for the existing train and
inference entry points. The model itself contains no Mamba block.

## 2. Literature-overlap audit

The audit is a design boundary, not a claim that no remotely related paper
exists. The searches were performed through July 2026.

| Candidate family | Direct prior in speech enhancement | Decision |
| --- | --- | --- |
| Shared encoder with magnitude/phase decoders | MP-SENet already uses a shared encoder, TF transformers, and separate magnitude/phase decoders | Rejected as a core contribution |
| Full-band/sub-band/time modeling | TF-GridNet and FullSubNet already establish this family in speech enhancement | Rejected as an independent direction |
| Native multi-resolution STFT branches | Shi et al. feed multiple STFT resolutions into an SE model and report VoiceBank gains | Rejected as a core contribution |
| Suppress then iteratively correct | TaylorSENet, GaGNet, and VoiCor cover residual/iterative correction families | Rejected because it overlaps Scheme 3 |
| Predictive uncertainty | Fang et al. explicitly model aleatoric and epistemic uncertainty for SE | Rejected as a standalone idea |
| Frame-wise dynamic convolution | Adaptive Convolution for CNN-based SE dynamically assembles kernels per frame | Not used; prompts modulate features rather than generate convolution kernels |
| Noise-aware conditioning | NASE conditions diffusion SE on noise embeddings; DisSR uses degradation priors in diffusion speech restoration | Recognized overlap; avoid diffusion, noise classes, and external priors |
| Global rotation-equivariant phase modeling | A 2026 SE paper directly introduces global rotation equivariance | Rejected as a new core direction |
| Activation-free image restoration | NAFNet validates SimpleGate and activation-free residual blocks in image restoration | Adopted as the backbone source, then made time/frequency anisotropic |
| Degradation prompts in image restoration | PromptIR dynamically guides one restoration model using learned degradation prompts | Adopted as conceptual inspiration, not copied directly |

Closest overlap is generic degradation-aware conditioning. The differentiating
mechanism tested here is the combination of:

1. three anisotropic prompt granularities: utterance, time, and frequency;
2. clean-referenced noise-profile supervision during training, with noisy-only
   prompt inference at test time;
3. multi-scale prompt injection into a single activation-free complex tower;
4. one bounded complex residual output rather than mask/mapping dual decoders,
   iterative correction, or a generative sampler.

Primary references:

- NAFNet: https://arxiv.org/abs/2204.04676
- PromptIR: https://proceedings.neurips.cc/paper_files/paper/2023/hash/e187897ed7780a579a0d76fd4a35d107-Abstract-Conference.html
- MP-SENet: https://arxiv.org/abs/2308.08926
- TF-GridNet: https://arxiv.org/abs/2211.12433
- Multi-resolution frequency encoder/decoder SE: https://arxiv.org/abs/2303.14593
- TaylorSENet: https://www.ijcai.org/proceedings/2022/582
- VoiCor: https://www.isca-archive.org/interspeech_2024/cao24_interspeech.html
- Uncertainty-aware SE: https://arxiv.org/abs/2305.08744
- Adaptive convolution for SE: https://arxiv.org/abs/2502.14224
- Noise-aware diffusion SE: https://arxiv.org/abs/2307.08029
- Degradation-prior speech restoration: https://arxiv.org/abs/2602.12701
- Rotation-equivariant phase SE: https://arxiv.org/abs/2602.08556

## 3. Problem diagnosis

The current Scheme 3 family has already explored strong suppression, harmonic
generation, complex residual restoration, STFT consistency, and dense
cross-stage information flow. Adding another bridge or output head would not
test a sufficiently different hypothesis.

It also applies one fixed parameter response to every utterance and every
time-frequency region. VoiceBank-DEMAND contains stationary and nonstationary
noise, different noise colors, and different SNRs. A fixed restorer must use the
same internal transformations for all of them. TriPrompt-NAF instead infers a
latent degradation description from each mixture and uses it to condition the
decoder at every scale.

## 4. Data flow

For compressed noisy magnitude `A_y` and phase `P_y`, form

```
Y_r = A_y cos(P_y)
Y_i = A_y sin(P_y)
s   = mean(A_y) + epsilon
X   = concat(Y_r / s, Y_i / s, log(1 + A_y / s))
```

The restoration tower receives the full time-frequency grid. No adjacent-bin
packing or pre-network frequency compression is used, so the first restoration
level sees every frequency bin in its original neighborhood.

```text
compressed complex mixture
        |
RI + normalized log magnitude
        |
Axis-NAF encoder level 1
        |
downsample -> Axis-NAF encoder level 2
        |
downsample -> tri-granular prompt estimator
        |                 |             |
   global weights   temporal weights  spectral weights
        \_________________|_____________/
                          |
             prompted Axis-NAF bottleneck
                          |
           upsample + skip + prompt + decoder
                          |
           upsample + skip + prompt + decoder
                          |
                    prompt + refine
                          |
                 two-channel complex residual
                          |
             noisy complex spectrum + residual
```

## 5. Axis-NAF block

Each block has two identity-initialized residual sub-blocks.

The spatial sub-block performs:

1. channel-wise LayerNorm;
2. point-wise expansion;
3. parallel depth-wise `(7, 1)` temporal and `(1, 7)` spectral filtering;
4. average fusion and SimpleGate multiplication;
5. activation-free channel scaling;
6. point-wise projection and a zero-initialized residual scale.

The feed-forward sub-block uses point-wise expansion, SimpleGate, projection,
and a second zero-initialized residual scale. Explicit ReLU, GELU, PReLU, and
sigmoid activations are absent from the restoration blocks. Softmax and tanh
remain where mathematically required for prompt weights and bounded output.

This design preserves the signed geometry of real/imaginary features while
still introducing nonlinearity through multiplication.

## 6. Tri-granular prompts

The bottleneck produces three probability fields over `K=6` learned bases:

```
w_g: [B, K]       utterance-level degradation
w_t: [B, K, T_b]  frame-varying degradation
w_f: [B, K, F_b]  spectral noise color
```

Every decoder scale owns separate global, temporal, and spectral prompt bases.
Weighted bases are interpolated to the feature scale and summed into a prompt
map. A small convolution converts the map to bounded FiLM gain and bias. Prompt
modulation is applied at the bottleneck, both decoder levels, and refinement.

No noise class is supplied. The same model handles all training and test noise.

## 7. Interpretable auxiliary target

During training only, the compressed clean complex spectrum `S` gives residual
noise energy

```
E_n(f,t) = |Y(f,t) - S(f,t)|^2
E_y(f,t) = |Y(f,t)|^2
```

Temporal and spectral log-noise ratios are

```
r_t(t) = log(mean_f E_n + eps) - log(mean_f E_y + eps)
r_f(f) = log(mean_t E_n + eps) - log(mean_t E_y + eps)
```

They are clipped to `[-6, 3]` and supervise two lightweight heads with
Smooth-L1 loss. These heads share the prompt estimator representation, making
the latent conditioning measurable. At inference, only noisy speech is used.

The unchanged baseline generator losses remain:

```
L = 0.05 L_metric
  + 0.90 L_mag
  + 0.30 L_phase
  + 0.10 L_complex
  + 0.20 L_time
  + 0.10 L_consistency
  + 0.03 L_noise_profile
```

## 8. Output and stability

The network predicts one two-channel complex residual directly on the original
time-frequency grid:

```
R = 2 tanh(R_raw) * (A_y + 0.1 s)
S_hat = Y + R
```

The small floor allows recovery in low-energy mixture bins. The output head is
zero-initialized, so the complete model initially returns the noisy complex
spectrum exactly. Axis-NAF residual scales are also zero-initialized. Gradient
clipping remains at 5.0.

## 9. Capacity and local verification

- Parameters: `2,319,046`
- Non-divisible `T=31` padding/cropping test: passed
- Forward finiteness, prompt normalization, identity initialization: passed
- Backward finiteness for enhancement and profile heads: passed

The full-resolution `B=2, F=256, T=256` backward test did not complete within
five minutes on the local 8 GB RTX 4060 Ti under WDDM. Model parameter count is
not the limiting factor; full-resolution activation storage is. A Linux 4090
model-only peak-memory test and a real one-step training smoke test are therefore
mandatory before launch. Existing full-data jobs must not be stopped to make
the new run fit.

Linux verification completed on the shared RTX 4090 while both existing
full-data jobs remained active:

- full-resolution model forward/backward: `0.831 s`;
- peak model allocation: `5.132 GiB`;
- peak model reservation: `5.309 GiB`;
- one real mini training batch, including discriminator, ISTFT, PESQ, original
  losses, and noise-profile loss: passed in `3.289 s`;
- initial generator/profile loss: `1.248 / 0.439`;
- initial prompt entropy: `1.758` (`ln(6) = 1.792`), showing no initial prompt
  collapse.

## 10. Experimental controls

Branch:

```
codex/exp-trigranular-prompt-naf
```

Launched mini run:

```
experiment: trigranular_prompt_naf_mini_v1
epochs: 200
resume: disabled
distributed port: 29507
```

The Scheme 3 full and residual-dense full processes are independent controls
and must not be stopped or modified by this run.

## 11. Decision criteria

Technical gate:

- no NaN/Inf or OOM in smoke training and the first 2,000 steps;
- profile loss must show a downward trend;
- prompt entropy must remain finite; exact collapse is a warning, not an
  automatic failure unless prompt ablation shows no effect.

Performance gate against the recorded mini baseline of about PESQ `3.18` at
`60k` steps:

- `>= 3.18` at comparable steps: strong candidate for a full-data run;
- `3.15-3.18` with a positive recent slope: continue mini and decide from the
  matched-step curve;
- `< 3.15` near `60k` with a flat or declining curve: stop and reject;
- an early isolated best point is insufficient; use repeated validation points.

## 12. Follow-up ablations only if the full model is promising

1. Plain Axis-NAF tower with all prompt modulation disabled.
2. Global prompt only.
3. Tri-granular prompts without profile supervision.
4. Full TriPrompt-NAF.
5. Reduce only width or use gradient checkpointing if a later deployment study
   requires a lower-memory variant; do not silently change the main result.

These ablations are intentionally deferred. Running them before the full mini
model passes the performance gate would recreate the overly microscopic branch
sprawl this experiment is meant to avoid.
