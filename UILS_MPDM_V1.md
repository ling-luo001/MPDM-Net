# UILS-MPDM v1

## Scope

UILS-MPDM v1 is a controlled bottleneck experiment for the existing asymmetric dual-tower Mamba-SEUNet. It does not replace either tower, alter the six middle stages, change the decoders, change losses, or change the data protocol. UILS is inserted once after all six middle stages and before level-2 decoder upsampling.

UILS is not presented as the first use of Kalman filtering or RTS smoothing in speech enhancement. The experiment tests whether a small, uncertainty-controlled temporal latent smoother helps this specific MPDM-Net bottleneck.

The closest prior work includes modulation-domain Kalman speech enhancement (Wang and Brookes, 2018), Neural Kalman Filtering for Speech Enhancement (Xue et al., 2021), deep-learning-assisted Kalman speech enhancement (Roy et al., 2020), and RTSNet (Revach et al., 2023). These works make a generic novelty claim about neural Kalman filtering or learned RTS smoothing untenable. UILS is therefore a performance-screening adaptation to the MPDM-Net latent space, not a pre-validated thesis contribution.

## Mechanism

The magnitude bottleneck is projected from 48 to 8 channels and the phase bottleneck from 24 to 4 channels. The projections are concatenated as `y[B, 12, T, F]`. A pointwise controller predicts time-varying diagonal transition and noise terms:

- `a = 0.98 * tanh(raw_a)`
- `Q = clamp(softplus(raw_q) + 1e-4, max=10)`
- `R = clamp(softplus(raw_r) + 1e-4, max=10)`

Each frequency bin is independent. Filtering and smoothing loop only over `T`; batch, frequency, and latent dimensions are vectorized. The recursive path runs in FP32. Filtering uses a diagonal Joseph covariance update. Backward fixed-interval RTS gains are clamped to `[-1, 1]`.

The smoothed innovation is `x_smooth - y`. Its 8/4 channel split is projected back to 48/24 channels and injected independently:

`feature + 0.1 * tanh(gate) * tanh(projected_innovation)`

Both gates are exactly zero at construction, while the internal projections retain nonzero standard initialization. Therefore the initial candidate is an exact output and input-VJP identity relative to bypassing UILS, while the gates can learn immediately.

UILS adds 2,258 trainable parameters, below the 15,000 parameter Gate0 limit.

## Reproducibility contract

Constructing candidate UILS is wrapped by CPU RNG save and restore. With the same seed, baseline and candidate models have bit-identical shared tensors and the same post-construction CPU RNG state. The candidate and baseline mini recipes are identical except for `model_cfg.uils_enabled`.

- Baseline: `recipes/Mamba-SEUNet/Mamba-SEUNet-UILS-baseline-mini.yaml`
- Candidate: `recipes/Mamba-SEUNet/Mamba-SEUNet-UILS-candidate-mini.yaml`
- Seed: 1234
- Batch size: 2
- Data: existing mini train and validation manifests

No training protocol is changed by this implementation.

The two recipes use the dedicated process-group endpoint `tcp://localhost:29521` so they can coexist with unrelated training processes. They must still run sequentially on the 24 GiB RTX 3090 because each full-resolution training process consumes several GiB of memory.

## Gate0

Run the CPU gate with the project environment:

```text
E:\anaconda3\envs\Mamba1\python.exe scripts\test_uils_gate0.py
```

The gate checks zero-gate output and VJP identity, model RNG/shared-parameter identity, an independent FP64 linear-Gaussian reference, smoother versus filter MSE, diagnostic bounds, long constant/pulse/random sequences, staged gate/internal gradients, interface shape and finiteness, recipe parity, and the parameter budget. The full-model construction check uses explicit inspection stubs only for unavailable optional Mamba CUDA dependencies; it does not claim runtime or CUDA validation.

## CUDA profile gate

Run on the intended CUDA host with the real project dependencies:

```text
/home/lz/anaconda3/envs/mamba/bin/python scripts/profile_uils_model.py
```

The script profiles baseline and candidate in one process with the same input and warmup. It emits one JSON record containing parameters, average forward-plus-backward time, CUDA maximum allocated memory, CUDA maximum reserved memory, ratios, and pass/fail checks.

Required profile limits:

- Candidate time no more than 1.15 times baseline.
- Candidate max allocated and reserved no more than 1.12 times baseline.
- Candidate max allocated no more than 11.5 GiB.
- Candidate max reserved no more than 12.0 GiB.

## Paired mini run

The training driver accepts `--max_steps` without changing its default behavior. The paired runner uses `MAX_STEPS=100001`: training processes the validation/checkpoint labeled step `100000`, then stops before processing step `100001`.

```text
EXP_ROOT=/var/tmp/mpdm-uils-paired \
  bash scripts/run_uils_paired_mini.sh
```

The candidate runs first. After it exits successfully, the same script launches the fresh baseline with the same seed, data ordering, optimizer, losses, and validation cadence. Output is kept off `/home` because that partition has insufficient free space for both checkpoint series.

## Mini promotion rule

Promote UILS beyond the paired mini experiment only if all criteria hold:

- PESQ at 100k is at least baseline plus 0.020.
- Mean PESQ across 80k, 90k, and 100k is at least baseline plus 0.015.
- PESQ at each of 80k, 90k, and 100k is not below baseline.
- STOI degradation is no more than 0.002.
- SI-SDR degradation is no more than 0.2 dB.

Failure of any criterion rejects promotion; it does not authorize a protocol change or a new research route.
