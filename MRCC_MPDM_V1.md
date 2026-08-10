# MRCC-MPDM v1 design and staged experiment

## Mechanism

MRCC is a post-estimator correction module. The original magnitude/phase dual towers, seven configured VSS interactions, mask decoder, phase decoder, input API, and three-output return contract are unchanged. The baseline compressed magnitude and phase are first converted to the native linear real/imaginary spectrum. The compressed noisy input is independently converted back to its linear native spectrum and waveform, so no dataset API changes are needed.

At 128/128/32 and 256/256/64 FFT/window/hop resolutions, a shared convolutional proposer receives six real-valued channels: noisy real/imaginary, baseline real/imaginary, and baseline-minus-noisy real/imaginary. A learned resolution embedding conditions all residual blocks. The two-channel proposal vector norm is clipped to at most one and bounded to `0.1 * (noisy auxiliary magnitude + eps)`. Reliability is sigmoid-mapped into `(0.05, 0.95)`.

Starting from exactly zero, MRCC performs exactly two differentiable shared-waveform correction iterations. At every iteration and auxiliary resolution, it analyzes the current waveform correction, forms the full-lattice residual `S_r(x) - d_r`, applies reliability independently at every time-frequency bin, and synthesizes the weighted residual back to a waveform update. The two synthesized updates and waveform damping are combined before one bounded, scalar-preconditioned step. Reliability is never averaged over frequency in either the update or objective.

`J(x) = damping * mean(x^2) + sum_r mean(w_r * |S_r(x) - d_r|^2)`.

The fixed step is at most one and is divided by damping plus the maximum reliability from each resolution. After each proposed step, the actual full-TF objective is recomputed. A detached per-example accept/no-op mask selects the trial only when that objective does not increase; the selected branch remains differentiable and there is no line search or step growth. The native STFT of the shared waveform correction is added to the original native complex estimate. Corrected linear real/imaginary values are then converted back to the repository's compressed magnitude, wrapped phase, and compressed real/imaginary convention.

Both per-resolution gains are naturally zero-initialized. Forward output therefore matches the baseline within numerical round-trip tolerance, and the first backward produces ordinary finite gradients for those gains. The proposer is intentionally inactive at exact zero gain; after the optimizer moves either gain away from zero, ordinary product-rule gradients activate the shared proposer. No surrogate or straight-through gradient is used.

The corrected spectrum is returned as an output-domain residual over the original baseline outputs. Consequently, zero gains preserve the original output values and Jacobian rather than merely reproducing the same numbers through a second magnitude/phase conversion. The tracked `max_grad_norm: 5.0` setting is applied to both generator and discriminator before their optimizer steps.

## Novelty boundary

The experiment tests cross-resolution correction agreement after the native MPDM estimate. It does not claim a new backbone, input-level magnitude/phase disentanglement, a new loss, or novelty from adding an ordinary prediction head. Any thesis claim requires later literature and result-based validation; this implementation alone establishes no novelty or quality gain.

## Tracked mini configuration

- Recipe: `recipes/Mamba-SEUNet/MRCC-MPDM-v1-mini.yaml`
- Native STFT: 510/510/120, Hann, centered, compression factor 0.3
- Auxiliary STFTs: 128/128/32 and 256/256/64, Hann, centered
- Shared proposer: width 64, four depthwise-separable residual blocks, condition dimension 240
- Consensus: exactly two iterations, fixed step 0.5, full-TF reliability, damping 0.05
- Training: existing mini manifests, 200 epochs, distributed port 29512
- Historical mini experiment: `mrcc_mpdm_mini_v1`
- Mini result: best validation PESQ `3.219019 @ 96k`; final recorded value `3.178951 @ 174k`
- Explicit mini launch: `python train.py --config recipes/Mamba-SEUNet/MRCC-MPDM-v1-mini.yaml --exp_name mrcc_mpdm_mini_v1`
- Resume: the numerically latest step present as both `g_XXXXXXXX.pth` and `do_XXXXXXXX.pth`

## Full deployment

- Recipe: `recipes/Mamba-SEUNet/MRCC-MPDM-v1-full.yaml`
- Dataset: 11,572 training pairs and 824 validation pairs on the 4090 host, byte-identical to the Stabilized TF-LCA full manifests
- Validation composition: 576 historical validation pairs plus 248 historical test pairs for compatibility with earlier full curves; final thesis evaluation requires a strictly separated validation/test protocol
- Training: 400 epochs, batch size 2, learning rate `5e-4`, distributed port 29512
- Default direct launch: `python train.py`
- Default experiment: `mrcc_mpdm_full_v1`
- Deployment commit: `d9c38c9`
- Initial 576-validation run was archived as `mrcc_mpdm_full_v1_invalid_val576` and excluded from comparisons
- Comparable fresh run was restarted from zero with the shared 824-pair validation manifest

## Gates

1. Disabled path preserves exact baseline return objects.
2. Zero gains preserve baseline outputs and their output Jacobian within numerical tolerance, while producing finite nonzero gain gradients and naturally zero proposer gradients.
3. Small nonzero gains activate finite nonzero proposer gradients.
4. Reliability and correction bounds hold; all outputs and backward gradients are finite.
5. Full-TF reliability placement changes consensus even when frequency means are equal.
6. The actual full-TF consensus objective is non-increasing over both fixed iterations.
7. Total generator parameters are 2.25M-2.45M.
8. Representative measured forward/backward ratio is at most 1.70x baseline where CUDA dependencies are available.
9. Mini training was launched only after the unit and profile gates passed in the target runtime environment.

## Required ablations

- Original MPDM-Net baseline with MRCC disabled.
- Capacity-matched native-only complex proposer, with no auxiliary resolutions.
- No-consensus averaging of the two projected correction waveforms.
- Full two-resolution MRCC with exactly two consensus iterations.
