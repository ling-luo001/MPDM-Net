# Copilot Agent Prompt: P001_V1_mag_refinement_local_cab

## 1. Version Control Info

- Version Name: `P001_V1_mag_refinement_local_cab`
- Base Branch: `<filled by user>`
- Base Commit: `<filled by user>`
- Current Experiment Branch: `<filled by user>`

Important:

The user has already prepared the correct Git branch before running this prompt.

Copilot Agent must not create, switch, merge, reset, rebase, or delete Git branches.

This experiment must be implemented as a single-variable modification based on the recorded base commit.

Unless explicitly stated, this version must not depend on other experiment versions such as V1, V2, or V3.

## 2. Goal

Test whether adding a MambaIR-style local convolution plus channel-attention residual to the final magnitude refinement feature improves spectral detail reconstruction and perceptual metrics.

## 3. Experiment Summary

Add one local CAB-style residual module after the existing magnitude refinement Mamba stack and before `mag_output`. Do not touch the phase branch, losses, data, training scripts, or input/output interface.

## 4. Baseline Context

MPDM-Net uses `MambaSEUNet` in `models/generator.py`. The magnitude branch uses `TFMambaBlock` for encoder, decoder, and refinement. The final magnitude refinement path computes `mag_y1`, adds `mag_copy_ref`, then sends the feature through `mag_output`. MambaIR suggests that local convolution and channel attention can compensate for local detail forgetting in Mamba-style restoration blocks.

## 5. Files to Inspect First

- `models/generator.py`
- `models/mamba_block.py`
- `models/codec_module.py`

## 6. Files Allowed to Modify

- `models/generator.py`
- `models/mamba_block.py`

## 7. Files Forbidden to Modify

- `train.py`
- `test.py`
- `inference.py`
- `datasets/`
- `dataloaders/`
- `recipes/`
- `models/loss.py`
- `models/cross.py`
- environment files
- checkpoint files
- log files
- config files unless the user explicitly permits config changes

## 8. Required Implementation

1. In `models/mamba_block.py`, add a compact MambaIR-style local convolution and channel attention module, for example `MambaIRLocalCAB`.
2. The module must accept and return a 4D tensor with shape `[B, C, T, F]`.
3. Use a residual form: `output = input + scale * local_attention(input)`.
4. Initialize the residual scale to zero or a very small value so the initial behavior is close to the baseline.
5. Use only local convolutions and channel attention:
   - local convolution path may use `Conv2d(C, C // r, 3, padding=1)`, `GELU`, `Conv2d(C // r, C, 3, padding=1)`;
   - channel attention may use adaptive average pooling followed by `1x1` convolutions and sigmoid.
6. Avoid spatial attention, cross attention, SS2D, or new Mamba blocks in this version.
7. In `models/generator.py`, import the new module.
8. Add one instance such as `self.mag_refine_local_cab = MambaIRLocalCAB(mag_dim[0])` in `MambaSEUNet.__init__`.
9. In `MambaSEUNet.forward`, apply it only in the magnitude refinement path after:
   - `mag_y1 = mag_y1 + mag_copy_ref`
   - and before `mag_final = self.mag_output(...) + mag_skip1`
10. Keep the phase refinement path unchanged.
11. Do not add config switches unless necessary.

## 9. Interface Constraints

The model input/output format must remain unchanged.

Preserve:

- noisy magnitude input
- noisy phase input
- magnitude mask output
- phase correction output
- enhanced complex spectrum output
- enhanced waveform output if produced by the caller

Feature shape constraints:

- Input to the new module: `[B, mag_dim[0], T, F']`
- Output from the new module: same shape `[B, mag_dim[0], T, F']`
- `mag_mask`, `denoised_mag`, `pred_pha`, and `denoised_com` shapes must remain unchanged.

## 10. What Not to Change

- Do not change training logic.
- Do not change dataset loading.
- Do not change loss functions or loss weights.
- Do not change STFT or ISTFT utilities.
- Do not change model input/output interface.
- Do not modify phase branch modules.
- Do not modify mid-fusion or global-fusion logic.
- Do not replace `TFMambaBlock`, `TMambaBlock`, or `FMambaBlock`.
- Do not introduce large refactoring.
- Do not perform any Git branch operation.

## 11. Sanity Checks

Copilot should check:

- syntax correctness
- tensor shape compatibility for `[B, C, T, F]`
- forward pass compatibility using the existing model construction pattern
- no NaN/Inf introduced by the new module
- only the allowed files changed
- the change can be disabled by removing one module instance and one forward call

## 12. Final Report Required from Copilot

Copilot must report:

1. changed files
2. new modules/classes/functions
3. modified forward path
4. shape assumptions
5. whether any config was changed
6. how to disable or rollback the change
7. confirmation that no Git branch operation was performed




• Processed paper_inputs/P001_input.md after reading codex_start_prompt.md, AGENTS.md, baseline_summary.md, and task_requirements.md.

  Created:

  - paper_notes/P001_report.md
  - prompts_for_copilot/P001_V1_prompt.md
  - prompts_for_copilot/P001_V2_prompt.md
  - prompts_for_copilot/P001_V3_prompt.md

  Updated:

  - experiment_registry.md

  Decision: P001 is worth testing only as a narrow MambaIR-style local convolution + channel-attention enhancement, not as a full MambaIR/SS2D backbone transplant. I designed three independent single-variable experiments targeting magnitude refinement, magnitude decoder, and
  magnitude bottleneck respectively.

  No source code, Git commands, configs, losses, training scripts, datasets, checkpoints, or logs were modified. Sources used: https://github.com/csguoh/MambaIR and https://arxiv.org/abs/2402.15648.

─ Worked for 3m 24s ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────Token usage: total=82,544 input=73,738 (+ 360,576 cached) output=8,806 (reasoning 1,035)
To continue this session, run codex resume 019e44e1-e8a1-7531-b07c-f20994a70291