# Copilot Agent Prompt: P001_V3_mag_bottleneck_local_cab

## 1. Version Control Info

- Version Name: `P001_V3_mag_bottleneck_local_cab`
- Base Branch: `<filled by user>`
- Base Commit: `<filled by user>`
- Current Experiment Branch: `<filled by user>`

Important:

The user has already prepared the correct Git branch before running this prompt.

Copilot Agent must not create, switch, merge, reset, rebase, or delete Git branches.

This experiment must be implemented as a single-variable modification based on the recorded base commit.

Unless explicitly stated, this version must not depend on other experiment versions such as V1, V2, or V3.

## 2. Goal

Test whether adding MambaIR-style local convolution plus channel attention to the magnitude bottleneck improves magnitude features before magnitude-phase fusion.

## 3. Experiment Summary

Add local CAB-style residual modules only after magnitude middle Mamba blocks and before the existing mid-fusion projection. Do not modify phase bottleneck processing or the fusion module itself.

## 4. Baseline Context

The MPDM-Net bottleneck alternates magnitude `FMambaBlock` and `TMambaBlock` with phase `TMambaBlock` and `FMambaBlock`, followed by `VSSBlock_Cross_new` mid-fusion. This is a high-impact area for magnitude-phase interaction. MambaIR's local enhancement may improve the magnitude representation before fusion, but it must be kept narrow to avoid confounding the fusion design.

## 5. Files to Inspect First

- `models/generator.py`
- `models/mamba_block.py`
- `models/cross.py`

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

1. In `models/mamba_block.py`, add a reusable MambaIR-style local CAB module if it does not already exist in this branch.
2. The module must preserve 4D tensor shape `[B, C, T, F]`.
3. Use local convolution plus channel attention only.
4. Use a residual scale initialized to zero or a very small value.
5. In `models/generator.py`, import the new module.
6. In `MambaSEUNet.__init__`, add a `nn.ModuleList` for magnitude bottleneck local CAB modules with length `self.num_mid_stages`, each using `mag_dim[2]` channels.
7. In `MambaSEUNet.forward`, inside the existing middle loop:
   - compute `mag_feat = mag_block(mag_x3)` as before;
   - apply the local CAB only to `mag_feat`;
   - keep `pha_feat = pha_block(pha_x3)` unchanged;
   - pass the enhanced `mag_feat` into `self.mid_in_proj_mag[idx]` as before.
8. Do not change `VSSBlock_Cross_new`, `mid_in_proj_*`, `mid_fusion_proj_*`, or residual additions.
9. Do not apply local CAB to the phase branch, decoder, encoder, refinement, or global fusion.
10. Do not add config switches unless necessary.

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

- `mag_feat` before local CAB: `[B, mag_dim[2], T_mid, F_mid]`
- `mag_feat` after local CAB: same shape `[B, mag_dim[2], T_mid, F_mid]`
- `mag_in_fuse = self.mid_in_proj_mag[idx](mag_feat)` must keep the same expected channel count.
- `mag_cat`, `mag_x3`, and the residual add with `mag_res` must remain shape-compatible.

## 10. What Not to Change

- Do not change training logic.
- Do not change dataset loading.
- Do not change loss functions or loss weights.
- Do not change model input/output interface.
- Do not modify phase branch modules.
- Do not modify `models/cross.py`.
- Do not change fusion module internals.
- Do not replace `FMambaBlock`, `TMambaBlock`, or `TFMambaBlock`.
- Do not introduce large refactoring.
- Do not perform any Git branch operation.

## 11. Sanity Checks

Copilot should check:

- syntax correctness
- shape compatibility before and after `self.mid_in_proj_mag[idx]`
- shape compatibility through `self.mid_fusions[idx]`
- forward pass compatibility
- only allowed files changed
- rollback requires removing one module list and one forward call inside the middle loop

## 12. Final Report Required from Copilot

Copilot must report:

1. changed files
2. new modules/classes/functions
3. modified forward path
4. shape assumptions
5. whether any config was changed
6. how to disable or rollback the change
7. confirmation that no Git branch operation was performed
