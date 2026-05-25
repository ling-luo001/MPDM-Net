# Copilot Agent Prompt: P001_V2_mag_decoder_local_cab

## 1. Version Control Info

- Version Name: `P001_V2_mag_decoder_local_cab`
- Base Branch: `<filled by user>`
- Base Commit: `<filled by user>`
- Current Experiment Branch: `<filled by user>`

Important:

The user has already prepared the correct Git branch before running this prompt.

Copilot Agent must not create, switch, merge, reset, rebase, or delete Git branches.

This experiment must be implemented as a single-variable modification based on the recorded base commit.

Unless explicitly stated, this version must not depend on other experiment versions such as V1, V2, or V3.

## 2. Goal

Test whether MambaIR-style local convolution plus channel attention improves magnitude decoder reconstruction after upsampling and skip fusion.

## 3. Experiment Summary

Add local CAB-style residual modules only to the magnitude decoder level-2 and level-1 paths after their existing `TFMambaBlock` stacks and before residual addition with the decoder copy tensors.

## 4. Baseline Context

In `models/generator.py`, the magnitude decoder reconstructs `mag_y2` and `mag_y1` after upsampling and skip concatenation. These features are close to the mask decoder and are likely to affect spectral texture and perceptual quality. MambaIR's local convolution plus channel attention is most relevant here as a local restoration enhancer.

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

1. In `models/mamba_block.py`, add a reusable MambaIR-style local CAB module if it does not already exist in this branch.
2. The module must preserve 4D tensor shape `[B, C, T, F]`.
3. Use local convolution plus channel attention only. Do not add spatial attention, cross fusion, SS2D, or new Mamba scans.
4. Use a residual scale initialized to zero or a very small value.
5. In `models/generator.py`, import the new module.
6. In `MambaSEUNet.__init__`, add:
   - one module for `mag_dim[1]` decoder level 2
   - one module for `mag_dim[0]` decoder level 1
7. In `MambaSEUNet.forward`, apply the level-2 module after the loop over `self.mag_TSMamba2_decoder` and before `mag_y2 = mag_y2_copy + mag_y2`.
8. Apply the level-1 module after the loop over `self.mag_TSMamba1_decoder` and before `mag_y1 = mag_y1_copy + mag_y1`.
9. Do not apply this module to the encoder, bottleneck, refinement, phase branch, mid-fusion, or global fusion.
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

- Level 2 input/output: `[B, mag_dim[1], T/2 or aligned decoder time, F_level2]`, exactly the same shape before and after the module.
- Level 1 input/output: `[B, mag_dim[0], T or aligned decoder time, F_level1]`, exactly the same shape before and after the module.
- The residual additions with `mag_y2_copy` and `mag_y1_copy` must remain valid without broadcasting.

## 10. What Not to Change

- Do not change training logic.
- Do not change dataset loading.
- Do not change loss functions or loss weights.
- Do not change model input/output interface.
- Do not modify phase branch modules.
- Do not modify bottleneck or fusion modules.
- Do not replace existing decoder `TFMambaBlock` stacks.
- Do not introduce large refactoring.
- Do not perform any Git branch operation.

## 11. Sanity Checks

Copilot should check:

- syntax correctness
- shape compatibility before `mag_y2_copy + mag_y2`
- shape compatibility before `mag_y1_copy + mag_y1`
- forward pass compatibility
- only allowed files changed
- rollback requires removing two module instances and two forward calls

## 12. Final Report Required from Copilot

Copilot must report:

1. changed files
2. new modules/classes/functions
3. modified forward path
4. shape assumptions
5. whether any config was changed
6. how to disable or rollback the change
7. confirmation that no Git branch operation was performed
