# P001 Paper-to-Experiment Report

## 1. Paper Idea Summary

- Paper/module name: MambaIR: A Simple Baseline for Image Restoration with State-Space Model
- Original task: Image restoration, especially image super-resolution and denoising.
- Core idea: Improve vanilla visual Mamba for low-level restoration by adding local convolutional enhancement and channel attention around state-space feature modeling.
- Main module: Residual State Space Block / VSSBlock with SS2D, local convolutional attention block, and residual state-space groups.
- Source: https://github.com/csguoh/MambaIR and https://arxiv.org/abs/2402.15648
- Why it may be relevant to MPDM-Net: MPDM-Net is also a low-level restoration model using Mamba-style blocks, but for speech enhancement. The transferable part is not the image backbone; it is the local detail enhancement plus channel interaction around Mamba features, especially in the magnitude branch where spectral texture and perceptual quality are prioritized.

## 2. Candidate Idea Extraction

| Idea ID | Idea | Category | Original role | Possible MPDM-Net target |
|--------|------|----------|---------------|---------------------------|
| P001-I1 | Add MambaIR-style local convolution + channel attention after Mamba features | Local restoration enhancement | Reduce local pixel forgetting and channel redundancy in VSS blocks | Magnitude refinement or decoder features |
| P001-I2 | Residual State Space Group with repeated VSS blocks and group-level convolution | Backbone grouping | Deep image restoration trunk | Not suitable as a direct transplant |
| P001-I3 | Four-direction SS2D image scan | 2D state-space scan | Global image dependency modeling | Mostly redundant with existing VSS-style cross fusion and axis-specific TMamba/FMamba |
| P001-I4 | Full image SR/denoising reconstruction head | Task-specific decoder | Pixel-space image reconstruction | Not compatible with speech enhancement output interface |

## 3. Comparison with MPDM-Net

| Idea ID | Similar part in MPDM-Net | Complementary or redundant | Risk | Decision |
|--------|---------------------------|-----------------------------|------|----------|
| P001-I1 | DenseBlock, Patch_Embed_stage, TFMambaBlock, TMambaBlock, FMambaBlock, unused CBAM utility | Complementary if applied narrowly after magnitude-side Mamba blocks | Low to medium; may over-smooth spectral detail if inserted too broadly | High priority |
| P001-I2 | Existing U-Net encoder/decoder with bottleneck and skip paths | Mostly redundant and too large | High; replaces architecture scale and grouping | Reject |
| P001-I3 | `VSSBlock_Cross_new` in mid/global fusion, `TMambaBlock` and `FMambaBlock` axis scans | Mostly redundant | Medium to high; duplicated scan logic and possible tensor layout errors | Reject |
| P001-I4 | `MagDecoder`, `PhaseDecoder`, waveform reconstruction logic | Incompatible | High; changes task output | Reject |

## 4. Recommended Experiments

### P001_V1

- Goal: Test whether a MambaIR-style local convolution + channel attention residual improves final magnitude detail reconstruction.
- Target module: Magnitude refinement path in `models/generator.py`.
- Modification: Add a new local CAB-style module in `models/mamba_block.py`, then apply it only after the magnitude refinement stack before `mag_output`.
- Expected benefit: Better spectral texture/detail reconstruction, potentially improving PESQ, CSIG, and COVL.
- Main risk: Extra local filtering may smooth fine harmonic detail if the attention gate becomes too strong.
- Files Copilot should inspect: `models/mamba_block.py`, `models/generator.py`, `models/codec_module.py`.
- Files Copilot may modify: `models/mamba_block.py`, `models/generator.py`.
- Files Copilot must not modify: training scripts, dataset code, configs, loss files, checkpoints, logs, unrelated model files.
- Sanity check: Forward pass preserves `[B, C, T, F]` for magnitude refinement features and leaves model input/output unchanged.
- Rollback method: Remove the new local CAB module and the single call/site in magnitude refinement.
- Version dependency:
  - Should start from the same stable baseline commit as other single-variable versions.
  - Should not depend on V2 or V3.

### P001_V2

- Goal: Test whether MambaIR-style local convolution + channel attention improves magnitude decoder reconstruction after skip fusion.
- Target module: Magnitude decoder level-1 and level-2 feature path in `models/generator.py`.
- Modification: Add local CAB-style residual modules after the magnitude decoder Mamba stacks at level 2 and level 1, before residual addition with the decoder copies.
- Expected benefit: Better local spectral reconstruction after skip concatenation and upsampling.
- Main risk: More parameters in decoder and possible redundancy with existing DenseBlock decoder.
- Files Copilot should inspect: `models/mamba_block.py`, `models/generator.py`, `models/codec_module.py`.
- Files Copilot may modify: `models/mamba_block.py`, `models/generator.py`.
- Files Copilot must not modify: training scripts, dataset code, configs, loss files, checkpoints, logs, unrelated model files.
- Sanity check: `mag_y2` and `mag_y1` shapes remain unchanged before residual addition.
- Rollback method: Remove the decoder local CAB modules and their calls.
- Version dependency:
  - Should start from the same stable baseline commit as other single-variable versions.
  - Should not depend on V1 or V3.

### P001_V3

- Goal: Test whether MambaIR-style local convolution + channel attention improves magnitude bottleneck features before magnitude-phase fusion.
- Target module: Magnitude middle stage in `models/generator.py`.
- Modification: Add a local CAB-style residual only on the magnitude bottleneck feature after each magnitude middle Mamba block and before the existing mid-fusion projection.
- Expected benefit: Stronger magnitude-side local spectral representation before cross-branch fusion.
- Main risk: The bottleneck already has FMamba/TMamba alternation and VSS cross fusion, so the gain may be small or redundant.
- Files Copilot should inspect: `models/mamba_block.py`, `models/generator.py`, `models/cross.py`.
- Files Copilot may modify: `models/mamba_block.py`, `models/generator.py`.
- Files Copilot must not modify: training scripts, dataset code, configs, loss files, checkpoints, logs, unrelated model files.
- Sanity check: `mag_feat`, `mag_in_fuse`, `mag_fused`, and `mag_x3` shapes remain compatible with existing mid-fusion projections.
- Rollback method: Remove the bottleneck local CAB module list and calls before mid-fusion.
- Version dependency:
  - Should start from the same stable baseline commit as other single-variable versions.
  - Should not depend on V1 or V2.

## 5. Rejected Ideas

| Idea | Reason |
|------|--------|
| Replace MPDM-Net Mamba blocks with MambaIR SS2D/VSSBlock | Too broad, duplicates existing VSS-style fusion, and risks changing the core backbone rather than testing one variable |
| Add full Residual State Space Groups | Would redesign the backbone and make comparison hard |
| Use MambaIR image upsampling/reconstruction heads | Incompatible with magnitude mask, phase correction, complex spectrum, and waveform output interface |
| Apply local CAB to both magnitude and phase branches | Phase branch has previously been sensitive to aggressive changes; this would mix branch effects and violate single-variable clarity |

## 6. Final Recommendation

P001 is worth testing only as a narrow local-restoration enhancement, not as a full MambaIR architecture transplant.

Recommended order:

1. `P001_V1`: safest and most directly tied to final magnitude detail.
2. `P001_V2`: useful if decoder detail is suspected to be the bottleneck.
3. `P001_V3`: possible bottleneck test, but most likely to overlap with existing mid-fusion logic.
