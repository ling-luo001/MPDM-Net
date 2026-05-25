# MPDM-Net Baseline Summary

## 1. Version Control Note

This file describes the model architecture and research priorities.

It does not define the exact code version.

The exact baseline version for each experiment must be determined manually by the user using Git:

- Base Branch
- Base Commit
- Current Experiment Branch

Every experiment should record its base commit before implementation.

By default, different experiment versions from the same paper should start from the same stable baseline commit.

## 2. Task and Goal

Task:

- Single-channel speech enhancement / speech denoising.

Current goal:

- Improve enhancement quality.

Main metrics:

- PESQ
- STOI
- SI-SDR
- CSIG
- CBAK
- COVL

Priority metrics:

- PESQ
- CSIG
- COVL

STOI and SI-SDR should not degrade significantly.

## 3. Input and Output Interfaces

Input:

- noisy magnitude
- noisy phase

Output:

- magnitude mask
- phase correction
- enhanced complex spectrum
- enhanced waveform

Hard constraint:

- The input/output interface must not be changed unless explicitly requested.

## 4. Overall Architecture

MPDM-Net is a Mamba-based magnitude-phase dual-branch speech enhancement model.

It follows an asymmetric U-Net / encoder-decoder structure.

Main components include:

- DenseEncoder
- DenseBlocks
- downsampling
- bottleneck
- upsampling
- skip connections
- TMamba
- FMamba
- TFMamba
- bidirectional Mamba blocks
- magnitude-phase cross fusion
- final global fusion

## 5. Magnitude Branch

The magnitude branch is the primary and stronger branch.

It focuses on:

- spectral texture reconstruction
- time-frequency dependency modeling
- magnitude mask prediction

It mainly uses:

- TFMamba
- alternating FMamba / TMamba blocks

Ideas related to better time-frequency modeling, spectral detail reconstruction, decoder refinement, or bottleneck representation may be useful.

## 6. Phase Branch

The phase branch is narrower and more conservative.

It focuses on:

- phase correction
- phase refinement
- phase rotation-style correction

It does not directly regress absolute phase.

It mainly uses:

- TMamba

Important note:

Previous aggressive phase-branch modifications performed poorly.

Therefore, prefer stable and minor phase correction improvements over large phase-branch redesigns.

## 7. Magnitude-Phase Fusion

The model already contains magnitude-phase interaction in:

- bottleneck fusion
- final global fusion

Existing fusion may include:

- local enhancement
- differential enhancement
- 2D selective scan / VSS-style interaction
- channel calibration

New fusion ideas must be compared against the existing fusion design.

Reject ideas that simply duplicate existing fusion mechanisms without clear additional benefit.

