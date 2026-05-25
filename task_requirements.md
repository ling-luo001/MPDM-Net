# Task Requirements

This file defines the workflow for converting a paper idea, paper module, or paper code into MPDM-Net experiment prompts.

Use this file after reading:

- `AGENTS.md`
- `baseline_summary.md`

## 1. Main Task

When I paste a paper summary, method section, module description, paper code, or GitHub code snippet, your task is to:

1. Understand the useful idea.
2. Compare it with MPDM-Net.
3. Judge whether it is worth testing.
4. Design small controlled experiments if useful.
5. Generate Copilot Agent prompts for implementation.
6. Save the results as Markdown files.

Do not directly modify source code.

The most important output is the V1 / V2 / V3 Copilot prompt files.

## 2. Analysis Focus

Always judge the paper idea against MPDM-Net.

Focus on whether the idea can improve:

- magnitude modeling
- phase correction
- magnitude-phase fusion
- time-frequency modeling
- bottleneck representation
- decoder/detail reconstruction
- perceptual quality metrics such as PESQ, CSIG, and COVL

Be conservative.

Reject ideas that are:

- redundant
- too risky
- too large
- mainly for lightweighting
- unrelated to MPDM-Net's current bottlenecks
- incompatible with the magnitude-phase dual-branch design

## 3. Decision Labels

For every candidate idea, assign one decision:

- High priority: worth trying first
- Medium priority: possible but not urgent
- Low priority: weak value or high uncertainty
- Reject: not suitable for current MPDM-Net experiments

Do not force experiments from weak ideas.

## 4. Experiment Design Rules

Each experiment must be:

- single-variable
- easy to rollback
- compatible with the current input/output format
- suitable for direct comparison with the baseline commit

Do not mix several changes in one version.

Do not combine model-structure changes with loss-function changes unless explicitly requested.

Generate at most 3 experiment versions for one paper.

By default:

- V1, V2, and V3 must all start from the same stable baseline commit.
- V2 must not depend on V1.
- V3 must not depend on V2.

If a version intentionally depends on a previous successful experiment, mark it clearly as a combined experiment.

## 5. Output Files

Use a short paper ID, for example:

- `P001`
- `CMGAN`
- `SEMamba`
- `TFGridNet`

For each suitable paper, create:

- `paper_notes/<paper_id>_report.md`
- `prompts_for_copilot/<paper_id>_V1_prompt.md`
- `prompts_for_copilot/<paper_id>_V2_prompt.md`
- `prompts_for_copilot/<paper_id>_V3_prompt.md`

Only create prompt files for experiments that are actually worth testing.

If the paper is not suitable, create:

- `paper_notes/<paper_id>_report.md`
- `rejected_ideas/<paper_id>_rejected.md`

Do not create unnecessary files.

## 6. Report Format

The report file should be saved as:

`paper_notes/<paper_id>_report.md`

Use this structure:

# <paper_id> Paper-to-Experiment Report

## 1. Paper Idea Summary

- Paper/module name:
- Original task:
- Core idea:
- Main module:
- Why it may be relevant to MPDM-Net:

## 2. Candidate Idea Extraction

| Idea ID | Idea | Category | Original role | Possible MPDM-Net target |
|--------|------|----------|---------------|---------------------------|

## 3. Comparison with MPDM-Net

| Idea ID | Similar part in MPDM-Net | Complementary or redundant | Risk | Decision |
|--------|---------------------------|-----------------------------|------|----------|

## 4. Recommended Experiments

### <paper_id>_V1

- Goal:
- Target module:
- Modification:
- Expected benefit:
- Main risk:
- Files Copilot should inspect:
- Files Copilot may modify:
- Files Copilot must not modify:
- Sanity check:
- Rollback method:
- Version dependency:
  - Should start from the same stable baseline commit as other single-variable versions.
  - Should not depend on V2 or V3.

### <paper_id>_V2

### <paper_id>_V3

## 5. Rejected Ideas

| Idea | Reason |
|------|--------|

## 6. Final Recommendation

Clearly state whether this paper is worth testing.

If worth testing, rank the experiment versions.

If not worth testing, explain why.

## 7. Copilot Prompt Format

Each Copilot prompt should be saved as:

`prompts_for_copilot/<paper_id>_V<num>_prompt.md`

Use this structure:

# Copilot Agent Prompt: <paper_id>_V<num>_<feature>

## 1. Version Control Info

- Version Name: `<paper_id>_V<num>_<feature>`
- Base Branch: `<filled by user>`
- Base Commit: `<filled by user>`
- Current Experiment Branch: `<filled by user>`

Important:

The user has already prepared the correct Git branch before running this prompt.

Copilot Agent must not create, switch, merge, reset, rebase, or delete Git branches.

This experiment must be implemented as a single-variable modification based on the recorded base commit.

Unless explicitly stated, this version must not depend on other experiment versions such as V1, V2, or V3.

## 2. Goal

Describe the exact goal of this controlled experiment.

## 3. Experiment Summary

Briefly describe the exact single-variable change.

## 4. Baseline Context

Describe the relevant MPDM-Net module and why this location is selected.

## 5. Files to Inspect First

List the files Copilot should inspect before editing.

## 6. Files Allowed to Modify

List only the files that may be modified.

## 7. Files Forbidden to Modify

List files or folders that must not be modified.

Usually forbidden:

- training scripts
- dataset code
- environment files
- checkpoint files
- logs
- unrelated model files
- loss files unless this experiment explicitly targets loss design

## 8. Required Implementation

Give step-by-step implementation instructions.

The instructions should be concrete and should not ask Copilot to freely optimize the model.

## 9. Interface Constraints

The model input/output format must remain unchanged.

Preserve:

- noisy magnitude input
- noisy phase input
- magnitude mask output
- phase correction output
- enhanced complex spectrum output
- enhanced waveform output

If tensor shapes are known, state them.

For feature blocks, preserve the input and output shape of the modified block unless explicitly stated otherwise.

## 10. What Not to Change

Clearly list forbidden changes.

Examples:

- do not change training logic
- do not change dataset loading
- do not change loss weights
- do not change model input/output interface
- do not modify unrelated branches
- do not introduce large refactoring
- do not add unnecessary config switches

## 11. Sanity Checks

Copilot should check:

- syntax correctness
- tensor shape compatibility
- forward pass compatibility
- whether the modified module can be rolled back
- whether unrelated files were not changed

## 12. Final Report Required from Copilot

Copilot must report:

1. changed files
2. new modules/classes/functions
3. modified forward path
4. shape assumptions
5. whether any config was changed
6. how to disable or rollback the change
7. confirmation that no Git branch operation was performed

## 8. Quality Standard

A good result should be:

- clear
- conservative
- executable
- useful for Copilot Agent
- based on MPDM-Net rather than generic model improvement
- suitable for single-variable experimental comparison

A bad result is:

- vague
- too ambitious
- hard to rollback
- modifying many modules at once
- ignoring the baseline design
- making V2 depend on V1 without explicit instruction
- asking Copilot to freely optimize the model

## 9. Default Behavior

If I paste paper content without extra instructions, follow this workflow automatically:

1. Read the pasted content.
2. Read `baseline_summary.md`.
3. Inspect project source files only if needed to identify real file names or module names.
4. Create one report in `paper_notes/`.
5. Create up to three Copilot prompts in `prompts_for_copilot/`.
6. If unsuitable, create a rejection note instead.

Keep the analysis practical and implementation-oriented.