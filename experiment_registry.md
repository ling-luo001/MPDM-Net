# Experiment Registry

This file records proposed and executed experiments.

The user manually controls Git branches and base commits.

Each single-variable experiment should record its own base branch and base commit.

Experiments from the same paper should normally start from the same stable baseline commit.

| Version | Paper | Base Branch | Base Commit | Experiment Branch | Target Module | Status | Result Summary |
|--------|-------|-------------|-------------|-------------------|---------------|--------|----------------|
| Example_V1 | ExamplePaper | main | abc1234 | exp/Example_V1 | bottleneck fusion | proposed | not tested |
| P001_V1 | MambaIR | `<filled by user>` | `<filled by user>` | `<filled by user>` | magnitude refinement local CAB | proposed | MambaIR-style local convolution plus channel attention after magnitude refinement; single-variable, not tested |
| P001_V2 | MambaIR | `<filled by user>` | `<filled by user>` | `<filled by user>` | magnitude decoder local CAB | proposed | MambaIR-style local convolution plus channel attention after magnitude decoder Mamba stacks; single-variable, not tested |
| P001_V3 | MambaIR | `<filled by user>` | `<filled by user>` | `<filled by user>` | magnitude bottleneck local CAB | proposed | MambaIR-style local convolution plus channel attention before mid-fusion on magnitude bottleneck only; single-variable, not tested |

## Status Labels

Use one of the following:

- proposed
- implemented
- running
- finished
- failed
- rejected
- kept

## Notes

If an experiment is based on another experiment rather than the stable baseline, mark it clearly as a combined experiment in the Result Summary.
