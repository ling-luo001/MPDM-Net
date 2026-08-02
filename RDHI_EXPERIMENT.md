# RDHI Mini Experiment

## Question

Does grouping restoration bottleneck features by the amount removed by Stage 1
improve recovery of similarly suppressed time-frequency regions?

This branch starts from residual-dense commit `404c982`. RDHI depends on its
coarse complex spectrum, so it is not an independent addition to the original
parallel dual-tower baseline.

## Mechanism

For noisy magnitude `X` and Stage-1 coarse magnitude `S0`, the restoration
demand is

```text
d = clip(abs(X - S0) / (X + eps), 0, 1)
```

The demand map is average-pooled to the restoration bottleneck and detached for
sorting. Bottleneck tokens are sorted by demand, divided into eight equal-size
buckets, mixed within each bucket, summarized across buckets, and restored to
their original time-frequency positions. A bounded residual scale starts at
`0.05`.

RDHI is inserted once, after `dense_bridges['middle']` and before the restoration
TMamba/FMamba blocks. Losses, Mamba depth, output heads, harmonic analysis, and
data selection are unchanged.

## Interpretation Boundary

The histogram interaction is inspired by Histoformer. The experiment tests a
speech-specific restoration-demand topology; it does not establish that
histogram attention or time-frequency attention is new by itself.

## Mini Gate

- At 2k: stop for non-finite values, broken gradients, or resource failure.
- At 20k: stop if the latest three PESQ values trail the matched baseline by
  more than 0.05 on average.
- At 40k: stop if best PESQ trails by more than 0.03 with no upward trend.
- At 60k: continue to full data only if best PESQ improves by at least 0.02 and
  the latest-three mean improves by at least 0.01.

## Launch

```bash
CUDA_VISIBLE_DEVICES=0 /home/g515528/software/anaconda3/envs/mambavision/bin/python \
  -u train.py \
  --exp_name restoration_demand_histogram_mini_v1 \
  --config recipes/Mamba-SEUNet/Mamba-SEUNet.yaml \
  --mini \
  --epochs 200
```
