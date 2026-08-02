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
sorting. Bottleneck tokens are sorted by demand and divided into eight
equal-size buckets. Pre-normalized raw tokens first receive a bounded bucket
attention residual. Bucket summaries are then computed from the resulting
local tokens and receive a separate cross-bucket attention residual. The two
updates are projected by an identity-initialized linear layer, added to the raw
sorted tokens, and strictly inverse-sorted to their original time-frequency
positions.

The local residual scale starts at `0.01`, close to the scale learned by the
original mini run. The cross-bin summary scale starts at zero, so it receives a
gradient immediately without injecting a random summary update on the first
step. This separation lets the model accept local interaction while rejecting
cross-bin interaction, instead of suppressing both through one shared scale.

RDHI is inserted once, after `dense_bridges['middle']` and before the restoration
TMamba/FMamba blocks. Losses, Mamba depth, output heads, harmonic analysis, and
data selection are unchanged.

## Stability Diagnostics

Training logs report the detached local and summary scales, each projected
update's RMS ratio relative to its input, mean demand span inside a bin, mean
restoration demand, and padding utilization. `rdhi_scale` remains available as
a compatibility alias for the local scale.

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
  --exp_name restoration_demand_histogram_stabilized_mini_v1 \
  --config recipes/Mamba-SEUNet/Mamba-SEUNet.yaml \
  --mini \
  --epochs 200
```
