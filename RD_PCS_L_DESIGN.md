# RD-PCS-L Gate 0 design

RD-PCS-L keeps the `148a4fe` Residual-Dense/Scheme3 data flow and restoration mechanism unchanged. Its two controlled changes are the PCS400 clean-speech training target and a configuration-only capacity gradient: `hid_feature=24`, `num_tfmamba=3`, `num_mid_pairs=3`, and `restoration_width_ratio=1.0`.

The dedicated recipe is `recipes/RD-PCS-L/RD-PCS-L.yaml`. Its future output namespace is `exp/rd_pcs_l_h24_t3_m3_pcs`, with logs under that directory. A future mini-data gate may use:

```powershell
python train.py --config recipes/RD-PCS-L/RD-PCS-L.yaml --exp_name rd_pcs_l_h24_t3_m3_pcs --mini
```

Do not pass any resume option and start only with a fresh output namespace. Gate 0 does not run this command, start training, or read weights.

Run local Gate 0 from this worktree with the project environment:

```powershell
E:\anaconda3\envs\Mamba1\python.exe gate0_rd_pcs_l.py
```

The script checks YAML parsing and route values, PCS silence/tiny/ordinary-wave behavior, train-only PCS routing, and a small finite forward/backward. It constructs H16/N2, an in-memory-only H20/N3 midpoint, and H24/N3, then requires their parameter counts to increase strictly in that order; only H24/N3 has a recipe. The script reports actual forward/backward timings. If `selective_scan_cuda` is unavailable, it prints a prominent notice and substitutes only Mamba's sequence operation with a shape-preserving differentiable stub. In that mode the forward/backward result is structural validation, not real CUDA Mamba runtime validation.
