# main-codex-phase 分支说明

## 目标

本分支围绕相位和复数谱输出做两个实验性改造：

1. 相位塔输入从裸相位改为复平面 RI 表示，减少 `-pi/pi` 断点带来的学习问题。
2. 输出端从单纯 `magnitude mask + phase rotation` 升级为 `base complex + bounded complex residual`，让模型能在 RI 平面做小幅精修。

## 相位塔输入

原始相位塔输入是：

```python
pha_in = [noisy_mag, noisy_pha]
```

当前分支改为 noisy complex RI：

```python
noisy_cos = cos(noisy_pha)
noisy_sin = sin(noisy_pha)
pha_in = [noisy_mag * noisy_cos, noisy_mag * noisy_sin]
```

也就是：

```python
pha_in = [noisy_real, noisy_imag]
```

幅度塔仍保持原输入：

```python
mag_in = [noisy_mag, noisy_pha]
```

这样相位塔不再直接处理裸角度，而是在连续的复平面坐标上建模；同时 `mag*cos/sin` 保留了幅度置信度，比单纯 `[cos, sin]` 信息更完整。

## 输出端结构

原始输出路径是：

```python
denoised_mag = mag_mask * noisy_mag
pred_phase = rotate(noisy_phase, delta_cos, delta_sin)
denoised_complex = denoised_mag * [cos(pred_phase), sin(pred_phase)]
```

当前分支保留这条稳定路径作为 base complex：

```python
base_real = denoised_mag * pred_cos
base_imag = denoised_mag * pred_sin
```

然后新增轻量 RI residual head：

```python
res_real, res_imag = residual_head(pha_fused)
residual = tanh(residual) * complex_residual_scale * noisy_mag
```

最终输出：

```python
enh_real = base_real + res_real
enh_imag = base_imag + res_imag

denoised_mag = sqrt(enh_real**2 + enh_imag**2)
pred_pha = atan2(enh_imag, enh_real)
denoised_com = [enh_real, enh_imag]
```

因此模型仍然保留 MP-SENet 风格的稳定 mask 和 phase rotation 主路径，但额外获得 complex-domain 精修能力。

## Residual Head 设计

`complex_residual_decoder` 是轻量 head，不复用完整 DenseBlock：

```python
Conv2d -> PixelShuffle -> depthwise Conv2d -> InstanceNorm2d -> PReLU -> Conv2d(2)
```

最后一层卷积被零初始化：

```python
nn.init.zeros_(self.complex_residual_decoder[-1].weight)
nn.init.zeros_(self.complex_residual_decoder[-1].bias)
```

这样训练开始时 residual 约为 0，模型初始行为接近原来的 `mag_mask + phase_rotation`，不会一开始就破坏已有稳定输出。

残差幅度默认受限于：

```python
complex_residual_scale = 0.1
residual = tanh(raw_residual) * 0.1 * noisy_mag
```

可以在 `model_cfg` 中配置 `complex_residual_scale`。建议初始实验使用 `0.1`，后续再比较 `0.05 / 0.1 / 0.2`。

## 训练影响

外部接口保持不变：

```python
return denoised_mag, pred_pha, denoised_com
```

但三者现在都来自最终增强后的 RI 复数谱，而不是只来自 mask 和 phase rotation。现有 magnitude loss、phase loss、complex loss、time loss 可以继续使用。

需要注意：

- 当前修改不兼容旧 checkpoint 的语义，即使部分权重形状能加载，也建议从头训练。
- residual head 是新增参数，旧 checkpoint strict 加载会缺少这些权重。
- 如果 residual 过强，可能绕过 mask/phase 主路径；因此保留了 `tanh` 和 `complex_residual_scale` 约束。

## 验证状态

已在 `mamba` conda 环境执行：

```bash
python -m py_compile models/generator.py
```

语法检查通过。

当前机器没有可用 CUDA GPU，完整 `python test.py` 会在项目已有的 `models/cross.py` 硬编码 `.cuda()` 处失败：

```text
RuntimeError: No CUDA GPUs are available
```

完整前向、训练和指标评估需要在可用 CUDA 环境中执行。

## 建议 Ablation

建议至少比较：

1. 原始 baseline：`pha_in = [mag, raw_phase]`，无 complex residual。
2. RI input only：`pha_in = [mag*cos, mag*sin]`，无 complex residual。
3. 当前方案：RI input + bounded complex residual。
4. residual scale 对比：`0.05 / 0.1 / 0.2`。

重点观察 PESQ、STOI、SI-SDR、complex loss、phase loss、训练稳定性，以及 residual 是否在验证集上带来真实收益。
