# main-codex-phase 分支说明

## 目标

本分支用于实验相位流的圆周几何建模。原始模型在 `generator.py` 的前向入口中将 `noisy_mag` 和裸角度 `noisy_pha` 直接拼接后送入幅度塔和相位塔：

```python
mag_in = [mag, phase]
pha_in = [mag, phase]
```

裸相位在 `-pi/pi` 处不连续，但卷积、归一化、Mamba/ASSM 等模块会把它当作普通欧氏连续值处理，这会让相位塔在边界附近学习到错误距离。该分支的核心实验是让相位塔接收复平面表示，避免直接建模裸角度断点。

## 当前最终方案

修改位置：`models/generator.py`

当前方案保持幅度塔输入不变，只修改相位塔输入：

```python
noisy_cos = cos(noisy_pha)
noisy_sin = sin(noisy_pha)

mag_in = [noisy_mag, noisy_pha]
pha_in = [noisy_mag * noisy_cos, noisy_mag * noisy_sin]
```

也就是相位塔输入从原来的 `[mag, raw_phase]` 改为 noisy complex 的 RI 表示：

```python
pha_in = [real, imag]
```

这样做的理由：

- 保留圆周拓扑：相位方向由单位圆上的 `cos/sin` 表示，不再暴露 `-pi/pi` 跳变。
- 保留幅度置信度：`mag*cos` 和 `mag*sin` 会让高能量时频点对相位塔更有意义，低能量时频点自然被压低。
- 参数量不增加：相位塔输入仍然是 2 通道，`pha_cfg['model_cfg']['input_channel']` 不需要从 2 改到 3。
- 比单纯 `[cos, sin]` 信息更完整：`[cos, sin]` 只有方向，没有局部幅度/SNR 线索。

## 输出端相位建模

输出端继续沿用单位复数旋转的思路：

```python
rot_vec = phase_decoder(pha_fused)
rot_vec = normalize(rot_vec)
delta_cos, delta_sin = chunk(rot_vec)
```

然后用预测的旋转量作用在 noisy phase 的单位向量上：

```python
pred_cos = noisy_cos * delta_cos - noisy_sin * delta_sin
pred_sin = noisy_sin * delta_cos + noisy_cos * delta_sin
pred_pha = atan2(pred_sin, pred_cos)
```

因此模型不是直接回归裸相位，而是预测相对 noisy phase 的单位复数旋转。`pred_pha` 仍然返回给现有训练、推理和 iSTFT 流程，以保持外部接口不变。

## 和原始输入的区别

原始相位塔：

```python
pha_in = [mag, raw_phase]
```

当前相位塔：

```python
pha_in = [mag*cos(raw_phase), mag*sin(raw_phase)]
```

关键变化是相位塔内部不再直接看到裸角度。它看到的是复平面坐标，卷积和归一化在这个表示上处理的是连续的实部/虚部信号。

## 参数量和兼容性

当前方案不增加输入通道数：

```python
mag_cfg['model_cfg']['input_channel'] = 2
pha_cfg['model_cfg']['input_channel'] = 2
```

因此第一层卷积形状不变，参数量相对 `[mag, cos, sin]` 三通道方案更稳定。

但需要注意：虽然权重形状不变，输入语义已经改变。旧 checkpoint 可以在形状上加载，但相位塔第一层原来学习的是 `[mag, raw_phase]`，现在输入变成 `[real, imag]`。如果继续使用旧 checkpoint，建议至少重新训练或使用较小学习率 warmup；更干净的实验是从头训练。

## 相关分支改动

除相位流修改外，当前分支相对 `main` 还包含：

- `train.py` 默认实验名改为 `main_codex_phase_mini`。
- `recipes/Mamba-SEUNet/Mamba-SEUNet.yaml` 中加入了相位圆周建模实验说明，并调整了 `num_workers`。
- 分支中删除了一批早期任务说明、prompt 和实验记录类 Markdown 文件；这些属于分支整理，不是相位建模核心逻辑。

## 验证状态

已在 `mamba` conda 环境下执行：

```bash
python -m py_compile models/generator.py
```

语法检查通过。

当前机器环境无法完整执行 `python test.py`，因为模型初始化路径中 `models/cross.py` 存在硬编码 `.cuda()`，在无 CUDA GPU 环境下会报：

```text
RuntimeError: No CUDA GPUs are available
```

因此完整前向和训练验证需要在可用 CUDA 环境中执行。

## 建议实验对照

建议至少跑三组 ablation：

1. 原始输入：`pha_in = [mag, raw_phase]`
2. 纯圆周输入：`pha_in = [cos(raw_phase), sin(raw_phase)]`
3. 当前方案：`pha_in = [mag*cos(raw_phase), mag*sin(raw_phase)]`

重点比较 PESQ、STOI、SI-SDR、phase loss、收敛速度和训练稳定性。当前方案预期在不增加参数量的前提下，比原始裸相位输入更符合相位的圆周几何，同时比纯 `[cos, sin]` 保留更多幅度置信度信息。
