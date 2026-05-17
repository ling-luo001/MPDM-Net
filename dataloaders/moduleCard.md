Module Card: Pha-FFN Pre-Decoder Adapter
1. 来源

Paper: Global Rotation Equivariant Phase Modeling for Speech Enhancement with Deep Magnitude-Phase Interaction
Year: 未确认，待后续从论文首页 / arXiv / GitHub 核实
Code: RENet / 论文代码中的 TransformerBlock -> ComplexFFN / Pha-FFN
Task: Speech Enhancement / Phase Modeling / Magnitude-Phase Dual-Stream Enhancement

2. 这个模块解决什么问题？

这个模块主要解决：

相位分支在最终 phase correction 前缺少专门的复值相位细化；
普通实值卷积 / 普通激活对 phase feature 的几何结构不友好；
phase decoder 直接从 pha_fused 输出旋转向量，可能缺少对相位局部结构的进一步 refinement。

该模块不是为了替代 Mamba，也不是为了完整实现 GRE，而是作为：

GRE-inspired complex phase refinement before phase decoder

用于增强 phase decoder 前的相位表达质量。

3. 核心结构

输入：

pha_fused: [B, C, T, F]

其中 C = pha_dim[0]，当前实验中为 8。

输出：

pha_fused_refined: [B, C, T, F]

保持与原 PhaseDecoder 输入完全一致。

主要操作：

使用 1x1 Conv2d 将实值 phase feature 投影为复值 pair：
[B, C, T, F] -> [B, 2C, T, F]
拆分为：
x_r, x_i
reshape 为序列形式：
[B, C, T, F] -> [B, T*F, C]
使用：
ComplexRMSNorm
ComplexFFN
ComplexRMSNorm
做 complex residual：
x_r = x_r + ffn_r
x_i = x_i + ffn_i
reshape 回 2D feature；
拼接 real / imag 后用 1x1 Conv2d 投影回实值 feature；
使用残差缩放：
out = x + res_scale * y

默认：

pha_ffn_res_scale: 0.1

是否需要改变主干网络：

不需要。

该模块不改变：

Magnitude branch；
Phase branch 主体 Mamba；
Cross fusion；
PhaseDecoder 输入输出；
loss；
training pipeline。
4. 能插到我现有模型哪里？

当前实验插入位置：

Decoder 前：是
Phase branch：是
Phase decoder 前：是，当前采用位置

具体位置：

global fusion
-> pha_fused
-> PhaFFNAdapter2D
-> PhaseDecoder
-> rot_vec
-> delta_cos / delta_sin

其他潜在位置：

Encoder 后：可以，但不是第一优先级；
Mamba block 内部：暂不适合，后续 Eq-Mamba 单独研究；
Magnitude branch：不适合；
Cross-branch fusion：可以作为后续 GRE gate 方向；
Loss 层：不涉及。
5. 预计收益

主要预期收益：

改善 phase correction 输出前的相位特征；
降低 phase loss；
降低 complex loss；
可能提升 PESQ、CSIG、COVL；
对 STOI、SI-SDR 的提升不一定明显；
参数和计算量增加较小；
不破坏原 baseline 主体结构。

该模块更可能提升：

phase loss / complex loss / PESQ / COVL

而不是主要提升：

SI-SDR / STOI
6. 改代码难度

难度：2 / 5

原因：

不需要改 Mamba；
不需要改 selective scan；
不需要改训练流程；
只需要复用已有 ComplexRMSNorm 和 ComplexFFN；
只在 phase_decoder 前新增 adapter。

已完成 sanity check：

PhaFFNAdapter2D sanity check passed on cuda
7. 风险

主要风险：

不是严格 GRE

因为前面的 phase branch 仍然包含普通 TMamba / FMamba / fusion，所以该模块不能保证整条相位分支全局旋转等变。

可能收益有限

它只是在 phase decoder 前做单点 refinement，不改变主干建模能力。

可能扰动 phase decoder 输入分布

如果 res_scale 过大，可能导致 phase decoder 不稳定。

ComplexFFN 序列长度较大

输入 reshape 为 [B, T*F, C]，如果 T/F 很大，计算会增加，但当前通道数较小，风险可控。

与后续 Eq-Mamba 实验存在变量混淆

因此 Eq-Mamba v1 应从原始 baseline 分支开始，不应直接叠加该模块。

8. 实验编号

EXP-001

实验名称：

EXP-001: Baseline + PhaFFN Pre-PhaseDecoder

建议配置：

use_pha_pre_decoder_ffn: true
pha_ffn_dropout: 0.0
pha_ffn_res_scale: 0.1

对照实验：

use_pha_pre_decoder_ffn: false

建议记录指标：

PESQ
STOI
SI-SDR
CSIG
CBAK
COVL
phase loss
complex loss
time-domain loss
9. 最终结论

当前结论：待复查

暂时判断：

保留进入实验队列，但不能作为严格 GRE 模块。

当前模块定位：

低风险、低侵入式的 phase decoder 前复值相位细化模块。

后续判断标准：

如果 phase loss / complex loss 降低，并且 PESQ / COVL 有提升：保留；
如果 loss 稳定但指标无提升：作为备选模块；
如果训练不稳定或 PESQ / COVL 下降：降低 res_scale 后复查；
如果多次实验无收益：淘汰。

最终状态：

EXP-001 状态：已实现，sanity check 已通过，等待完整训练与指标验证。