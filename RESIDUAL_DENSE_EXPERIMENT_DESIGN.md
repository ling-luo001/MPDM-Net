# 层级残差密集融合实验

## 1. 三个已有实验的共同问题

方案二轻量版、方案二全宽版和方案三虽然输出头不同，但都建立在同一条串行主路径上：抑噪塔先形成粗复谱，恢复塔主要读取原始复谱、粗复谱和最终 bottleneck 上下文。

已有结构并不缺少局部残差：Mamba block 内部有残差，encoder/decoder/refinement 外部有残差，DenseEncoder 内部也有四层密集连接。真正缺失的是跨越“抑噪 -> 恢复”边界的层级信息流：

- 恢复塔看不到抑噪塔浅层保存的高分辨率细节；
- 抑噪 encoder 与 decoder 的同尺度互补状态没有传给恢复塔；
- bottleneck 只传最后状态，FMamba/TMamba 交替过程中的中间状态被丢弃；
- Downsample/Upsample 改变分辨率和通道数，没有可学习的投影捷径。

全宽方案二后期优于轻量版，说明容量有帮助，但仅扩大通道不能修复上述信息瓶颈。方案三的相位损失改善而 PESQ 波动较大，也说明新增先验能够学习，但恢复阶段仍缺少稳定的多尺度观测锚点。

## 2. 实验假设

> 如果性能损失主要来自串行阶段的信息与梯度瓶颈，那么在不改变方案三目标函数和生成头的前提下，引入零起步的跨阶段层级残差密集桥，应当提高收敛稳定性和验证 PESQ。

本实验不检验“增加更多参数是否有效”，而检验“恢复塔能否持续访问抑噪塔的多尺度状态”。

## 3. ResidualDenseBridge

每个桥接收恢复塔当前状态 `R` 和若干同分辨率抑噪状态 `S_i`：

```text
U = Project(Norm(Concat(R, S_1, ..., S_n)))
R_out = R + tanh(alpha) * U
```

`Project` 由 `1x1` 压缩、深度可分离 `3x3` 局部建模和 `1x1` 输出组成。隐藏宽度为目标通道数的 0.5 倍。

标量 `alpha` 初始化为 0，因此新模型在初始化时严格退化为方案三。投影权重采用正常初始化，使 `alpha` 首步即可获得梯度；没有同时把投影和标量都置零造成死亡分支。

## 4. 六个跨阶段桥

| 位置 | 恢复状态 | 密集抑噪上下文 |
|---|---|---|
| Encoder L1 | `restore_x1` | encoder L1、decoder L1、refinement output |
| Encoder L2 | `restore_x2` | encoder L2、decoder L2 |
| Bottleneck | `restore_x3` | 初始 middle 状态及每次 FMamba/TMamba 交替结果 |
| Decoder L2 | `restore_y2` | encoder L2、decoder L2 |
| Decoder L1 | `restore_y1` | encoder L1、decoder L1、refinement output |
| Output | `restore_final` | suppression final feature |

Encoder 桥让恢复塔从开始就获得抑噪层级信息，decoder 桥避免这些信息在恢复塔内部再次衰减，output 桥为两个复残差生成头提供直接的抑噪特征锚点。

## 5. 分辨率变换残差

两个塔的所有 Downsample/Upsample 增加形状匹配的投影捷径：

```text
Y = MainTransform(X) + tanh(beta) * ProjectionShortcut(X)
```

八个 `beta` 均初始化为 0，保留原始尺度变换，同时允许训练后形成更直接的梯度路径。

## 6. 受控变量

以下内容完全继承方案三：

- 强抑噪复数深度滤波；
- 软 F0、清浊音和谐波先验；
- 谐波/非周期双复残差头；
- 所有主损失和辅助损失权重；
- 恢复塔宽度、训练数据、学习率和随机种子。

因此与方案三的差异只包含层级残差密集桥和尺度变换捷径。

## 7. 判据

- 新桥和尺度捷径的平均绝对 scale 应从 0 逐渐增大，否则说明模型不需要新增通路；
- 清浊音、深度滤波和生成残差活动不能因新桥而坍缩；
- 重点比较 12k、20k、30k、40k 和 60k 的 PESQ，而不是单个最高波动点；
- 同时比较最近五次验证均值、Magnitude Loss 和 Phase Loss；
- 若只增加参数却不能超过方案三的滚动均值，则跨阶段信息瓶颈假设不成立。

## 8. 风险

密集桥可能重复传递噪声或让恢复塔绕过粗谱。零起步 scale、粗复谱监督和谐波支持损失用于限制这一风险。若桥接 scale 快速增大但验证性能恶化，应优先判断为上下文污染，而不是继续扩大桥宽。
