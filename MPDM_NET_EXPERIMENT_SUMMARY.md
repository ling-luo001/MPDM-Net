# MPDM-Net 实验方向与结果总结

> 汇总日期：2026-08-10
> 任务：VoiceBank-DEMAND 单通道语音增强
> 主指标：`Validation/PESQ Score`
> 本文用于梳理实验谱系和阶段性判断，不代替最终论文中的统一复现实验表。

## 1. 证据口径

本文的数值来自以下三类证据：

1. 4090 服务器现存 TensorBoard event 文件，统一读取标量 `Validation/PESQ Score`；
2. 各独立 Git 分支的模型代码、配置、设计文档和提交关系；
3. A/B/C/D 输出端章节中已经固化的实验记录。

需要遵守四个比较边界：

- mini 与 full 结果分开比较，不能把 mini PESQ 与 full PESQ 放在同一排名中；
- 不同实验的最佳 step 不同，`best PESQ` 只能用于筛选，不能替代匹配 step 曲线；
- A/B/C/D 的历史 `PESQ≈3.45` 口径没有绑定到后续大结构实验的精确基线 commit；
- 3090 当前免密 SSH 认证失效，因此 RDHI stable full、TriPrompt residual-dense 和原始 wavelet full 的当前远程状态没有在本次重新核验。

TensorBoard 指标于 2026-08-10 从以下 4090 run 目录读取；同一目录下的续训 event 由 TensorBoard `EventAccumulator` 合并，`best` 取该 tag 的最大值，`latest` 取按 event 顺序读取的最后值：

- 原始 mini：`/home/g515528/PycharmProjects/PythonProject001/MPDM-Net/exp/base_mini/logs`；
- Scheme 2：`MPDM-Net-progressive-sr/exp/progressive_sr_mini_v1/logs` 与 `MPDM-Net-progressive-full-restore/exp/progressive_sr_full_restore_mini_v1/logs`；
- Harmonic：`MPDM-Net-harmonic-generative/exp/harmonic_generative_mini_v2/logs` 与 `harmonic_generative_full_v2/logs`；
- Residual-Dense：`MPDM-Net-residual-dense/exp/residual_dense_mini_v1/logs` 与 `residual_dense_full_v2/logs`；
- TF-LCA：`MPDM-Net-multiscale-local-channel/exp/multiscale_local_channel_full_v1/logs` 与 stabilized 对应目录；
- RDHI：`MPDM-Net-restoration-demand-histogram/exp/restoration_demand_histogram_mini_v1/logs`；
- MRCC：`MPDM-Net-mrcc-mpdm-v1/exp/mrcc_mpdm_mini_v1/logs`；
- Wavelet/TriPrompt：各自独立仓库的 `exp/*_mini_v1/logs`。

历史大结构 full 使用的统一清单哈希为：

- train clean：`212b4300...bcfaf3`；train noisy：`2786390e...b02fc`；
- valid clean：`8ac24d65...691e7`；valid noisy：`87243214...b5580`。

该验证清单共 824 对，由 576 条 `clean_valid/noisy_valid` 和 248 条 `clean_test/noisy_test` 合并而成。MRCC full 为了和历史 full 曲线可比，当前也使用这份 824 清单。最终论文不能继续用其中的 test 子集进行 checkpoint 选择；应重新建立严格独立的 validation/test 协议，并对冻结 checkpoint 做统一测试集复评。

## 2. 原始 MPDM-Net 基线

后续 Wavelet 和 MRCC 的直接结构基线是 `f5a379b`。它是一个非对称、任务专门化的幅度/相位双塔，而不是已经证明的输入级幅相解耦：

- 两个塔都接收 `(noisy_mag, noisy_pha)`；
- 幅度塔较宽，以 TFMamba 和交替 FMamba/TMamba 为主；
- 相位塔约为幅度塔的一半宽度，以 TMamba 和保守相位旋转为主；
- 配置 `num_mid_pairs=3` 时，中间有六次 VSS 交互，输出前还有一次全局交互；
- 输出由 magnitude mask、单位复数相位旋转和增强复谱组成。

可复核的原始 mini 日志基线为：

- 参数量：约 `2,262,712`；
- 最佳 PESQ：`3.184735 @ 60k`；
- 继续训练到 `498k` 后为 `3.072402`，存在明显后期退化。

因此本文将 `3.1847 @ 60k` 作为后续 mini 路线的主要筛选参考。

## 3. 实验谱系

```text
原始双塔 f5a379b
├─ Wavelet 子带交互
│  └─ Wavelet residual-dense
└─ MRCC-MPDM v1

A/B 输出端主线
└─ Scheme 2：串行抑噪塔 -> 恢复塔
   ├─ 独立的全宽恢复塔容量实验
   └─ Scheme 3 / Harmonic-Generative（自身采用全宽恢复塔）
      └─ Hierarchical Residual-Dense
         ├─ TF-LCA
         │  └─ Stabilized TF-LCA
         └─ RDHI
            └─ Stabilized RDHI

独立替代路线
└─ TriPrompt-NAF
   └─ Residual-Dense TriPrompt-NAF
```

这里最重要的事实是：Residual-Dense、TF-LCA 和 RDHI 不是三个直接加在原始双塔上的独立模块，而是一条累积主线。MRCC 与 Wavelet 才是直接从原始双塔重新开始的对照路线。

## 4. Mini 实验结果

| 方向 | 直接父模型 | 最佳 PESQ | 最佳 step | 最新记录 | 相对原始 mini 峰值 | 阶段结论 |
|---|---|---:|---:|---:|---:|---|
| 原始 MPDM-Net | - | 3.184735 | 60k | 3.072402@498k | - | mini 基准 |
| Scheme 2 轻量恢复塔 | A/B 主线 | 3.129890 | 70k | 3.059485@76k | -0.054845 | 失败 |
| Scheme 2 全宽恢复塔 | Scheme 2 | 3.136170 | 46k | 3.112268@52k | -0.048565 | 扩容仍未解决问题 |
| Harmonic-Generative | Scheme 2 | 3.193263 | 52k | 3.141331@62k | +0.008528 | 弱阳性，波动较大 |
| Residual-Dense | Harmonic-Generative | 3.205222 | 112k | 3.147820@114k | +0.020487 | 阳性，收敛较慢 |
| RDHI 原版 | Residual-Dense | 3.214019 | 52k | 3.188940@76k | +0.029284 | 较好 mini 候选 |
| MRCC-MPDM v1 | 原始双塔 | **3.219019** | 96k | 3.178951@174k | **+0.034284** | 当前最佳新结构 mini 峰值 |
| Wavelet residual-dense | Wavelet | 3.183897 | 74k | 3.137563@90k | -0.000838 | 未超过基线，淘汰 |
| TriPrompt-NAF | 独立单塔 | 3.074511 | 58k | 2.998549@84k | -0.110224 | 明显失败 |

### 4.1 对 MRCC mini 的严格解释

MRCC 的最高值高于原始 mini 约 `0.0343`，但不能只看峰值：

- 原始基线在 `60k` 为 `3.184735`；
- MRCC 在相同 `60k` 只有 `3.151`；
- MRCC 到 `96k` 才达到 `3.219019`；
- `98k` 为 `3.210`，`168k` 又达到 `3.212656`，说明 `3.21` 附近并非完全孤立的单点；
- 训练结束 `174k` 回落到 `3.178951`。

因此 MRCC 应定性为“峰值弱到中等阳性、收敛较慢”，而不是已经证明显著优于基线。它获得 full 晋升的主要理由，是数值达到筛选线，同时与原始双塔的因果对照比累积主线更干净。

## 5. Full 实验结果

### 5.1 大结构主线

| 方向 | 参数量 | 最佳 PESQ | 最佳 step | 最新记录 | 状态 |
|---|---:|---:|---:|---:|---|
| Harmonic-Generative full v2 | 1,912,940 | 3.482928 | 396k | 3.393324@1028k | 已结束，峰值后验证 PESQ 回落 0.0896 |
| Residual-Dense full v2 | 1,961,130 | **3.512338** | 688k | 3.458960@1028k | 已结束，当前最高峰值 |
| TF-LCA 原版 full | 1,989,658 | 3.493809 | 402k | 3.414116@666k | 已结束，后期退化 |
| Stabilized TF-LCA full | 1,989,730 | **3.507198** | 550k | 3.416635@572k | 仍在训练，峰值后明显波动 |
| MRCC-MPDM v1 full | 2,433,789 | 尚无可比 PESQ | - | fresh run 已启动 | 2026-08-10 按统一 824 验证口径重启 |

截至本次汇总，观察到的 full 峰值排名是：

1. Residual-Dense：`3.512338 @ 688k`；
2. Stabilized TF-LCA：`3.507198 @ 550k`；
3. TF-LCA 原版：`3.493809 @ 402k`；
4. Harmonic-Generative：`3.482928 @ 396k`。

Residual-Dense 与 Stabilized TF-LCA 的峰值差只有约 `0.00514`。在没有统一随机种子复现、滚动均值和其他客观指标前，不能断言 TF-LCA 优于 Residual-Dense；当前更稳妥的结论是两者属于同一最强性能梯队。

### 5.2 A/B/C/D 输出端章节

| 组别 | 修改 | 最佳 PESQ | step | 相对 A |
|---|---|---:|---:|---:|
| A | mask-output + 有界 complex residual | 3.465936 | 366k | - |
| B | energy-gated complex residual | 3.467033 | 670k | +0.001097 |
| C | magnitude-weighted phase loss | 3.458621 | 918k | -0.007315 |
| D | complex residual regularization | 3.450929 | 732k | -0.015007 |

A 是这一组的主要有效结构，B 只有很小的正增益；C/D 是有价值的负消融。该组适合写成“输出端复谱估计与约束”的完整探索，不适合把 B 的 `+0.0011` 单独包装成重大创新。

## 6. 各方向具体做了什么

### 6.1 Scheme 2：渐进式抑噪 -> 恢复

核心变化是把原先并行、相互交互的任务专门化双塔改成串行两阶段：

1. Stage 1 通过 mask 与轻量相位旋转生成粗复谱；
2. Stage 2 同时读取 noisy complex 和 coarse complex，预测门控复残差；
3. 抑噪 bottleneck 通过零起步单向连接提供恢复上下文；
4. 容量实验把恢复塔宽度从 `0.5x` 提高到 `1.0x`。

两个版本 mini 都只有约 `3.13`。这说明问题不只是参数不足，而是串行边界丢失了抑噪塔浅层、多尺度和中间状态。该失败为后续 Residual-Dense 提供了清晰动机。

### 6.2 Scheme 3 / Harmonic-Generative

Scheme 3 与独立的 Scheme 2 全宽容量实验都从轻量 Scheme 2 的共同节点分出。Scheme 3 并不继承全宽容量实验的 commit，但在自己的实现中同样把恢复塔设为全宽，并增加：

- `[-2,-1,+1,+2]` 邻帧复数深度滤波；
- 60--500 Hz、64 候选的可微软 F0 后验；
- 谐波占据图和清浊音门控；
- harmonic/aperiodic 两个受约束复残差头；
- coarse complex、pitch、voicing、harmonic support 辅助损失。

它提出了一个可以写进论文的问题：强抑噪会损伤弱谐波，能否借助显式语音周期先验做受约束恢复。full 达到 `3.482928`，证明结构可以训练，但明显低于后续 Residual-Dense，且到 `1028k` 已回落约 `0.09`。它更适合作为机理清晰的父实验，而不是最终最佳模型。

### 6.3 Hierarchical Residual-Dense

该方向不改 Scheme 3 的深滤波、谐波先验和双复残差头，只修复跨阶段信息与梯度瓶颈：

- 六个 `ResidualDenseBridge`，覆盖恢复塔 encoder L1/L2、bottleneck、decoder L2/L1 和 output；
- 每个桥压缩并复用同尺度抑噪特征及中间状态；
- 两个塔共八个 downsample/upsample 投影捷径；
- 所有外加路径使用零起步有界标量，初始化时退化为 Scheme 3。

结果从 harmonic full 的 `3.482928` 提升到 `3.512338`，观察峰值增加约 `0.02941`。这是当前最强的观察结果，与“层级信息传递是串行抑噪-恢复的重要瓶颈”这一假设一致；但一次同时加入了六个 bridge 和八个 shortcut，仍需 bridge-only、shortcut-only 和重复种子实验确认因果归属。

但论文表述必须克制：Residual-Dense 本质上是稳定性和信息传输结构，适合作为强工程基座或关键消融，不宜只凭 bridge/skip 本身宣称全新机制。

### 6.4 TF-LCA 与 Stabilized TF-LCA

TF-LCA 从 Residual-Dense `404c982` 出发，在抑噪塔和恢复塔每个 stage-level Mamba stack 后加入一个局部通道适配器，共 12 个：

- GroupNorm 与 pointwise 投影；
- depthwise `3x3`、`7x1` 时间条带和 `1x7` 频率条带卷积；
- ECA 风格通道选择；
- 有界残差更新。

稳定版进一步加入三个局部分支间的零起步密集复用、可学习分支权重，并使初始通道增益严格为 1。它只比原版增加 72 个控制标量。

论文问题可以表述为：Mamba 擅长长程建模，但阶段输出是否仍需要局部二维时频细节与通道重标定。稳定版达到 `3.507198`，进入最强梯队，但没有超过 Residual-Dense 的观察峰值。它目前更像“有解释力、结果接近持平的局部补偿机制”，需要匹配 step 和多次复现后再决定是否作为主贡献。

### 6.5 RDHI

RDHI 也从 Residual-Dense 出发，但研究的是不同问题：

```text
d = clip(abs(noisy_mag - coarse_mag) / (noisy_mag + eps), 0, 1)
```

- 根据 restoration demand 对 bottleneck token 排序并分为八个桶；
- 先做桶内 attention，再做桶摘要之间的交互；
- 最后严格 inverse-sort 回原时频位置；
- local scale 从 `0.01` 起步，summary scale 从 0 起步。

原版 mini 达到 `3.214019 @ 52k`，在匹配阶段具有较好竞争力。它的论文问题比普通 attention 更明确：受损程度相似但空间不相邻的时频单元，是否应该共享恢复信息。

Stable RDHI full 已在历史流程中部署到 3090，但本次无法重新连接 3090，因此不把未核验的 full 数字写入排名。

### 6.6 MRCC-MPDM v1

MRCC 直接从原始双塔 `f5a379b` 出发，不继承 Scheme 2/3、谐波先验或 Residual-Dense：

- 原有双塔、七个 VSS、mask decoder、phase decoder 和三输出接口保持不变；
- 在 `128/128/32` 与 `256/256/64` 两个辅助 STFT 分辨率上，共享 proposer 估计复修正和逐 TF reliability；
- 从零波形修正开始，严格执行两次共享波形 consensus iteration；
- 每次重算完整 TF objective，仅接受目标不增大的更新；
- 两个分辨率增益零初始化，保持原始输出及其 Jacobian；
- 参数量 `2,433,789`，重复测得的最坏前反向计算比约 `1.583x`，低于设定的 `1.70x` 上限。

MRCC 的论文命题不是“再加一个 residual head”，而是：不同分析分辨率提出的复谱修正，必须在同一共享波形上满足带可靠度的跨分辨率一致性。

它的优势是与原始基线直接对照，因果关系清楚；不足是 mini 同 step 收敛较慢，full 结果尚未产生。正式论文至少需要以下消融：

1. 原始 MPDM-Net；
2. 容量匹配的 native-only proposer；
3. 两分辨率直接平均、无 consensus；
4. 完整两分辨率、两次迭代 MRCC。

### 6.7 Wavelet 子带交互

该方向直接替换原始双塔的七个 VSS 交互：

- 用可逆二维 Haar 分解得到 LL、时间细节、频率细节和联合细节；
- 每个子带采用方向特定卷积与幅相双向交换；
- phase 更新使用更保守的 ReZero 上限；
- residual-dense 版本再增加尺度捷径、子带密集适配器和 coarse-to-fine context。

优化版最高 `3.183897 @ 74k`，仍未超过原始基线 `3.184735 @ 60k`，且末期降到 `3.137563`。在当前单次 mini 口径下未形成可确认增益，因此不晋升 full；这不能普遍否证可逆子带交互，只能否证当前实现和训练设置。

### 6.8 TriPrompt-NAF

TriPrompt-NAF 是一个独立 hedge，而不是原始 MPDM-Net 的小改：

- 完全移除 Mamba，改为单塔 activation-free Axis-NAF；
- 估计 utterance/time/frequency 三粒度 degradation prompt；
- 多尺度 FiLM 调制 decoder；
- 直接预测有界 complex residual；
- residual-dense 版本增加跨 stage、尺度、skip 和输出密集通路。

原版最高仅 `3.074511`，明显低于原始 mini。Residual-dense 版本在首个 `2k` 点从 `2.4258` 提升到 `2.5488`，后续根据既有实验决策未晋升，但本次能够访问的 4090 event 中没有它的完整最终曲线。可以确认的是，该路线大幅更换 backbone，削弱了与原 MPDM-Net 的继承关系，而现有证据没有建立优于基线的优势。

### 6.9 更早期的小模块与替换实验

历史日志还包括 phase-CFFN、Eq-Mamba/GRE phase、CrossMamba、ASSM bottleneck、Replace-All-Mamba 和 asymmetric variants。可核实的若干 mini 峰值为：

- phase-CFFN：`3.170122`；
- Eq-Mamba v1：`3.163110`；
- Eq-Mamba v2：`2.885834`；
- Eq-Mamba v3：约 `2.76--2.82`；
- CrossMamba mini：`3.046065`。

这些结果总体没有超过 `3.184735` mini 基线，支持早期总结中的判断：激进相位分支修改和简单 Mamba 替换风险较高，不应继续拆成大量细碎论文贡献。

## 7. 当前最值得重点写的几个方向

### 7.1 性能最强：Residual-Dense 主线

从原始思想到最终模型的累积变化是：

```text
任务专门化并行双塔
-> 串行粗抑噪/复谱恢复
-> 邻帧复滤波 + 软 F0/谐波与非周期双残差
-> 六个跨阶段层级密集桥 + 八个尺度变换捷径
```

它给出了目前最高 full PESQ `3.512338`。论文中应把重点放在“为什么粗抑噪后的恢复需要持续访问多尺度抑噪状态”，并用 Scheme 2、Scheme 3、Residual-Dense 三段实验形成完整递进，而不是只说“加残差连接提高了效果”。

### 7.2 结果接近最强：Stabilized TF-LCA

它在最强 Residual-Dense 基座上补充 Mamba 缺少的局部二维细节和通道选择，达到 `3.507198`。适合写成局部-长程互补机制，但当前峰值没有超过父模型，因此必须保留无 TF-LCA 的 Residual-Dense 对照，避免把累计结果全部归因于 adapter。

### 7.3 对原始基线最干净：MRCC-MPDM v1

MRCC mini 峰值 `3.219019`，是直接从原始双塔出发的最好新路线之一。它不依赖谐波生成主线，核心机制与论文问题清楚。是否成为论文主贡献，应由同一 824 清单上的原始 full 基线、native-only/no-consensus 消融、匹配 step、重复种子、其他客观指标和计算代价共同决定，不能只用是否达到 `3.50` 判断。

### 7.4 有独立问题定义：RDHI

RDHI mini `3.214019 @ 52k`，收敛阶段比 MRCC 更早。它按 restoration demand 重排交互，问题定义清楚，但依赖 Residual-Dense/Scheme 3 父模型，最终价值取决于 3090 full 的可核验结果。

## 8. 建议的论文组织

一种相对完整且不过度碎片化的组织方式是：

1. **输出端谱估计章节**：A/B/C/D，A 为有效结构，B 为轻微正向，C/D 为负消融；
2. **渐进抑噪与结构化恢复章节**：Scheme 2 -> Harmonic-Generative -> Residual-Dense，完整解释从失败到最强 full 的过程；
3. **局部与长程互补章节**：Residual-Dense -> TF-LCA，重点验证 Mamba 后的局部二维时频/通道补偿；
4. **跨分辨率一致性章节或候选主贡献**：原始 MPDM-Net -> MRCC，保持一条与谐波主线正交、因果关系干净的路线。

RDHI 可在 full 结果足够强时作为第三或第四章候选；Wavelet、TriPrompt 和早期 phase/Mamba 替换更适合放在探索总结或附录，不建议分别扩写成独立主章节。

论文措辞还应保持准确：当前模型只能称为“幅度/相位任务专门化的非对称双塔”，不能称为已经实现了输入级幅相解耦。

## 9. 2026-08-10 16:54 CST 部署观察快照

### 9.1 Stabilized TF-LCA full

- 服务器：RTX 4090；
- 进程：`PID 4242`；
- 当前观察最高：`3.507198 @ 550k`；
- `572k` 为 `3.416635`，峰值后波动明显；
- 日志目录：`/home/g515528/PycharmProjects/MPDM-Net-multiscale-local-channel-stabilized/exp/multiscale_local_channel_stabilized_full_v1/logs`；
- 截至快照仍在运行。

### 9.2 MRCC-MPDM v1 full

- 分支：`codex/exp-mrcc-mpdm-v1`；
- full 配置提交：`d9c38c9`；
- 配置：`recipes/Mamba-SEUNet/MRCC-MPDM-v1-full.yaml`；
- 训练集：11,572 对，验证集：824 对，缺失文件 0，训练/验证文件名重叠 0；
- 四个清单与 Stabilized TF-LCA full 的 SHA256 完全一致；
- 服务器：RTX 4090；
- 进程：`PID 51004`；
- fresh run 已完成 `step 600`；最新记录为 generator loss `2.059`、discriminator loss `0.005`，未出现 NaN/Inf/OOM；
- 首个统一 824 对验证口径的 PESQ 将在 `step 2k` 产生，当前不提前写入结果；
- 日志：`/home/g515528/PycharmProjects/MPDM-Net-mrcc-mpdm-v1/mrcc_mpdm_full_v1.log`；
- 先前 576 验证口径的 run 已归档为 `mrcc_mpdm_full_v1_invalid_val576`，其 PESQ 不进入任何排名。

远程直接启动或成对检查点恢复命令：

```bash
cd /home/g515528/PycharmProjects/MPDM-Net-mrcc-mpdm-v1
CUDA_VISIBLE_DEVICES=0 /home/g515528/software/anaconda3/envs/mambavision/bin/python -u train.py
```

## 10. 阶段性结论

1. 单纯增加恢复塔、替换 Mamba 或激进修改相位分支，普遍没有超过原始 mini；
2. 显式谐波恢复有较好的论文解释，但单独 full 结果并非最强；
3. Residual-Dense 对 Scheme 3 的提升最明确，当前保持 full 最高峰值；
4. TF-LCA 进入最强梯队，但暂未证明优于其 Residual-Dense 父模型；
5. RDHI 与 MRCC 是当前问题定义较清楚的两条候选，其中 MRCC 的原始基线对照最干净；
6. MRCC full 的结果将决定下一阶段是否围绕跨分辨率一致性开展完整消融和论文主叙事。
