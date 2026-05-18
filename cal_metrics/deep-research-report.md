# 面向 Mamba 双流幅相语音增强基线的可落地增益调研报告

## Executive Summary

你的 baseline 已经具备相当强的“论文完成度”：双流幅相分支、U-Net 式多尺度主干、时间/频率串行 Mamba、bottleneck 多次幅相融合、末端全局幅相融合，这意味着下一步最值得做的，不是重写整网，而是给现有骨干补上**更强的跨分支交互、显式频带建模、局部时频细节建模，以及更贴合相位本质的建模/损失**。下文的瓶颈分析都以你在本会话里给出的 baseline 描述为依据。fileciteturn0file0

最值得优先尝试的方向，我建议压缩成下面这几条：

- **首选方向是把 MambAttention 的“共享时间-频率多头注意力”作为可插拔模块嫁接到你现有 TFMambaBlock 或 bottleneck**。它的价值不在于“再加一层 attention”，而在于它专门针对单通道语音增强中的**跨语料泛化**与**时频联合选择性**设计，而且作者还证明了同样的共享时频注意力对 LSTM/xLSTM 也有效，这说明它本身就是一个可迁移的增强插件，而不只是整网思路。citeturn25search3turn39search4turn39search1

- **第二个高概率增益点是 RENet 的“深度幅相交互 + 全局旋转等变相位建模”**。你的 baseline 已经有多次 magnitude-phase fusion，但从公开新工作看，单纯拼接/相加/串行融合并不足以真正建模相位的圆周结构；RENet 直接把相位看作具有旋转等变性质的对象，并引入深层幅相交互，这正好对准你现在最可能的结构短板。citeturn29search15turn29search6turn29search0

- **第三个应优先补上的，是显式频带/子带建模**。你当前虽然有 FMamba，但“沿频率维做序列建模”并不等价于“显式地按语音频带结构建模”。BSRNN 的 band-split 思路、HDF-Net 的 sub-band + deep filtering，以及 DMHRN 的 harmonic restoration，都指向同一件事：**频带层级与谐波结构需要被显式建模，而不是只交给统一的频率序列块去学**。citeturn16search16turn16search11turn17search3

- **跨领域最值得借鉴的不是“纯 2D Mamba 全盘替换”，而是 MambaIR 的 local enhancement + channel attention**。它解决的是 Mamba 在低层视觉恢复里对局部相似性、细粒度纹理和通道冗余处理不足的问题；把这套思路迁移到语谱图，天然对应你的**谐波条纹、共振峰带、瞬态辅音边缘**。citeturn11search15turn9search3turn8search1

- **RWSA-MambaUNet 的 resolution-wise shared attention 很适合你现在的 skip connection 与跨尺度融合**。你现在已经有 encoder-decoder、多尺度上下采样、skip 和 refinement；RWSA 的价值在于让“同一分辨率上的时间/频率注意力”共享，减少尺度间注意力漂移，对 cross-corpus generalization 也有明确收益。citeturn25search1turn25search2turn25search5

- **如果你只想做低改造成本实验，先不要把 2DMamba/原生 2D selective scan 放在第一轮**。这类方法很强，但通常依赖新的扫描算子、额外 kernel 或显著的数据排布改写，更适合第二轮。第一轮更建议先用“共享时频注意力、带分裂、局部增强、深度幅相交互”这些改动成本更可控的模块。citeturn13search3turn13search7turn11search4

- **相位方向不要只靠“拓宽 phase branch 通道数”解决，更应该同时改表示和损失**。Global phase bias-aware loss 说明“精确绝对相位”未必是合理训练目标，而 GRE 相位建模进一步说明相位不适合被简单当作平坦欧氏变量回归。citeturn27search0turn29search15

- **训练目标层面，SQA/metric-guided supervision 可以做第二阶段微调，而不是第一阶段主线**。Uni-VERSA-Ext 这类多指标语音质量评估模型，确实能把 SE 训练往更多主观/客观指标上拉，但如果一开始就上，增益来源会不易归因；更适合作为“结构实验筛完后的第二阶段精调”。citeturn20search6turn19search4turn19search0

## Baseline 瓶颈与检索映射

你的 baseline 核心是：**noisy magnitude + noisy phase 双输入、双流幅相分支、U-Net/encoder-decoder、多尺度下采样与上采样、TMamba/FMamba/TFMamba 串行建模、多次 cross-branch fusion、最终幅相联合输出**。这类设计已经覆盖了“幅相分离 + 多尺度 + 时频建模”三条主线，但也因此更容易在几个地方形成典型瓶颈。fileciteturn0file0

| Baseline bottleneck | Search direction | Representative papers | Why relevant |
|---|---|---|---|
| 幅度分支和相位分支早期输入相同，早期解耦可能不足 | 显式幅相分离后再交互；并行幅相预测；深度幅相交互 | MP-SENet、BSP-MPNet、RENet citeturn5search1turn26search19turn30search1turn31search1turn29search15turn29search0 | 这些方法都不是“先混后分”而是更强调**幅相各司其职、再做条件交互**，尤其适合你这种已经有双流、但可进一步强化“分而治之 + 定向交互”的基线。 |
| 只有 FMamba 沿频率维建模，但没有显式 band-wise / harmonic-aware 结构 | band-split、sub-band、harmonic restoration、frequency-specific block | BSRNN、HDF-Net、DMHRN、SEMamba++ citeturn16search16turn16search11turn17search3turn37search8turn37search5 | 频率序列建模不等于显式频带建模。近年的结果反复说明：**子带、邻域 TF bin、谐波恢复**往往能补统一骨干学不到的语音先验。 |
| TFMamba 是时间后频率的串行建模，2D 上下文与多方向扫描可能不足 | 2D selective scan、multi-directional scan、local-global modeling | VMamba、2DMamba、MambaIR、H-vmunet citeturn11search4turn13search3turn13search7turn11search15turn9search3turn36search0turn36search2 | 这些工作共同指向一点：**串行 1D scan 在 2D 数据上会丢掉局部连续性与方向性**。对语谱图而言，这对应谐波条纹、共振峰走向、瞬态边界的表达不足。 |
| cross-branch fusion 很多，但如果融合方式只是堆叠/相加/简单门控，信息利用不充分 | magnitude-guided phase、phase-guided magnitude、shared attention、channel-aware fusion | CADB-Conformer、MambAttention、RWSA-MambaUNet、RENet citeturn22search11turn21view0turn25search3turn39search4turn25search1turn25search2turn29search15 | 这些方法都在强调“**不是有 fusion 就足够，而是 fusion 必须有目标指向与共享结构**”。这与你当前多次 cross-fusion 的优化空间高度重合。 |
| 相位分支可能既“容量偏弱”，又“表示不匹配相位本质” | phase-aware head、global phase bias-aware loss、rotation-equivariant phase modeling | MP-SENet、Global Phase Bias-Aware CMGAN、RENet citeturn5search1turn27search0turn29search15 | 你的 phase branch 如果仍在普通实值空间里直接学 phase correction，很可能吃亏；近作更倾向于**显式相位头 + 适配相位几何结构的损失**。 |
| 结构与训练目标可能耦合不紧，导致 PESQ/STOI/主观质量提升不稳定 | metric-guided SE、multi-metric SQA、metric optimization | Uni-VERSA-Ext、MetricGAN-OKD、SEMamba 的 metric-oriented 设置 citeturn20search6turn19search4turn19search0turn20search0turn20search4turn37search11 | 结构改了但 loss 没跟上，常会出现 SI-SDR 提升但 PESQ/COVL 不升。训练目标是你后续第二区域最值得补的点。 |

综合来看，我认为你当前**最可能的三个一阶瓶颈**是：  
其一，**幅相分支“分得不够早、交互也不够定向”**；其二，**没有显式 band/harmonic inductive bias**；其三，**串行 T→F Mamba 对 2D 局部连续结构建模还是偏弱**。这三点恰好对应了后文我给出的第一轮实验优先级。citeturn29search15turn16search16turn11search4turn25search3

## 候选论文总表

下面这张表只保留我认为**对你当前 baseline 真正有改造价值**的条目。表中“代码链接”不直接贴原始 URL；请点击最后一列引用跳转到论文页或仓库页。

优先级解释：**S = 强烈建议首轮尝试；A = 很值得做，但可放首轮后半段；B = 第二轮；C = 更适合作为灵感或对比，不建议先做。**

| Paper | Year | Venue/arXiv | Field | Core module | Code available | Code link | Relevance to my baseline | Difficulty | Priority | Key sources |
|---|---:|---|---|---|---|---|---|---|---|---|
| MP-SENet | 2023/2025 | Interspeech / Neural Networks | 语音增强 | 并行幅度-相位估计、显式 phase decoder | 是 | 官方 GitHub | 与你的双流幅相输出最接近；适合借鉴并行解码与相位头 | 中 | S | citeturn5search1turn26search19 |
| SEMamba | 2024 | IEEE SLT / arXiv | 语音增强 | 将 Mamba 作为 SE backbone，并验证与感知/指标导向设置兼容 | 是 | 官方 GitHub | 适合作为你现有 Mamba 骨干的强对照和拆件来源 | 低-中 | A | citeturn37search0turn37search2turn37search11 |
| CADB-Conformer | 2024 | Interspeech / arXiv | 语音增强 | Channel Feature Branch + Self-Channel Attention + Band Feature Branch | 否 | — | 对“cross-branch fusion 怎么做得更有指向性”很有启发 | 中 | S | citeturn22search11turn21view0 |
| Unrestricted Global Phase Bias-Aware SE with CMGAN | 2024 | ICASSP / arXiv | 语音增强 | 允许全局 phase bias 的相位重建目标 | 否 | — | 非常适合改你的 phase loss；结构改动小 | 低 | A | citeturn27search0 |
| MambaDC | 2024 | arXiv | 语音增强 | 卷积 + Mamba 混合 backbone | 否 | — | 适合把你现有 Mamba block 改成更强 local-global hybrid | 中 | A | citeturn33search0turn33search12 |
| MambAttention | 2025 | arXiv / IEEE TASLP | 语音增强 | 共享时间-频率多头注意力 + Mamba 混合块 | 是 | 官方 GitHub | 最适合直接包裹/替换你现有 TFMambaBlock；对泛化尤其强 | 中 | S | citeturn25search3turn39search4turn39search1 |
| RWSA-MambaUNet | 2025/2026 | arXiv / ICASSP 2026 | 语音增强 | resolution-wise shared attention 的 hybrid Mamba-U-Net | 是 | 官方 GitHub | 适合改你的 skip fusion、decoder 前融合、跨尺度交互 | 中 | S | citeturn25search1turn25search2turn25search5 |
| Global Rotation Equivariant Phase Modeling with Deep MPI | 2026 | arXiv | 语音增强 | 全局旋转等变 phase modeling + deep magnitude-phase interaction | 是 | 官方 GitHub | 直接命中你的 phase branch 与 cross-fusion 问题 | 中 | S | citeturn29search15turn29search6turn29search0 |
| HDF-Net | 2025 | arXiv / Interspeech 2025 | 语音增强 | sub-band input + decoupled deep filtering + TAConv | 否 | — | 适合补“周围 TF bin 信息”和 output head 的局部修复能力 | 中 | A | citeturn16search11turn16search3 |
| High Fidelity Speech Enhancement with BSRNN | 2023 | Interspeech | 语音增强 | 显式 band-split 建模 | 有高质量第三方 | 第三方 GitHub | 最适合补你对 band-wise / frequency-aware 的建模缺口 | 中 | S | citeturn16search16turn16search4turn16search0 |
| DMHRN | 2025 | Interspeech | 语音增强 | Deep Mask + Harmonic Restoration 二阶段恢复 | 否 | — | 对提升主观质量和谐波自然度有价值，但实现细节较少 | 中 | B | citeturn17search3turn15search1 |
| BSP-MPNet | 2025 | ICME / arXiv | 语音增强 | FS-SSL 幅相分离嵌入 + 双路径 REMA 解码 + PCS | 是 | 官方 GitHub | 与双流幅相很接近，但引入 SSL 代价较高 | 中-高 | B | citeturn30search1turn31search1 |
| MambaIR | 2024 | ECCV / arXiv | 图像复原 | local enhancement + channel attention for Mamba | 是 | 官方 GitHub | 非常适合作为你每个 Mamba 块的“局部细节增强外壳” | 中 | S | citeturn11search15turn9search3turn8search1 |
| MambaIRv2 | 2024/2025 | arXiv / CVPR 2025 | 图像复原 | attentive state-space equation + semantic-guided neighboring | 是 | 官方 GitHub | 适合解决 Mamba 因因果扫描导致的上下文不对称问题 | 高 | A | citeturn9search1turn9search0 |
| VMamba | 2024 | arXiv / OpenReview | 通用视觉 Mamba | SS2D，四方向扫描 | 是 | 官方 GitHub | 是 2D selective scan 的经典来源，但工程改动较大 | 高 | B | citeturn11search4turn12search3 |
| 2DMamba | 2025 | CVPR 2025 | 通用视觉 Mamba | 原生 2D selective scan operator | 是 | 官方 GitHub | 理论上更贴合语谱图 2D 结构，但第一轮移植成本偏高 | 高 | B | citeturn13search3turn13search7turn13search11 |
| LKM-UNet | 2024 | arXiv / MICCAI 2024 | 医学图像分割 | hierarchical bidirectional large-kernel Mamba block | 是 | 官方 GitHub | 很适合迁移到语谱图中建模谐波条纹/共振峰等局部各向异性结构 | 中 | A | citeturn10search10turn10search1turn10search4 |
| H-vmunet | 2024/2025 | arXiv / Neurocomputing | 医学图像分割 | H-SS2D + Local-SS2D | 是 | 官方 GitHub | 对“减少 SS2D 冗余 + 强化局部学习”很有参考价值 | 中-高 | A | citeturn36search0turn36search2 |

如果只按“**能否在 1–2 周内做出有意义对比实验**”来排序，表里真正最像“短平快可插拔增益件”的，是 **MambAttention、RENet、BSRNN、MambaIR、CADB-Conformer、RWSA-MambaUNet**。其共同特点是：要么有公开高质量代码，要么核心模块抽象非常清晰，且都不要求你放弃现有双流 U-Net 主干。citeturn39search1turn29search0turn16search4turn8search1turn22search11turn25search5

## 最值得尝试的模块

下面是我给你的 **Top 10 模块**。这里我不再按“论文”组织，而是按“模块是否值得往你现有网络里插”来组织。

| 来源论文 | 模块名称 | 解决的问题 | 可插入 baseline 的位置 | 预计收益 | 改造难度 | 主要风险 | 第一轮实验建议 |
|---|---|---|---|---|---|---|---|
| MambAttention | 共享时间-频率多头注意力 | 纯 Mamba 容易过拟合、跨语料泛化不足 | **TFMambaBlock 内部**、**bottleneck**、**decoder 前** | 长程依赖、时频联合选择、泛化能力、PESQ/STOI | 中 | 注意力加太多会增显存且掩盖增益来源 | 先只加在 deepest encoder stage + bottleneck；不要全网铺满。citeturn25search3turn39search4turn39search1 |
| GRE 相位建模论文 | 深度幅相交互 + 全局旋转等变相位头 | 相位分支实值回归不匹配相位的圆周结构；cross-fusion 仍浅 | **phase branch**、**magnitude-phase cross fusion**、**loss/objective** | 相位恢复、幅相交互、主观质量、复杂失真下稳健性 | 中 | 需要把你现有 phase rotation 表达与新相位头对齐 | 先不改主干，只替换最后一层 phase head + 加一个深度 MPI 模块。citeturn29search15turn29search6turn29search0 |
| BSRNN | Band-split encoder / band-wise modeling | 频率维序列化 ≠ 显式频带建模 | **DenseEncoder 后**、**magnitude branch**、**FMamba 前** | 频带建模、局部谐波、共振峰和高频细节 | 中 | 子带划分策略不当会打乱 skip 对齐 | 先只在 magnitude branch 试验固定子带切分，再做 band-wise block。citeturn16search16turn16search4turn16search0 |
| CADB-Conformer | Self-Channel Attention + Channel Feature Branch | 多次 fusion 但缺少“谁引导谁”的机制 | **magnitude-phase cross fusion**、**TFMambaBlock 内部** | 幅相交互、通道重标定、局部与全局互补 | 中 | 你的通道语义与论文不同，直接套可能不稳定 | 用 CADB 的 CFB 生成 gating/query bias，而不是整块替换。citeturn22search11turn21view0 |
| RWSA-MambaUNet | Resolution-wise shared attention | skip connection 和跨尺度特征对齐不够强 | **skip connection**、**decoder 前**、**bottleneck 两侧** | 泛化能力、跨尺度一致性、语音自然度 | 中 | 若高低分辨率统计差异过大，共享权重可能欠拟合 | 先在 deepest 两个尺度上做共享 attention，再决定是否扩展全网。citeturn25search1turn25search2turn25search5 |
| MambaIR | Local enhancement + channel attention wrapper | Mamba 擅长长程，但对局部纹理/边缘/细节补得不够 | **DenseEncoder 后**、**每个 TFMambaBlock 前后**、**decoder 前** | 局部时频细节、辅音瞬态、谐波边缘、主观质量 | 中 | 如果和 DenseEncoder 功能重叠，收益可能被稀释 | 不是替换主块，而是把它当成“外壳”包住当前 Mamba block。citeturn11search15turn9search3turn8search1 |
| HDF-Net | Sub-band input + decoupled deep filtering head | 只预测 mask/phase correction 时，局部邻域修复能力不足 | **DenseEncoder 后**、**输出头/decoder 前** | 局部时频细节、频带建模、残余噪声清理 | 中 | deep filtering 会改变你的输出参数化与重建流程 | 第一轮建议只借鉴 sub-band 模块；deep filtering 放到后续。citeturn16search11turn16search3 |
| MP-SENet | 显式并行幅相解码 | 双流已经有了，但“显式 phase decoder + 并行重建”仍可加强 | **magnitude branch**、**phase branch**、**末端全局幅相融合** | 相位恢复、幅相解耦、最终重建质量 | 中 | 与你现有双流设计相似，增益可能来自 head 而非 backbone | 第一轮以“换输出头，不换主干”为原则复用其思路。citeturn5search1turn26search19 |
| MambaDC | Conv-Mamba hybrid block | 纯 Mamba 对局部模式不足，纯卷积对长依赖不足 | **TFMambaBlock 内部**、**bottleneck** | 局部+长程互补、时频细节与稳定性 | 中 | 没有公开代码，细节需自行补齐 | 只在 bottleneck 放 1–2 个 hybrid block，先看是否优于原始 TFMamba。citeturn33search0turn33search12 |
| LKM-UNet | Large-kernel bidirectional Mamba block | 语谱图上的谐波条纹、共振峰带是各向异性的局部结构 | **DenseEncoder 后**、**TFMambaBlock 外围**、**skip fusion 前** | 局部 pattern、频带走向、细节连续性 | 中 | 大核参数增多，可能和现有 down/up sampling 冲突 | 用“深层少量大核 + 浅层不动”的策略，先试 deepest two scales。citeturn10search10turn10search1turn10search4 |

这 10 个模块里，我认为**最像“首轮一定值得做”的前五个**是：  
**共享时频注意力、深度幅相交互/GRE 相位头、band-split、MambaIR 局部增强外壳、CADB/RWSA 这类注意力式融合器**。因为它们既对准了你的主要瓶颈，又不要求你推倒现在的双流框架。citeturn25search3turn29search15turn16search16turn11search15turn22search11turn25search1

## 第一轮实验推荐

下面给你一个我认为最适合 1–2 周窗口的实验清单。原则是：**不重写整网、先做插拔式 ablation、一次只改一处核心归因点**。

| 实验方向 | 具体改法 | 推荐插入位置 | 主要瞄准指标 | 为什么适合首轮 |
|---|---|---|---|---|
| 共享时频注意力微改 | 在现有 deepest 1–2 个 TFMambaBlock 后增加共享 time/frequency MHA；先共享参数，再与不共享对照 | bottleneck、最深两层 encoder | PESQ、STOI、COVL、跨噪声泛化 | 代码公开、改动局部、最容易在不破坏主干的前提下看出增益。citeturn25search3turn39search4turn39search1 |
| 深度幅相交互 + 新 phase head | 保留当前 magnitude branch；把 phase 输出头改成 GRE 风格相位头，并把末端全局幅相融合换成深度交互模块 | 末端全局幅相融合、phase branch | 相位恢复、PESQ、CSIG、COVL | 直接命中你双流系统最可能的结构短板，而且不用全面重构。citeturn29search15turn29search0 |
| 显式带分裂 | 在 DenseEncoder 后对频率轴做固定 band split；每个 band 内做 band-wise block，再把结果送入现有 FMamba | DenseEncoder 后、magnitude branch | PESQ、STOI、局部细节、噪声残留 | 与现有 FMamba 完全兼容；对语音先验最强，也最容易解释。citeturn16search16turn16search4turn16search11 |
| 局部增强外壳 | 用 MambaIR 风格 local enhancement + channel attention 包裹现有 TFMamba block，而不是替换它 | 每个 TFMambaBlock 前后、decoder 前 | 局部时频细节、主观质量、辅音清晰度 | 跨领域迁移成本低，且与你当前 U-Net/Mamba 组合契合度高。citeturn11search15turn9search3turn8search1 |
| 注意力式跨分支融合 | 用 CADB 的 Self-CA 或 RWSA 的 shared attention 替换你现在最关键的两次 cross-branch fusion | bottleneck 的多次 fusion、skip fusion | 幅相交互、泛化、最终重建质量 | 可以验证“fusion 方式”本身是不是当前性能瓶颈。citeturn22search11turn21view0turn25search1turn25search2 |

如果按**真正推荐的执行顺序**来排，我建议是：

1. **先做共享时频注意力微改**。这是最稳、最容易复现、最容易在你现有 Mamba 主干上直接验证收益的模块。citeturn25search3turn39search4  
2. **再做深度幅相交互 + 新 phase head**。因为你的 baseline 已经有多次 fusion，这一步最能回答“问题是在 backbone，还是在 fusion/phase 表示”。citeturn29search15  
3. **第三步做显式带分裂**。如果前两步有效，你就已经知道“交互问题”和“2D/全局问题”有改进空间；这一步继续补上 frequency prior。citeturn16search16turn16search11  
4. **第四步做局部增强外壳**。这一步通常对主观质量很友好，但要注意不要和 DenseEncoder 功能完全重叠。citeturn11search15turn9search3  
5. **最后再做注意力式跨分支融合替换**。这是为了确认你当前 bottleneck 多次 cross-fusion 是不是“有量无质”。citeturn22search11turn25search1  

如果你问我“**哪三个实验最像能把你的小论文再往上抬一截**”，我会投下面三项：

- **MambAttention 风格共享时频注意力**  
- **RENet 风格深度幅相交互 + GRE phase head**  
- **BSRNN/HDF 风格显式 band-split/sub-band 建模**  

这三项组合起来，刚好覆盖了**长程建模、幅相交互、频带先验**三类最高价值缺口。citeturn25search3turn29search15turn16search16turn16search11

## 暂不优先的方向

有几类方向看起来很热门，但我不建议你把它们放在首轮。

**整网替换型工作**不建议先做。比如 Mamba-SEUNet、MaTSE、以及更“完整系统化”的 BSP-MPNet，它们都不是单一模块，而是把 backbone、特征流、甚至训练范式一起换掉。你当然可以把它们当强对照或第二轮方案，但它们不适合回答“在我现有 baseline 上，哪一处最值得改”。citeturn32search1turn34search0turn30search1turn31search1

**纯 2D scan 或新 scan operator 驱动的视觉 Mamba**不建议在第一轮硬移植。VMamba/2DMamba 很值得关注，但从实验成本看，数据排布、扫描路径、算子依赖和性能调参都明显高于共享注意力、band-split、local enhancement 这几类模块。第一轮把它们作为“idea source”远比“马上落地完整替换”更现实。citeturn11search4turn13search3turn13search7

**扩散式或重生成式增强**也不适合当前阶段优先做。复杂循环一致性扩散这类方法确实与幅相联合恢复强相关，但它们更偏完整生成框架，训练与采样成本高、归因困难，而且通常很难在 1–2 周内做出公平而稳定的模块级结论。citeturn5search9

**以 GAN / metric-guided training 为主轴的训练范式**不适合一开始就上。MetricGAN-OKD 和 Uni-VERSA-Ext 都有价值，但我更建议你在结构筛选之后再上。否则即便指标涨了，你也很难判断是结构真有用，还是 loss 把指标“拉上去了”。citeturn20search0turn20search4turn20search6turn19search4

**无代码或代码状态不清晰的新趋势论文**，例如 SEMamba++、DMHRN、以及部分 2025–2026 的 hybrid Mamba 论文，我建议先盯住思想，不急着首轮落地。它们很可能是第二轮的好灵感，但不是最适合你当前节奏的工程切口。citeturn37search8turn37search5turn17search3turn6search5

## 行动清单与参考链接

按优先级排序，我会这样推进：

1. **把 MambAttention 的共享时频注意力做成一个独立 wrapper**，只接在 bottleneck 和 deepest encoder stage。先验证是否提升 PESQ/STOI/COVL，再决定是否全网铺开。citeturn25search3turn39search1turn39search4  
2. **把末端全局幅相融合改成“深度幅相交互 + GRE 相位头”**，并尝试引入 phase-aware 目标。先不动主干。citeturn29search15turn29search0turn27search0  
3. **在 DenseEncoder 后加固定 band-split 前端**，优先放在 magnitude branch；如果有效，再考虑带到 phase branch。citeturn16search16turn16search4  
4. **用 MambaIR 的 local enhancement + channel attention 包裹现有 TFMambaBlock**，观察局部语谱细节、爆破音/擦音清晰度与残余噪声。citeturn11search15turn8search1  
5. **把最关键的两次 cross-branch fusion 换成 CADB/RWSA 风格的 attention-guided fusion**，判断 fusion 机制是不是瓶颈。citeturn22search11turn25search1  
6. **如果前五步里至少有一个方向显著有效，再尝试 HDF 风格 sub-band + local filtering head**，把输出头从“只做 mask/phase correction”推进到“利用邻域 TF 信息做修补”。citeturn16search11  
7. **第二阶段再试 metric-guided fine-tune**，例如 Uni-VERSA-Ext 或更稳健的多指标监督；不要把它当首轮主线。citeturn20search6turn19search0  
8. **等首轮收敛后，再决定要不要挑战 VMamba/2DMamba/LKM-UNet 这类更强的 2D/local-global 结构迁移。**citeturn11search4turn13search3turn10search10  

最后给你一份**按用途分组的参考入口**，方便你直接点进去看论文或仓库：

- **直接可用的语音增强论文/代码**：MP-SENet、SEMamba、MambAttention、RWSA-MambaUNet、RENet、BSRNN、BSP-MPNet。citeturn5search1turn26search19turn37search0turn37search2turn25search3turn39search1turn25search1turn25search5turn29search15turn29search0turn16search16turn16search4turn30search1turn31search1  
- **可迁移的图像复原 / 医学图像 Mamba 模块**：MambaIR、MambaIRv2、VMamba、2DMamba、LKM-UNet、H-vmunet。citeturn11search15turn8search1turn9search1turn9search0turn11search4turn12search3turn13search3turn13search7turn10search10turn10search1turn36search0turn36search2  
- **适合作为灵感、但不建议首轮重投入的论文**：Mamba-SEUNet、MaTSE、SEMamba++、Complex-Cycle-Consistent Diffusion、DMHRN。citeturn32search1turn34search0turn37search8turn37search5turn5search9turn17search3  
- **训练目标/评价导向方向**：Global Phase Bias-Aware loss、Uni-VERSA-Ext、MetricGAN-OKD。citeturn27search0turn20search6turn19search4turn19search0turn20search0turn20search4

如果把整份调研压缩成一句最实用的话，那就是：**先别换整网，先在你现有的双流 Mamba U-Net 上，按“共享时频注意力 → 深度幅相交互/相位头 → 显式带分裂 → 局部增强外壳”的顺序做可插拔 ablation；这是最像会在 1–2 周内带来稳定指标收益的路线。**citeturn25search3turn29search15turn16search16turn11search15