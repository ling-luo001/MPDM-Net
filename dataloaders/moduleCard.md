# Module Card: 模块名称

## 1. 来源
Paper:
Year:
Code:
Task:

## 2. 这个模块解决什么问题？
例如：增强局部时频细节 / 加强长程依赖 / 降低参数 / 改善幅相交互

## 3. 核心结构
输入:
输出:
主要操作:
是否需要改变主干网络:

## 4. 能插到我现有模型哪里？
- Encoder 后
- Mamba block 内部
- Magnitude branch
- Phase branch
- Cross-branch fusion
- Decoder 前
- Loss 层

## 5. 预计收益
性能提升 / 参数降低 / 泛化增强 / 训练稳定

## 6. 改代码难度
1-5 分

## 7. 风险
维度不匹配 / 显存增加 / 训练慢 / 依赖特殊 CUDA / 没有代码

## 8. 实验编号
EXP-xxx

## 9. 最终结论
保留 / 淘汰 / 待复查