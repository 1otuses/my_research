# 需要补齐的知识基础

按当前严格 CTDE 研究问题安排阅读，完成一个可解释的最小原型再深化理论。

| 顺序 | 内容 | 阅读目标 | 主要材料 |
| --- | --- | --- | --- |
| 1 | Dec-POMDP、局部历史、CTDE | 能逐项标记训练与执行的信息来源 | MADiff §2–3、DoF |
| 2 | 条件扩散、CFG、DDIM | 理解噪声预测、条件丢弃、采样步数与引导的关系 | MADiff、Decision Diffuser、DDIM |
| 3 | 轨迹窗口编码、技能潜变量、局部解码 | 区分训练后验与执行先验，理解潜变量坍塌和动作保真 | LDCQ、CLS-DP |
| 4 | VIB、KL 与任务充分性 | 说明 KL 上界对应哪个互信息，压缩损失保留什么 | Deep Variational Information Bottleneck |
| 5 | 协同模式、分布分解 | 理解独立采样为何可能失配，掌握 IGD 的条件 | DoF、MAC-Flow |
| 6 | 图松弛与稀疏注意力 | 区分软门控、硬采样、因果识别和计算稀疏 | Concrete Distribution、GCD、CoFlow |
| 7 | 连续控制的频域分析 | 验证 EMA 延迟、频谱、残差可预测性与闭环控制效果 | 原任务动作序列与后续受控实验 |

每次阅读记录：文章解决的问题、真实执行输入、优化目标、可迁移模块、与当前假设的差异、尚未理解的公式。原文提到的理论结论应保留假设条件。

第 4 项的一手来源：[Deep VIB](https://arxiv.org/abs/1612.00410)。第 6 项的一手来源：[Concrete Distribution](https://arxiv.org/abs/1611.00712)。其他文献链接见 [阅读清单](../references/reading_list.md)。
