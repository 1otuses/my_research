# 文献阅读清单

核查日期：2026-10-14。下表包含本轮核心材料；具体重合点见 [文献核查](../notes/related_work_verified.md)。代码“公开存在”与“已成功复现”分别记录。

| 优先级 | 文献 | 阅读用途 | 一手来源 |
| --- | --- | --- | --- |
| P0 | MADiff: Offline Multi-agent Learning with Diffusion Models，NeurIPS 2024 | 准确定位观察轨迹生成、逆动力学、ego 条件执行 | [论文](https://arxiv.org/abs/2305.17330)、[代码](https://github.com/zbzhu99/madiff) |
| P0 | Reasoning with Latent Diffusion in Offline Reinforcement Learning，ICLR 2024 | 潜空间技能、解码、压缩强度与 Q 约束 | [论文](https://arxiv.org/abs/2309.06599)、[代码](https://github.com/ldcq/ldcq) |
| P0 | Distilling Collaborative Dynamics into Latent Space for Implicit Coordination in Decentralized Multi-Agent Manipulation，2026 预印本 | 协同 latent 从训练特权信息转为本地推断 | [论文](https://arxiv.org/abs/2606.22982) |
| P0 | DoF: A Diffusion Factorization Framework for Offline Multi-Agent Reinforcement Learning，ICLR 2025 | 分布分解与独立执行的结构性条件 | [论文](https://openreview.net/forum?id=OTFKVkxSlL)、[代码](https://github.com/xmu-rl-3dv/DoF) |
| P1 | CoFlow: Coordinated Few-Step Flow for Offline Multi-Agent Decision Making，2026 预印本 | 协同门控、少步生成、效率比较 | [论文](https://arxiv.org/abs/2605.01457)、[源码 release](https://github.com/Guowei-Zou/coflow-release) |
| P1 | Multi-agent Coordination via Flow Matching，ICLR 2026 | 联合行为生成到快速个体策略 | [论文](https://arxiv.org/abs/2511.05005)、[代码](https://github.com/DongsuLeeTech/mac-flow) |
| P1 | Graph-Conditioned Diffusion for Offline Multi-Agent Reinforcement Learning，AAMAS 2026 | 图通信嵌入与团队泛化；标注通信条件 | [作者代码与论文入口](https://github.com/CORE-Robotics-Lab/GCD) |
| P1 | Efficient Multi-Agent Offline Coordination via Diffusion-Based Trajectory Stitching，ICLR 2025（MADiTS） | 拼接有效性、双向动力学与不平衡数据 | [论文](https://openreview.net/forum?id=EpnZEzYDUT) |
| P2 | CoDiMAD: Diffusion-Based Privileged Distillation for Communication-Free Multi-Robot Coordination，2026 预印本 | 多模态局部动作蒸馏；其 oracle 依赖在线采集 | [论文](https://arxiv.org/abs/2607.09587) |
| P2 | Deep Variational Information Bottleneck，ICLR 2017 | 任务预测与 KL 的信息论解释 | [论文](https://arxiv.org/abs/1612.00410) |
| P2 | The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables，ICLR 2017 | 正确的离散图松弛与梯度估计 | [论文](https://arxiv.org/abs/1611.00712) |
| P2 | Denoising Diffusion Implicit Models，ICLR 2021 | 与相同步数采样器做公平速度比较 | [论文](https://arxiv.org/abs/2010.02502) |

本地 `CoFlow.pdf` 是 arXiv v2（2026-05-12），在线已有后续版本。实验范围和结论引用必须标记具体版本，不能混用 v2 摘要、后续正文与 release README。

本轮还抽读了 MASIA、MF-Diffuser、CODI 等本地材料。MASIA 提供任务相关信息聚合参考但依赖通信；MF-Diffuser 使用平均场设定；CODI 本地版本为匿名稿。这些材料需要后续身份、版本及适用假设核查，当前不把它们作为主结论依据。
