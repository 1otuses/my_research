# Bi-ST-MADiff 相关工作初步核查

核查日期：2026-10-14。任务边界：固定离线数据训练、完全合作环境、执行时每个智能体仅使用自己的局部历史且无通信。以下“已核查”指论文或作者源码的事实；“判断”指基于这些事实提出的研究建议。此次检索用于初筛，不构成完整的新颖性证明。

## 初步结论

**继续以已复现的 MADiff 为实现基础可行；研究主线应收敛为“在严格分布式执行下，压缩局部历史如何保留协同信息与多模态行为”。** “协同 latent + decentralized diffusion”已经有明确近邻 CLS-DP；“latent diffusion + decoder”已有 LDCQ；“压缩图结构和节点信息”已有 CGIBNet。因此，Bi-ST-MADiff 的组件组合只能作为候选方案，贡献需要落实为具体机制、可证伪假设及证据。来源见下表。[CLS-DP](https://arxiv.org/html/2606.22982v1)、[LDCQ](https://arxiv.org/html/2309.06599v1)、[CGIBNet](https://arxiv.org/abs/2112.10374v4)。

## 八项优先相关工作

| 工作与核查来源 | 已核查的主要思想 | 对当前设想的重合与可借鉴点 | 设定与代码边界 |
|---|---|---|---|
| **CLS-DP**，2026：[论文全文](https://arxiv.org/html/2606.22982v1)，§IV-B、式(7)–(12) | 训练时以未来联合轨迹形成 privileged posterior，KL 对齐局部 RGB 与共同任务指令产生的 prior；冻结 contextualizer 后训练各智能体 diffusion action expert。 | **优先级最高**：已经研究将协作动态压缩成局部可推断 latent，再条件化去噪。可借鉴训练时 posterior / 执行时 prior 的清晰接口。需要解释 VIB 历史压缩与其 CVAE latent 的目标差异。 | 执行无通信；实验是六个 RoboFactory 模仿学习任务，非对混合质量数据进行回报优化的标准 offline MARL。本文核查了全文方法；本轮未确认可复现的作者代码。 |
| **MAC-Flow**，ICLR 2026：[论文](https://openreview.net/pdf?id=2L6MffR0ut)、[官方代码](https://github.com/DongsuLeeTech/mac-flow)、[实现](https://github.com/DongsuLeeTech/mac-flow/blob/main/agents/macflow.py) | 先拟合 joint flow behavior policy，再结合蒸馏与全局价值目标学习分布式单步策略。 | 是“集中式训练得到协同行为 → 局部高效执行”的直接比较对象；可借鉴蒸馏目标与回报提升机制。将 DDIM 加速作为主创新，难以与其单步方案形成充分区别。 | 标准 offline MARL 且支持分布式执行。源码实际使用 **JAX / Flax / Optax**，已查到 agents/macflow.py 等文件；尚未运行。 |
| **CoFlow**，2026：[当前 v4 全文](https://arxiv.org/html/2605.01457v4)，§3.2–3.4、§4；[作者源码 release](https://github.com/Guowei-Zou/coflow-release) | 联合协调 velocity attention、门控和 finite-difference consistency，使少步生成仍建模跨智能体依赖。CoFlow-D 在每个局部预测中遮掉不可用 teammate context，仅执行自己的动作。 | 高度覆盖“协同 + 门控 + 少步生成”。可借鉴协调保持的机制实验，比较同样计算预算下的收益。 | **网页仓库 coflow 与源码仓库 coflow-release 要区分**。GitHub API 已确认 release 的 diffuser/、exp_specs/、run_scripts/、run_experiment.py 与依赖文件真实存在；未运行。当前 v4 主文使用 MPE/SMAC，旧版本及仓库可能仍描述 MA-MuJoCo。 |
| **GCD**，AAMAS 2026：[论文 DOI](https://doi.org/10.65109/BMST1644)、[作者代码](https://github.com/CORE-Robotics-Lab/GCD)、[运行说明](https://github.com/CORE-Robotics-Lab/GCD/blob/main/gcd/README.md) | Graph communication embeddings 与团队回报共同作为 diffusion 的 CFG 条件；SMACv2 上研究团队组成泛化。 | 已覆盖“graph + embedding + CFG + offline MARL”。可借鉴训练图作为教师信息，以及异构图与泛化实验。 | **其 decentralized 允许通信**：本地 GCD.pdf 第3–4页 §4.1 明确每次去噪向邻居传消息，故信息条件与本项目不同。作者仓库基于 MADiff，含真实模型、配置、训练及评估脚本；未运行。 |
| **LDCQ** / Reasoning with Latent Diffusion in Offline Reinforcement Learning，ICLR 2024：[全文](https://arxiv.org/html/2309.06599v1)，§4、§6；[作者代码](https://github.com/ldcq/ldcq) | VAE 压缩轨迹为 latent skills，扩散建模 latent prior；Q 在扩散产生的候选中选择，再由低层 decoder 执行动作。 | latent diffusion 和低层解码已有基础，可借鉴两阶段训练与数据支持约束。特别要区分“压缩历史以形成条件”与“压缩未来动作作为扩散目标”。 | 单智能体 D4RL，迁移到无通信 MARL 需要新增联合协调机制。源码已公开。作者明确指出长 horizon 的收益存在任务依赖，locomotion 的时间抽象收益有限。 |
| **MADiTS**，ICLR 2025：[会议正文](https://proceedings.iclr.cc/paper_files/paper/2025/file/3e7cf447f21cd11c846463affefce665-Paper-Conference.pdf)，§4、§5、附录F | 生成并拼接高回报段，加入双向动力学一致性约束；通过离线 credit assignment 与局部重加噪改善较差智能体的段。 | “时空协同”这一动机已有直接相关工作。可借鉴数据不平衡、动力学一致性和协同失败指标。 | 属于数据增强流程，再训练下游 offline MARL；本轮未确认作者完整 MADiTS 开源仓库。论文提到的 OffPyMARL 是下游算法实现，须避免当作 MADiTS 全部代码。 |
| **CoDiMAD**，2026：[全文](https://arxiv.org/html/2607.09587v1)、[摘要与版本](https://arxiv.org/abs/2607.09587) | 在线 MAPPO 训练 privileged oracle，收集 local-observation / oracle-action 离线对，再用 diffusion 蒸馏为无通信学生。 | 可借鉴对局部观察歧义、确定性蒸馏平均模式及 diffusion 多模态恢复的分析；适合检查独立学生是否选择兼容的协作模式。 | 学生执行无通信，但数据来自在线训练 oracle；当前固定数据集研究应明确这一差异。摘要写接受后公开源码，本轮未确认已发布作者代码。 |
| **CGIBNet**，arXiv 2021 / v4 2022：[作者论文](https://arxiv.org/abs/2112.10374v4) | 同时用 graph information bottleneck 压缩通信图边与节点信息，分别学习与谁交流、交流什么。 | 对“VIB + 稀疏图”是基础近邻。可借鉴目标构造，但需要证明压缩保留的是任务相关信息。 | 研究带宽受限的有通信 MARL，不能直接满足本项目无通信执行约束；训练图教师是潜在迁移方式。本轮核查摘要与版本，未审计实现。 |

**版本记录**：本地 `MARL/offline/CoFlow.pdf` 是 2026-05-12 v2；线上最新是 2026-09-21 v4，正文 §4 以 48 个 MPE/SMAC 配置描述。表中的方法及信息条件以 v4 为准，避免把旧摘要、README 与新正文的配置数字拼接。版本变化见 [arXiv 提交历史](https://arxiv.org/abs/2605.01457)。

## 两项补充阅读

[OPAL](https://arxiv.org/abs/2010.13611)是 LDCQ 之前的时间扩展行为 primitive 学习基础，可用于理解 temporal abstraction、decoder 与数据支持；它本身没有 diffusion 或多智能体协调目标。

[MF-Diffuser v2](https://arxiv.org/html/2605.30190v2)主要在 mean-field / 大规模群体设定研究代表性智能体、分布空间规划及规模扩展。其问题假设与当前少量异构智能体的严格 CTDE 有距离；本轮读到正文并核查版本，未确认作者开源项目，适合作为后续规模化方向阅读。

## 当前方案需要首先澄清的机制

**严格 CTDE 的条件应作用于完整推理链。** 如果执行时用真实联合历史集中生成 latent，再各智能体局部解码，则 latent 仍携带执行时共享信息。建议训练时可使用 joint graph teacher / privileged posterior，执行时使用每智能体局部历史的 prior 或 student。CLS-DP 已给出这种接口，因此应进一步比较“只重构未来动态”与“明确保留协同且压缩历史”的目标。上述判断基于 [CLS-DP §IV-B](https://arxiv.org/html/2606.22982v1)和 [GCD 方法与代码](https://github.com/CORE-Robotics-Lab/GCD)。

**局部多模态边缘分布不能自动证明联合协同保持。** 在观察有歧义时，各智能体独立采样可能选中不兼容模式。建议用存在多种对称成功模式的小任务，统计联合模式冲突和成功率；再讨论是否能仅靠历史辨识解决，及共享随机性是否被允许。这里是可验证的风险分析，并非上述论文已证明的普遍不可能性定理。[CoDiMAD](https://arxiv.org/html/2607.09587v1)、[MAC-Flow](https://openreview.net/pdf?id=2L6MffR0ut)可提供蒸馏与协调动机。

**EMA 低频标签及图平滑需要单独验证语义。** 同队合作允许不同角色采取不同甚至相反动作，图邻居接近不能直接解释为协作。建议先检验任务是否确有慢协同 / 快控制分离，并把平滑作用于经语义监督的协同表示；预先定义异构角色、急转动作及协调反例。这是当前方案的研究假设，尚无本轮文献证据足以确认。

## 最小可证伪验证

| 假设 | 最小对照 | 推翻或收缩方向的信号 |
|---|---|---|
| 历史压缩去除冗余并保留协同 | 相同 latent 维度、参数与采样预算：确定性压缩、无 KL、VIB；clean 与预设观察扰动评估 | 仅动作重构更好，联合成功或协调指标变差；或同预算收益不稳定 |
| 协同监督比通用表示重构更有价值 | CLS-DP 思想的未来联合轨迹重构教师，对照加入明确协同目标的学生 | 增益完全由更大历史窗口或教师额外信息解释 |
| 稀疏训练图有独立价值 | 全连接、固定/随机稀疏、可学习稀疏；保持执行信息相同 | 去掉图学习后效果相同，或图平滑伤害异构角色 |
| 低高频分工有任务依据 | 无分解、EMA、多种截止尺度；测突变动作任务 | smoothing 导致关键反应延迟，或低频无法稳定反映协作角色 |

判断：第一阶段完成 MADiff 分布式评估与“局部历史 latent”最小改动，先验证压缩 / 协同目标；将稀疏图和 EMA 留作逐个模块的消融。MAC-Flow 适合效率与严格执行条件的强基线；CLS-DP 适合作为机制借鉴与概念近邻；GCD 的通信条件须单独标注。无需因较新方法的存在立即迁移已经复现的 PyTorch 工程。

## 核查范围与局限

本轮通过论文全名、方法名、作者 GitHub 搜索交叉确认身份；事实依据为 arXiv 作者正文、会议正文及作者代码。GCD 还读取了本地九页正文第1、3–4、8页；MADiTS 核查正文与附录；MAC-Flow 核查论文方法和 JAX 实现；CoFlow 核查 v4 方法、实验及 GitHub API 的实际文件；CLS-DP 核查 §IV 方法。本轮未运行任何第三方算法，也未保证新方法优于用户已有复现。

这里只支持“多个组件及关键机制已有近邻”的结论。关于 Bi-ST-MADiff 是否具有足够新意，应在执行信息条件澄清、核心目标形式化及最小实验结果出现后继续核查；不得据此宣称不存在相同工作、首次提出或已验证有效。
