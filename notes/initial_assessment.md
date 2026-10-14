# Bi-ST-MADiff 初步评估

日期：2026-10-14。依据：本地 `offline-marl.pdf`，5 页，以及本轮核查的一手论文和作者代码。当前属于可行性调研，尚未证明创新性或实验收益。

## 当前结论

MADiff 适合作为第一阶段 baseline。用户已有复现基础，fork 已有 CTDE 配置、DDIM 接口和数据转换说明。四个模块同时引入会使失败原因和收益来源难以辨认；先验证潜空间压缩保留协同信息，再逐项增加约束，是本轮建议。[用户 fork](https://github.com/1otuses/madiff)、[MADiff 论文](https://arxiv.org/abs/2305.17330)

整体思想具有多个直接近邻：LDCQ 已在离线 RL 中学习技能潜空间并进行扩散；CLS-DP 已把协同动力学压缩为潜变量并从局部观察驱动分布式扩散策略；GCD 已结合图结构与多智能体扩散；CoFlow 已结合协同门控与少步生成。模块组合本身不足以确立新颖性。[LDCQ](https://arxiv.org/abs/2309.06599)、[CLS-DP](https://arxiv.org/abs/2606.22982)、[GCD](https://github.com/CORE-Robotics-Lab/GCD)、[CoFlow](https://arxiv.org/abs/2605.01457)

更值得检验的候选问题是：严格无通信的执行条件下，任务相关的局部潜空间压缩，能否在混合质量数据上保留协同模式，同时降低规划代价？这是研究假设，仍需与普通自编码器、局部预测辅助损失和少步生成基线比较。

## 草稿中需要先修正的部分

| 原文位置 | 发现与影响 | 建议 |
| --- | --- | --- |
| 第 1 页摘要与贡献 | 已写“实验证明”“显著超越”，文中没有结果表、种子或统计证据 | 改为拟研究的假设与评估计划 |
| 第 1–2 页 MADiff 描述 | 将 MADiff 概括为原始联合动作上的扁平扩散；其主要方案生成观察轨迹，经逆动力学得到动作，并在个体潜在特征间做注意力交互 | 准确定位要替换的观察/轨迹表示与代码分支 |
| 第 3.7 节执行流程 | 测试时用真实全局状态规划、再下发潜变量，违反用户本轮确认的严格无通信执行约束 | 潜变量及图条件必须从本地历史和本地生成的队友预测得到 |
| 第 3.4 节 Local VIB | 文字目标含未来状态，实际损失只预测回报；总损失没有动作重建/解码训练目标 | 明确表征监督与动作解码损失，监测潜变量塌缩和关键动作信息丢失 |
| 第 3.5 节 TIG | 边密度加图平滑是稀疏化正则；尚未给出拓扑互信息的变量、概率模型或变分界 | 先称“稀疏协同图”，因果与信息瓶颈结论另行证明 |
| 第 3.5 节图正则 | 单看边密度与平滑的两个非负项，空图会同时消除两项；图平滑还可能惩罚互补角色 | 增加任务相关的结构效用证据，监测空图/表示塌缩，并比较同密度图 |
| 第 3.5–3.6 节门控 | Bernoulli Concrete 通常使用 logistic 噪声；单个 Gumbel 加 sigmoid 的分布需重新推导。硬阈值掩码也会截断该路径梯度 | 明确二元 Concrete/直通估计方式，训练先用软门控；保留自环以避免整行被屏蔽 |
| 第 3.3 节频域假设 | 高频动作可能是紧急避障或精确协同所需信号；低频也可能是次优漂移 | 先验证频谱和局部残差的可预测性，连续控制先做；离散动作类别编号不适合直接 EMA |
| 第 1 页与第 3.6 节效率 | 低维潜变量与稀疏掩码不自动保证少步收敛或真实加速；密集 attention 乘掩码仍可能执行完整矩阵运算 | 报告总延迟、模型调用次数、编码/解码开销，并与相同步数 DDIM 比较 |
| 第 4 节实验名称 | OG-MARL 是数据/基准体系，MPE 数据还涉及 OMAR；MAT 与 MADT 命名需要消歧；SMACv2 数据不能自动由 SMAC 推断存在 | 逐任务核查数据版本、代码与执行条件 |
| 第 3.1–3.6 节符号 | 原稿环境时间与扩散步分别使用了混杂的 $t$ 与 $k$ | 统一环境时间 $k$、扩散时间 $t$，规划长度 $H$ |

上述 MADiff 定位以论文和固定版本代码为依据：[diffusion.py](https://github.com/1otuses/madiff/blob/00f88e9caf9cb8921e54f5bc13d0855d0206b537/diffuser/models/diffusion.py)、[evaluator.py](https://github.com/1otuses/madiff/blob/00f88e9caf9cb8921e54f5bc13d0855d0206b537/diffuser/utils/evaluator.py)。二元松弛参考 [Concrete Distribution](https://arxiv.org/abs/1611.00712)。其余问题属于对当前草稿公式和信息流的直接分析。

## 开源项目选择

| 项目 | 建议用途 | 本轮确认的限制 |
| --- | --- | --- |
| [MADiff fork](https://github.com/1otuses/madiff) | 第一阶段实现基础与主要 baseline | 静态核查；尚未核验已有复现实验日志 |
| [CoFlow release](https://github.com/Guowei-Zou/coflow-release) | 少步生成强比较；若研究目标转向少步推理，可评估作为实现基础 | PyTorch；正文版本与 release 文档的实验范围需对齐；未运行 |
| [DoF](https://github.com/xmu-rl-3dv/DoF) | 中心化生成到个体生成的结构性参考与比较 | IGD 的假设、分解目标需细读，不能直接套到任意潜变量图上 |
| [GCD](https://github.com/CORE-Robotics-Lab/GCD) | 图条件与 SMACv2 参考 | 论文使用通信设定，须标注信息预算差异 |
| [LDCQ](https://github.com/ldcq/ldcq) | 编码器、技能解码器、潜空间扩散的模块参考 | 单智能体；原方法带 Q-learning，移植会引入额外估值问题 |
| [MAC-Flow](https://github.com/DongsuLeeTech/mac-flow) | 快速分布式策略与蒸馏参考 | 官方实现使用 JAX，迁移到当前 PyTorch 工程需成本 |

本轮建议先继续 MADiff。是否迁移 CoFlow，应由目标问题与一轮可复现实验决定；少步速度已是竞争较充分的方向。[MAC-Flow 论文](https://arxiv.org/abs/2511.05005)、[CoFlow 官方代码](https://github.com/Guowei-Zou/coflow-release)

初步检索不能支持“该组合无人研究”或“理论空白”的结论。下一轮需要围绕协同表征保真、局部潜变量预测、信息压缩与多模态协调做更集中检索。
