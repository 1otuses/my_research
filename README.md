# my_research

记录基于扩散模型的离线多智能体强化学习研究：问题定义、文献证据、研究决策、实验协议与阶段成果。

当前阶段：S0 初步调研与方向收敛。更新时间：2026-10-15。所有算法收益均为待验证假设；已完成阅读、代码静态核查和研究文档初始化，尚未开展新算法实验。

GitHub 仓库：[1otuses/my_research](https://github.com/1otuses/my_research)。远端已创建，初始文档仍待上传：2026-10-15 写入请求返回连接器权限错误 403。后续阶段通过分支和 PR 记录。

当前依据是用户提供的 `offline-marl.pdf` 中的 Bi-ST-MADiff 设想。历史 copula、credit diffusion 等研究只作为既有项目背景，当前工作优先服从本轮设想。

执行约束已确认：完全合作场景、严格 CTDE、执行时各智能体只读取局部观察历史且无通信。用户已选择将本研究仓库公开。

初步建议：以已有 MADiff fork 为第一阶段开发基础，先验证协同信息保真的潜空间压缩；CoFlow-D 与 DoF 作为后续重点比较对象。时序分频和稀疏图门控分别经过独立消融后，再决定是否组合。这个建议属于研究判断。

| 入口 | 内容 |
| --- | --- |
| [本轮简报](notes/initial_assessment.md) | 相似工作、可行性、草稿中需要修正的关键问题 |
| [文献核查](notes/related_work_verified.md) | 一手来源、重合点、适用边界 |
| [代码基础评估](notes/baseline_audit.md) | 固定版本、修改入口、baseline 选择 |
| [当前问题定义](research/current_direction.md) | 严格 CTDE 下的候选假设与最小方案 |
| [阶段计划](research/roadmap.md) | 逐步验证与退出条件 |
| [知识基础](research/foundations.md) | 阅读顺序与需要掌握的概念 |
| [文献清单](references/reading_list.md) | 必读与补充材料 |
| [实验协议](experiments/protocol.md) | 公平比较、种子、数据与延迟口径 |
| [GitHub 流程](research/github_workflow.md) | 阶段提交、分支、PR 与上传状态 |
| [研究日志](logs/2026-10-14.md) | 本轮完成项和未解决项 |
| [上传记录](logs/2026-10-15.md) | GitHub 初始化与同步方式 |
| [源文件索引](sources/manifest.json) | 原始 PDF 指纹和文献版本 |
| [资料库目录](sources/library_inventory.csv) | 本轮可访问的 150 份本地 PDF 路径与大小 |

算法实现继续在 [1otuses/madiff](https://github.com/1otuses/madiff) 的研究分支推进。本仓库通过代码 commit、配置和实验记录关联实现。

数学记号统一为环境时间步 $k$、扩散去噪时间步 $t\in[0,T]$、全局状态 $s$、局部观察 $o_i$、动作 $a_i$、联合动作 $\mathbf a$。规划长度使用 $H$。
