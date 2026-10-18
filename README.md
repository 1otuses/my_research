# my_research

记录基于扩散模型的离线多智能体强化学习研究：问题定义、文献证据、研究决策、实验协议与阶段成果。

当前阶段：S0 初步调研与方向收敛。更新时间：2026-10-18。所有算法收益均为待验证假设；已完成阅读、代码静态核查和研究文档初始化，尚未开展新算法实验。

GitHub 仓库：[1otuses/my_research](https://github.com/1otuses/my_research)。初始研究文档已通过本机 Git 推送到 `main`。后续阶段通过分支和 PR 记录。

当前依据是用户提供的 `offline-marl.pdf` 中的 Bi-ST-MADiff 设想。历史 copula、credit diffusion 等研究只作为既有项目背景，当前工作优先服从本轮设想。

当前原稿按用户确认的讨论更新为“时间抽象＋协同修正”的潜空间规划，移除动作分频及对应残差动作网络，保留原有章节组织与严格 CTDE 约束。当前以 `research/proposals/offline-marl.docx` 为准；2026-10-15、2026-10-16 记录保留为历史版本。

执行约束已确认：完全合作场景、严格 CTDE、执行时各智能体只读取局部观察历史且无通信。用户已选择将本研究仓库公开。

当前实施顺序：以已有 MADiff fork 为基础，先验证观察片段编码解码和动作恢复，再训练潜空间时间生成分支，最后检验协同残差与软门控的增益。方案尚未实现，所有收益待验证。

| 入口 | 内容 |
| --- | --- |
| [Word 原稿](research/proposals/offline-marl.docx) · [本次修改范围](research/proposals/2026-10-18-temporal-coordination-changes.md) | 时间抽象、协同修正、潜空间生成及三阶段训练 |
| [历史改写建议](research/proposals/2026-10-15-latent-madiff-revision.md) | 此前整篇改写的分析记录，已不作为当前原稿 |
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
| [原稿恢复记录](logs/2026-10-16.md) | 恢复原稿及通信局部修改的验证与同步 |
| [设计更新记录](logs/2026-10-18.md) | 移除动作分频并写入用户确认的新框架 |
| [源文件索引](sources/manifest.json) | 原始 PDF 指纹和文献版本 |
| [资料库目录](sources/library_inventory.csv) | 本轮可访问的 150 份本地 PDF 路径与大小 |

算法实现继续在 [1otuses/madiff](https://github.com/1otuses/madiff) 的研究分支推进。本仓库通过代码 commit、配置和实验记录关联实现。

数学记号统一为环境时间步 $k$、扩散去噪时间步 $t\in[0,T]$、全局状态 $s$、局部观察 $o_i$、动作 $a_i$、联合动作 $\mathbf a$。规划长度使用 $H$。
