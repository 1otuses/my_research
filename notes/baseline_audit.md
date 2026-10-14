# MADiff 代码基础核查

核查日期：2026-10-14。用户仓库：<https://github.com/1otuses/madiff>。默认分支 `main`。固定 commit：`00f88e9caf9cb8921e54f5bc13d0855d0206b537`。

本轮通过公开仓库读取和本地 shallow clone 做静态检查。用户已说明完成原算法复现，本轮没有读取对应 checkpoint、训练日志或复現曲线，故把复现状态记录为“用户报告”。

| 代码入口 | 已确认行为 | 后续修改位置 |
| --- | --- | --- |
| `diffuser/models/diffusion.py`，`GaussianDiffusion` | 支持逆动力学、观察轨迹去噪与 DDIM 调度接口 | 潜变量扩散目标、latent 归一化、解码动作损失 |
| `diffuser/models/ma_temporal.py`，`SharedConvAttentionDeconv` | 共享骨干与跨智能体特征交互 | 潜变量输入形状、软门控和图结构接口 |
| `diffuser/models/helpers.py`，`SelfAttention` | 先投影时序特征，再做智能体间注意力 | 图掩码语义与真实稀疏计算的区别 |
| `diffuser/datasets/sequence.py`，`get_conditions` | 可选单个智能体条件 | 训练后验与执行条件分离；保持 episode 边界 |
| `diffuser/utils/evaluator.py`，`_generate_samples` | CTDE 分支只填入 ego 历史，生成队友预测后取 ego 片段 | 潜空间方法必须保持同一真实信息访问范围 |
| `exp_specs/mamujoco/2halfcheetah/ctde_medium_history.yaml` | `decentralized_execution: True`、DDIM 开关、多个种子 | 原 baseline 的采样步数对照与最小试验配置 |
| `docs/vault_dataset_conversion.md` | Vault 转换与训练采用隔离环境 | 核对真实数据文件、质量标签和数据版本 |

链接全部固定到本轮版本：[模型](https://github.com/1otuses/madiff/blob/00f88e9caf9cb8921e54f5bc13d0855d0206b537/diffuser/models/diffusion.py)、[骨干](https://github.com/1otuses/madiff/blob/00f88e9caf9cb8921e54f5bc13d0855d0206b537/diffuser/models/ma_temporal.py)、[attention](https://github.com/1otuses/madiff/blob/00f88e9caf9cb8921e54f5bc13d0855d0206b537/diffuser/models/helpers.py)、[数据](https://github.com/1otuses/madiff/blob/00f88e9caf9cb8921e54f5bc13d0855d0206b537/diffuser/datasets/sequence.py)、[执行](https://github.com/1otuses/madiff/blob/00f88e9caf9cb8921e54f5bc13d0855d0206b537/diffuser/utils/evaluator.py)、[转换](https://github.com/1otuses/madiff/blob/00f88e9caf9cb8921e54f5bc13d0855d0206b537/docs/vault_dataset_conversion.md)。

开发时固定这个 commit 或另选经过验证的 commit，并记录相对上游的差异。fork 中已有历史研究目录；当前阶段的配置、实验与结论采用独立分支追踪。

依赖清单与 README 的版本口径存在需核对之处：README 安装示例为 PyTorch 1.12.1，`requirements.txt` 又含 PyTorch 2.0.1 的安装行。下一阶段先记录实际可运行的环境冻结文件，再讨论升级依赖。本轮未安装深度学习环境。
