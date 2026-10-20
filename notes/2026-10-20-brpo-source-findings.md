# BRPO 设计相关原论文核查

核查日期：2026-10-20。范围仅包括 OMIGA、AlberDICE、CFCQL；论文事实与对本研究的推断分别标明。页码为 PDF 印刷页码，与本地文件一致。

## 1. OMIGA：等价转换依赖具体结构

来源：[NeurIPS 2023 正文](https://proceedings.neurips.cc/paper_files/paper/2023/file/a46c84276e3a4249ab7dbf3e069baf7f-Paper-Conference.pdf)，本地 `MARL/offline/OMIGA.pdf`；[官方代码](https://github.com/ZhengYinan-AIR/OMIGA)。

**论文事实。** §4.1，p.4 明确假设联合行为分布可按局部观察分解。§4.2 Eq.(8)，p.5 对联合 Q 与联合 V 使用同一组非负线性混合权重、同一偏置：

```latex
\mu_{\mathrm{tot}}(\mathbf a\mid\mathbf o)
=\prod_{i=1}^{N}\mu_i(a_i\mid o_i),
\qquad
Q_{\mathrm{tot}}(\mathbf o,\mathbf a)
=\sum_i w_i(\mathbf o)Q_i(o_i,a_i)+b(\mathbf o),
\qquad
V_{\mathrm{tot}}(\mathbf o)
=\sum_i w_i(\mathbf o)V_i(o_i)+b(\mathbf o),
\quad w_i(\mathbf o)\ge0.
```

符号：\(N\) 为智能体数，\(i\) 为智能体索引；\(\mathbf o=(o_1,\ldots,o_N)\) 为联合观察；\(\mathbf a=(a_1,\ldots,a_N)\) 为联合动作；\(\mu_{\rm tot}\) 为联合行为策略，\(\mu_i\) 为局部行为策略；\(Q_i,V_i\) 为局部动作价值和状态价值，\(Q_{\rm tot},V_{\rm tot}\) 为联合值；\(w_i\) 为依赖联合观察但不依赖动作的非负混合权重；\(b\) 为共享偏置。

**论文事实。** Proposition 4.3 / Eq.(9)，p.6 的局部策略形式来自行为分解、上述 Q/V 同结构，以及指数函数对求和的乘积转换：

```latex
\pi_i^*(a_i\mid o_i)
=\mu_i(a_i\mid o_i)
\exp\!\left[
\frac{w_i(\mathbf o)}{\alpha}
\left(Q_i^*(o_i,a_i)-V_i^*(o_i)\right)
\right].
```

新增符号：\(\pi_i^*\) 是文中最优局部策略；星号表示相应最优解；\(\alpha>0\) 是行为正则系数。

**论文事实。** Eq.(10) 要求每个局部策略归一化；Eq.(11)/(12) 用指数回归学习 \(V_i\)，Eq.(13) 用数据转移及 \(V_{\rm tot}\) 的 TD 目标训练 Q；Eq.(14)，p.7 用权重 \(\exp[w_i(Q_i-V_i)/\alpha]\) 做局部加权行为克隆。训练三步均只查询数据动作。

**研究推断与限制。** 可以借鉴“信用分配与策略约束共用权重”“从同一全局目标推导训练项”“in-sample 学习”。若替换为一般非单调混合器，指数中的交互项无法按该证明分成独立因子，不能继续宣称 Eq.(9) 的等价性。若离线数据是多个协调团队策略的混合，其联合行为分布通常也不满足上述乘积假设。

**更细的可执行性注意。** 原文 Eq.(9) 左侧只写 \(o_i\)，右侧却含 \(w_i(\mathbf o)\)。其最终实现为局部 Actor 用带全局信息的权重做加权 BC。若相同 \(o_i\) 对应多个全局上下文，不能仅凭该代数式就保证一个只读 \(o_i\) 的 Actor 对所有上下文精确满足同一个点态最优条件；这是局部策略表示/投影的问题，而非额外信息可以用于执行。

## 2. AlberDICE：联合占用分布约束与当前队友校正

来源：[原论文](https://arxiv.org/abs/2311.02194)，[公开 PDF](https://openreview.net/pdf?id=LhVJdq4cZm)，本地 `MARL/offline/AlberDICE.pdf`；[官方代码](https://github.com/dematsunaga/alberdice)。

**论文事实。** §3 Eq.(2)-(3)，p.3-4 固定其他智能体当前策略，优化当前智能体占用分布：

```latex
\max_{d_i\ge0}\;
\mathbb E_{(s,a_i)\sim d_i,\;\mathbf a_{-i}\sim\pi_{-i}(\cdot\mid s)}
[r(s,\mathbf a)]
-\alpha D_{\mathrm{KL}}
\left(
d_i(s,a_i)\pi_{-i}(\mathbf a_{-i}\mid s)
\;\middle\|\;d_{\mathcal D}(s,\mathbf a)
\right).
```

该优化还受 Bellman flow 约束，不是任意的概率拟合。

符号：\(s\) 是全局状态，\(a_i\) 是自身动作，\(\mathbf a_{-i}\) 为其他智能体动作；\(d_i\) 为当前固定队友下可实现的折扣状态—自身动作占用分布；\(\pi_{-i}=\prod_{j\ne i}\pi_j\) 为去中心化当前队友策略；\(d_{\mathcal D}\) 为数据联合状态—动作分布；\(r\) 为共享奖励；\(\alpha>0\) 为正则系数；\(D_{\rm KL}\) 为 KL 散度。

**论文事实。** Eq.(9)，p.5 用自回归 BC 预训练条件队友行为分布 \(\pi_{-i}^{\mathcal D}(\mathbf a_{-i}\mid s,a_i)\)。Eq.(11) 的校正比为：

```latex
\rho_i(s,\mathbf a)
=
\frac{\prod_{j\ne i}\pi_j(a_j\mid s)}
{\pi_{-i}^{\mathcal D}(\mathbf a_{-i}\mid s,a_i)}.
```

新增符号：\(\rho_i\) 为队友策略重要性比；\(\pi_j\) 为队友当前目标策略；\(\pi_{-i}^{\mathcal D}\) 为以状态和自身动作为条件的联合队友数据分布。分母通过条件自回归模型表示，**不是独立边缘行为分布的乘积**。

**论文事实。** 重要性重采样根据 \(\rho_i\) 调整数据，使当前 agent 的 reduced MDP 反映变化中的队友策略。文中明确指出比值含 \(N-1\) 个策略项，可能方差很大，并使用重采样缓解。§4 Lemma 4.1 保证精确构造下各 agent 优化同一个正则化团队目标；Theorem 4.2 是表格情况下交替求解正则 LP 的 Nash 收敛结论。实践则每轮对值函数和策略做一步梯度更新，不能直接等同理论的精确 best response。

**对齐启发。** 该方法最值得借鉴的是“各 agent 优化同一个联合正则目标”，以及用条件数据模型将当前队友与数据队友联系起来。其 KL 包含状态占用偏移，固定数据状态上的 BRPO Actor KL 一般只约束动作条件分布，两者不能直接称为同一目标。

**局限。** 自回归模型的误差、重要性权重方差、数据覆盖及局部策略提取误差仍然存在。其理论以 MMDP 全状态策略为基础，严格部分可观测 CTDE 的扩展在实验/附录 J.1 中处理；不能把表格全状态 Nash 定理原封不动移植到神经网络局部历史 Actor。

**XOR 证据边界。** p.2 Fig.1 和 p.7 Fig.2/Table1 演示了特定方法/数据下的联合 OOD，以及用去中心化确定策略选择一个协同最优模式的可能性。该例不能支持“所有 IGM 方法都无法得到任何最优联合动作”这种过强结论。

## 3. CFCQL：Q 正则的反事实分解

来源：[原论文](https://arxiv.org/abs/2309.12696)，[公开 PDF](https://openreview.net/pdf?id=62zmO4mv8X)，本地 `MARL/offline/CFCQL.pdf`。

**论文事实。** §4.3 Eq.(4)，p.5 只替换一个 agent 的动作，固定数据中的队友动作，构建集中式 Q 的保守项。为避免与 BRPO 的行为符号冲突，下式将论文 proposal \(\mu_i\) 改记为 \(\nu_i\)，将论文行为 \(\beta\) 改记为 \(\mu_{\mathcal D}\)：

```latex
\mathcal L_Q
=\mathcal L_{\mathrm{TD}}
+\alpha
\left[
\sum_{i=1}^{N}\lambda_i
\mathbb E_{s\sim\mathcal D,\,
a_i\sim\nu_i(\cdot\mid s),\,
\mathbf a_{-i}\sim\mu_{\mathcal D,-i}(\cdot\mid s)}
Q(s,a_i,\mathbf a_{-i})
-
\mathbb E_{(s,\mathbf a)\sim\mathcal D}Q(s,\mathbf a)
\right],
\qquad
\lambda_i\ge0,\quad\sum_i\lambda_i=1.
```

符号：\(\mathcal L_Q\) 是 Critic 损失；\(\mathcal L_{\rm TD}\) 是 TD 回归误差；\(Q\) 是集中式联合动作价值；\(\mathcal D\) 是离线数据集；\(\nu_i\) 是用于保守项的动作 proposal；\(\mu_{\mathcal D,-i}\) 表示数据队友动作分布；\(\alpha\) 是保守系数；\(\lambda_i\) 是各 agent 保守贡献权重；其余动作和 agent 符号同前。

**论文事实。** §4.4 Eq.(7)，p.7 给出局部动作 log-sum-exp 形式；Eq.(9) 按各 agent 的策略/行为偏移量选择 softmax 权重。离散版本 CFCQL-D 使用 QMIX，连续版本 CFCQL-C 使用 MADDPG（Algorithm 1）。因此反事实保守项本身不强制使用 IGM mixer。

**关键定理限制。** §4.2，p.5 明确把目标策略、proposal 和行为策略都设为各 agent 分布的乘积；§4.3 的 divergence 比较及 agent-count 结论建立在该设定上。Theorem 4.3/4.4 还假设各局部行为概率有正下界。不能将这些保证无条件推广到有相关模式的混合行为数据。

**研究推断。** 该正则可作为降低 Q 外推风险的工具。它评价“一个动作发生替换、其余来自数据”的切片，无法仅凭此保证“全部来自最新 Actor 的动作组合”得到准确约束。它也不等价于 Actor 的联合 reverse KL，更不能替代协同模式相容性模型。若为新方案添加该项，应作为单独 ablation，而非声称沿用了原定理。

## 可交给主设计的结论

三个可分别借鉴的要点为：OMIGA 的全局目标—局部信用一致性；AlberDICE 的共同正则目标及当前队友校正；CFCQL 的低维反事实价值保守估计。它们分别解决不同环节，直接拼接不会自动产生统一保证。非单调 Q、相关联合行为模型与最新队友更新需要重新明确同一个 Actor 目标及其完整梯度。
