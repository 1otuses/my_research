# BRPO：非 IGM 价值分解、结构化行为约束与一致的策略更新

日期：2026-10-20。状态：文献核查、公式推导和代码静态审计；算法尚未实现，性能提升尚未验证。本文记录新候选方向，保留此前 MADiff 方案和 Word 原稿。

## 结论与候选方案

可以在 Actor–Critic 中使用非单调的非线性价值分解。执行动作由局部 Actor 输出，不依赖每个局部效用的贪心最大化，因此无需强制 IGM。FACMAC-nonmonotonic 已提供这一设计的先例，不能将“去掉 IGM”本身作为新贡献。[FACMAC](https://arxiv.org/abs/2003.06709)

推荐首先检验“非单调分解 Critic＋正规化的稀疏条件行为模型＋完整联合正则目标”。控制复杂度的对象是行为模型中每个动作的条件依赖数量，而不是把联合 KL 换成独立边缘 KL。Q 与行为约束在同一个最新联合动作上计算，并统一进入一个 Actor 目标。

预期收益的假设是：保留必要的动作依赖、减少不必要的条件维度，并保留完整梯度，可以比独立约束和局部截断 score 提供更可靠的更新。模型逼近误差、非凸优化、局部可观测性和高阶协作均可能使该假设失败。

## 1. 数据、执行条件与符号

固定离线数据集记为 \(\mathcal D\)。每条 episode 包含同步的联合观察 \(\mathbf o_k\)、联合动作 \(\mathbf a_k\)、团队即时奖励 \(r_k\)、结束标记 \(d_k\)，以及可选全局状态 \(s_k\)。\(k\) 是环境时间步；若使用扩散，\(t\in[0,T]\) 是扩散时间，\(T\) 是其上限。

```latex
h_{i,k}=(o_{i,k-L+1},\ldots,o_{i,k}),
\qquad
\mathbf h_k=(h_{1,k},\ldots,h_{N,k}),
\qquad
\pi_\theta(\mathbf a\mid\mathbf h)
=\prod_{i=1}^{N}\pi_{\theta_i}(a_i\mid h_i).
```

其中，\(N\) 是智能体数量，\(i\) 是智能体索引，\(o_{i,k}\) 是局部观察，\(h_{i,k}\) 是长度 \(L\) 的本地观察历史，\(\mathbf h_k\) 是联合历史，\(a_i\) 是个体动作，\(\mathbf a=(a_1,\ldots,a_N)\) 是联合动作；\(\theta_i\) 是局部 Actor 参数，\(\theta\) 收集所有 Actor 参数。第一版可取 \(L=1\)，与 OMSD 当前实现一致；部分可观测性严重时再增加记忆，短窗口的 Markov 性不作保证。

乘积分解以所有局部历史已给定、各 Actor 采样随机性独立为条件。执行时只有本地 \(h_i\)。训练期的联合模型读取 \(\mathbf h\)，不把联合观察冒称为环境全局状态，也不加入未来观察或事后奖励作为行为条件。

\(\mu(\mathbf a\mid\mathbf h)\) 表示离线数据诱导的联合行为分布；\(\widehat\mu_\psi\) 是参数为 \(\psi\) 的估计。本文不再使用含义宽泛的“上下文 \(c\)”。

第一版使用具有可计算概率密度的随机 Actor，例如 tanh 变换的高斯策略；实际实现需包含变换的 Jacobian。连续动作上的 Dirac Actor 对连续行为密度的 KL 通常为无穷；若保留 OMSD 确定性 Actor，\(Q+\lambda\log\widehat\mu\) 只能称为行为正则代理，不能宣称精确优化下式。

```latex
J(\theta)
=\mathbb E_{\mathbf h\sim\mathcal D}
\left[
\mathbb E_{\mathbf a\sim\pi_\theta(\cdot\mid\mathbf h)}
Q_\phi(\mathbf h,\mathbf a)
-\lambda D_{\mathrm{KL}}
\left(\pi_\theta(\cdot\mid\mathbf h)\|
\widehat\mu_\psi(\cdot\mid\mathbf h)\right)
\right],
\qquad \lambda=1/\beta>0.
```

\(Q_\phi\) 是参数为 \(\phi\) 的团队价值估计，\(J\) 是待最大化的固定数据上下文策略改进目标，\(\lambda\) 是正则强度，\(\beta\) 为其倒数；\(D_{\mathrm{KL}}\) 是 reverse KL。目标是数据分布上的策略改进代理，不等于已解决执行时状态占用偏移。有限 KL 还要求 Actor 的分布被行为参考覆盖。

## 2. 非 IGM 的非线性价值分解

### 2.1 最小可实现结构

```latex
z_i=E_{\phi_i}(h_i,a_i),
\qquad
g=E_{\phi_g}(\mathbf h),
\qquad
Q_\phi^{\mathrm{tot}}(\mathbf h,\mathbf a)
=M_{\phi_m}(g,z_1,\ldots,z_N).
```

\(E_{\phi_i}\) 是个体观察—动作编码器，输出 \(d_z\) 维向量 \(z_i\)，\(d_z\) 为嵌入宽度；\(E_{\phi_g}\) 是联合历史编码器，输出集中式特征 \(g\)；\(M_{\phi_m}\) 是不施加正权/单调限制的非线性 mixer，输出一个团队 Q 标量；\(\phi\) 包含上述全部参数。向量 \(z_i\) 是特征，不直接解释为“真实个体价值”。

若需要最小的标量效用消融，可让各编码器输出标量 \(u_i\)，再令 \(Q^{\mathrm{tot}}=M(g,u_1,\ldots,u_N)\)。但标量瓶颈可能丢失动作信息，向量版本应作为独立结构选项记录，不能将两者混称。

输入输出规格：batch 大小 \(B\)，局部历史张量 \([B,N,L,d_o]\)，联合动作 \([B,N,d_a]\)，编码张量 \([B,N,d_z]\)，双 Critic 输出各为 \([B,1]\)。\(d_o\) 是观察维度，\(d_a\) 是动作维度；异构智能体采用各自维度及映射，不强行截断。两套 Q 使用独立参数。

不要求 \(\partial Q^{\mathrm{tot}}/\partial u_i\ge0\)。局部 Actor 输出动作，所有因子/mixer 只在训练中提供评价和梯度，因此不需要从 \(u_i\) 独立取 argmax 来执行。[FACMAC 论文与代码](https://github.com/oxwhirl/facmac/blob/main/src/modules/mixers/qmix_ablations.py)

自身动作对团队价值的梯度经过完整 mixer 传播：

```latex
\nabla_{a_i}Q^{\mathrm{tot}}
=
\left(\frac{\partial z_i}{\partial a_i}\right)^{\!\top}
\nabla_{z_i}M_{\phi_m}(g,z_1,\ldots,z_N).
```

\(\partial z_i/\partial a_i\) 是个体编码器对自身动作的 Jacobian，\({}^\top\) 表示转置，\(\nabla_{z_i}M\) 是 mixer 对该嵌入的梯度。此式假定 \(g\) 只依赖历史、每个 \(z_j\) 只依赖对应 \(a_j\)，与上面的结构一致。其他动作固定，但 mixer 梯度仍取决于当前所有队友嵌入。

具体反例：两智能体二值动作时，取 \(u_1=a_1,u_2=a_2\)，\(M(u_1,u_2)=u_1+u_2-2u_1u_2\)，可表达 XOR：联合动作 01、10 价值为 1，00、11 为 0。此时对 \(u_1\) 的偏导为 \(1-2u_2\)，会改变符号。它证明结构可表达此类非单调关系，不证明学习一定成功。

### 2.2 明确交互的可选结构

```latex
Q_\phi^{\mathrm{tot}}(\mathbf h,\mathbf a)
=b_\phi(\mathbf h)
+\sum_i f_{\phi_i}(\mathbf h,a_i)
+\sum_{(i,j)\in\mathcal E_Q}
f_{\phi_{ij}}(\mathbf h,a_i,a_j).
```

\(b_\phi\) 是只依赖联合历史的标量项，\(f_{\phi_i}\) 是个体动作因子，\(f_{\phi_{ij}}\) 是双智能体动作交互因子，\(\mathcal E_Q\) 是价值交互边集合，\(i,j\) 为两个智能体。各 \(f\) 可以由非线性网络实现；最后的加法并不使它退化为只含个体项的 VDN。该想法与协调图的 payoff factorization 有直接先例。[DCG](https://proceedings.mlr.press/v119/boehmer20a.html)

若每个节点的邻居数有界，动作交互项数量可以随 \(N\) 近似线性增长；若使用全连接图则为二次。只有 pairwise 因子的模型不能普遍表达三阶及更高阶动作交互，可能需要小型高阶因子或全局残差。DCG 的运行时消息传递不纳入本方案，执行仍由本地 Actor 完成。

### 2.3 分解与信用分配的区别

团队回报监督通常不能唯一确定每个因子的物理贡献。给某因子加常数、给另一因子减常数，可能不改变总 Q。因此不将因子数值直接称为真实或因果信用。

采用反事实的操作性信用定义：

```latex
A_i^Q(\mathbf h,\mathbf a)
=Q_\phi(\mathbf h,\mathbf a)
-\mathbb E_{\widetilde a_i\sim\rho_i(\cdot\mid h_i)}
Q_\phi(\mathbf h,\widetilde a_i,\mathbf a_{-i}).
```

\(A_i^Q\) 表示保持队友动作不变时，自身动作相对参考的团队价值贡献；\(\mathbf a_{-i}\) 收集其他智能体动作；\(\widetilde a_i\) 是替代动作；\(\rho_i\) 是选定参考分布，第一版可用当前局部策略的冻结快照。该量依赖参考分布和 Q 精度，不等于环境中的因果效应。它借鉴 COMA 的反事实 baseline，连续动作可通过有限采样估计。[COMA](https://arxiv.org/abs/1705.08926)

使用 score-function 策略梯度时，baseline 必须与本次实际 \(a_i\) 无关，且对其停止梯度；使用重参数化梯度时，直接对联合目标求导即可，不能再把反事实项任意相减来改变梯度。

### 2.4 Critic 训练先采用 IQL

```latex
\mathcal L_V
=\mathbb E_{\mathcal D}
\left[
\left|\tau-\mathbf1\{u<0\}\right|u^2
\right],
\qquad
u=\min_{\ell\in\{1,2\}}
Q_{\bar\phi_\ell}(\mathbf h,\mathbf a)-V_\nu(\mathbf h),

\mathcal L_Q
=\frac12\sum_{\ell=1}^{2}
\mathbb E_{\mathcal D}
\left[
\left(
Q_{\phi_\ell}(\mathbf h_k,\mathbf a_k)
-r_k-\gamma(1-d_k)V_\nu(\mathbf h_{k+1})
\right)^2
\right].
```

\(V_\nu\) 是参数为 \(\nu\) 的集中式状态/历史价值网络，\(\tau\in(0,1)\) 是 expectile 参数，\(\mathbf1\) 是指示函数，\(u\) 是 expectile 残差；\(\ell\) 是双 Critic 索引，\(\bar\phi_\ell\) 是目标 Q 参数；\(\gamma\) 是折扣率；\(r_k\) 必须是即时团队奖励，若数据只有整条回报则无法直接使用该 TD 目标；\(d_k\) 是禁止 bootstrap 的真实终止标记，time-limit truncation 必须按环境协议另行处理。目标项计算时停止梯度。

这些损失只查询数据内训练动作，但后续 Actor 在新动作上的 Q 查询仍可能外推。IQL 预训练所得 Q 是用于策略提取的价值代理，不应直接宣称始终等于每次更新后的 \(Q^{\pi_\theta}\)。第一版冻结 Critic 以便对照 OMSD，后续如改策略评估方法，需单独验证。[IQL](https://arxiv.org/abs/2110.06169)

## 3. 如何折中联合 KL 与个体更新

### 3.1 精确关系：个体条件约束保留联合信息

固定 \(\mathbf h\)，对乘积目标策略有：

```latex
D_{\mathrm{KL}}(\pi_i\pi_{-i}\|\mu)
=
\mathbb E_{\mathbf a_{-i}\sim\pi_{-i}}
D_{\mathrm{KL}}
\left(
\pi_i(\cdot\mid h_i)
\middle\|
\mu(\cdot\mid\mathbf h,\mathbf a_{-i})
\right)
+
D_{\mathrm{KL}}(\pi_{-i}\|\mu_{-i}).
```

式中为简洁省略部分共同历史条件；\(\pi_i=\pi_{\theta_i}\)，\(\pi_{-i}\) 为其他局部目标策略的乘积，\(\mu_{-i}\) 是联合行为分布关于队友动作的边缘分布，\(\mu(a_i\mid\mathbf h,\mathbf a_{-i})\) 是给定所有当前队友动作的完整行为条件分布。

固定队友策略更新 \(\pi_i\) 时，最后一项与 \(\theta_i\) 无关。因此“个体更新”可以与“联合 KL”严格相容，关键是使用行为条件分布而非自身边缘分布。全条件仍可能昂贵，下一步才引入结构近似。

### 3.2 第一版选择正规化树形行为模型

```latex
\widehat\mu_{\psi,\mathcal T}(\mathbf a\mid\mathbf h)
=
p_{\psi_r}(a_r\mid\mathbf h)
\prod_{j\ne r}
p_{\psi_j}(a_j\mid\mathbf h,a_{\operatorname{pa}(j)}).
```

\(\mathcal T\) 是固定根的有向树，\(r\) 是根节点，\(\operatorname{pa}(j)\) 是节点 \(j\) 的父节点；各 \(p_{\psi_j}\) 是正规化的条件概率密度，\(\psi\) 包含其全部参数。树有 \(N-1\) 条边，每个非根因子只以一个队友动作为条件。对每个 \(\mathbf h\) 顺序积分叶节点即可证明整体归一化。

这是对真实行为依赖的模型假设，不是任意 \(\mu\) 的无损分解。若条件依赖不能用树表达，使用有界父集的 DAG 或高阶因子。图结构在第一版固定、仅随观察决定的结构选择作为后续工作，避免把图搜索和算法改进混在一起。

同一树上根方向的变化在“精确且一致的同一树分布”下只是表示变化；有限样本分别拟合的条件网络会有差异。行为因子顺序不等于 Actor 优化顺序。

树上的一轮密度评价只有 \(N\) 个局部因子；若联合历史编码共享且编码复杂度受控，因子评价部分是线性的。若联合历史编码用全连接 attention，编码本身仍可能为 \(O(N^2)\)。这是计算结构控制，不能证明统计样本复杂度不随维度恶化。

可以使用可显式求 `log_prob` 的条件混合密度网络或条件 flow。它们是第一版检验精确目标的工具；OMSD 对 GMM/MAF 的消融表现并非普遍更优，因此不能预设换模型即提升回报。应以留出数据似然、模式恢复、真实闭环回报共同判断。[OMSD D.7.5](https://arxiv.org/html/2505.05968v3#A4.SS7.SSS5)

### 3.3 必须计算完整联合梯度

```latex
\nabla_{a_i}\log\widehat\mu_{\psi,\mathcal T}(\mathbf a\mid\mathbf h)
=
\nabla_{a_i}\log p_{\psi_i}(a_i\mid\mathbf h,a_{\operatorname{pa}(i)})
+
\sum_{j:\operatorname{pa}(j)=i}
\nabla_{a_i}\log p_{\psi_j}(a_j\mid\mathbf h,a_i).
```

根节点的第一项改为 \(\nabla_{a_r}\log p_{\psi_r}(a_r\mid\mathbf h)\)。第二项遍历 \(i\) 的所有子节点，表达自身动作作为子节点条件时造成的概率变化。队友动作数值固定，不代表这些条件项对 \(a_i\) 的导数为零。

对结构化联合模型做一次完整 `log_prob` 计算并自动求导，可自然保留上述两类项。分别训练全条件网络后简单求和通常是 pseudo-likelihood，不应未经兼容性检查就声称等价于某个正规化联合 KL。

OMSD v3 明确其局部坐标正则只使用自身条件 score，将后续因子作为固定参考。因此该完整梯度路线应与 OMSD 原局部 score 单独对照，不能将自身的完整目标假定为 baseline 已实现的目标。[OMSD §3.2](https://arxiv.org/html/2505.05968v3#S3.SS2)

### 3.4 与 copula 的关系和高阶边界

条件 copula 可以作为某些条件密度因子的实现方式，但保留完整 log-density 梯度的要求不变。先前所讨论的边缘密度比只有在各边缘采用相同集中式条件、且真正属于同一联合分布时才具有精确的 copula 解释。

最小反例：三智能体二值动作均匀分布在 000、011、101、110 上，各单变量及所有二变量边缘都与独立均匀分布相同，三阶 parity 依赖仍然存在。只用这些真实单/双变量边缘建立的树分布变成全部八种组合均匀，和原分布的总变差距离为 0.5。因此即使模型拟合完全准确，固定 pairwise/tree 方案也存在明确表达上限。

## 4. 三篇参照论文的可借鉴点与适用条件

| 来源 | 已核对的机制 | 可以借鉴的部分 | 不可直接迁移的部分 |
| --- | --- | --- | --- |
| OMIGA | 正权线性 value mixer 与行为乘积分解假设下，推导 global-to-local 正则 | 对照全局目标与局部更新的推导方式；in-sample 价值训练 | 不能把其等价结论用于一般非单调 mixer 与相关行为混合 |
| AlberDICE | 联合折扣占用分布正则，交替策略改进；条件队友行为模型和重要性修正 | 最新队友策略适应、联合支持与占用偏移意识 | 占用分布 KL 不等于固定历史下的动作 KL；IS 权重可能高方差 |
| CFCQL | 固定数据队友动作，只改变自身动作，构造 Critic 反事实保守惩罚 | 控制保守惩罚维度；作为独立 Critic 消融 | 不是 Actor 的完整联合 KL；理论采用的策略分解假设需单独检查 |

参见 [OMIGA](https://arxiv.org/abs/2307.11620)、[AlberDICE](https://arxiv.org/abs/2311.02194)、[CFCQL](https://arxiv.org/abs/2309.12696)。三者的详细公式、页码和边界见[配套文献笔记](2026-10-20-brpo-source-findings.md)。

纠正此前可能造成的误解：AlberDICE 的 \(\pi^D_{-i}(\mathbf a_{-i}\mid s,a_i)\) 是依赖自身动作的条件队友模型；将联合占用写成自身边缘乘以该条件项，本身不构成行为独立假设。不能直接沿用其他论文对此的概括而忽略其原始公式。

## 5. Q 与行为约束如何对齐

### 5.1 使用同一个标量改进目标

```latex
S_{\phi,\psi}(\mathbf h,\mathbf a)
=Q_\phi(\mathbf h,\mathbf a)
+\lambda\log\widehat\mu_{\psi,\mathcal T}(\mathbf a\mid\mathbf h),

J(\theta)
=\mathbb E_{\mathbf h,\mathbf a\sim\pi_\theta}
\left[
S_{\phi,\psi}(\mathbf h,\mathbf a)
-\lambda\sum_i\log\pi_{\theta_i}(a_i\mid h_i)
\right].
```

\(S_{\phi,\psi}\) 是价值与行为密度合成的动作评分，不是额外的环境奖励或独立训练的新 Critic；期望中的 \(\mathbf h\) 来自 \(\mathcal D\)。第二项保留了随机 Actor 的熵，删除它会改变 BRPO 目标。

所有 Q 与行为模型必须使用同一条数据的联合历史、同一组当前联合动作、同一动作坐标和归一化。行为模型参数冻结，仍需保留其输入动作上的计算图。若模型在变换后的坐标定义密度，应把 Jacobian 纳入原动作空间密度，特别是非线性变换时。

Q 网络参数与行为网络参数不必共享。价值依赖图与统计行为依赖图也未必相同：共同状态可以产生相关性，奖励可能依赖高阶动作组合。第一版可以共享人为选定的候选边，但必须设置“不共享拓扑”的对照，不能把两图相同当作理论事实。

### 5.2 同一目标下的信用分配

```latex
A_i^{\mathrm{reg}}(\mathbf h,\mathbf a)
=
S_{\phi,\psi}(\mathbf h,\mathbf a)
-
\mathbb E_{\widetilde a_i\sim\rho_i(\cdot\mid h_i)}
S_{\phi,\psi}(\mathbf h,\widetilde a_i,\mathbf a_{-i}).
```

\(A_i^{\mathrm{reg}}\) 是正则化后的反事实贡献，\(\rho_i\) 与替代动作含义同第 2.3 节。这里替换自身动作时同时重新评价 Q 和联合行为密度；不能用当前队友动作评价 Q，却用另一组历史队友动作评价正则。

若采用 score-function 梯度，基于 \(A_i^{\mathrm{reg}}\) 的估计还必须包含熵项：

```latex
\nabla_{\theta_i}J
=
\mathbb E
\left[
\nabla_{\theta_i}\log\pi_{\theta_i}(a_i\mid h_i)
\left(
A_i^{\mathrm{reg}}(\mathbf h,\mathbf a)
-\lambda\log\pi_{\theta_i}(a_i\mid h_i)
\right)
\right].
```

期望按同一当前乘积策略采样；固定其他 Actor、Q、行为模型，baseline 停止梯度；\(\rho_i\) 不依赖本次实际自身动作。省略的熵导数常数项期望为零。连续可重参数化 Actor 优先直接对第 5.1 节的 Monte Carlo 目标求导，反事实优势用于诊断和另一种估计器对照，不重复叠加。

### 5.3 最新队友动作与更新稳定性

一轮坐标更新开始时采样当前所有 Actor 的动作；更新 \(i\) 时保持其他动作数值固定，对自身动作及完整 \(S\) 求导。完成该 Actor 的优化步后，重新计算其动作，供后续 Actor 使用。这里“最新目标策略”指正在学习的 \(\pi_{\theta_j}\)，不是 TD 目标网络的延迟副本。

每个 Actor 改变后，Q 和行为项都使用刷新后的同一联合动作。可增加相邻迭代策略的 KL 步长约束作为独立可选机制。它控制更新幅度，不能保证回到数据支持内；同样，优化共同目标也不能保证非凸问题全局收敛或所有角色的动作方向相同。

正则强度通过固定奖励尺度和统一验证预算选择。逐样本把 Q 梯度归一化后再与行为 score 相加，一般不再对应固定 \(\lambda\) 的上述标量目标；若沿用 OMSD 的梯度归一化选项，应明确作为另一种更新规则对照。

## 6. 初步实现路径与代码基础

### 6.1 首选 OMSD 工程，参考 FACMAC 的 Critic

2026-10-20 静态核查的 OMSD 官方 `main` commit：

`6290d437437a7a5f49d6a5d9adb6712c5dbcd2a2`。

Git clone 遇到网络连接重置；使用 GitHub API 读取 commit/tree，并从该 SHA 的 raw 文件核查以下路径。未安装训练环境，未运行 baseline。

| 模块 | 已核查入口 | 候选修改 |
| --- | --- | --- |
| 双 Q 与网络 | `algorithms/model.py:39` 的 `TwinQ` | 替换内部单体 MLP 为非单调 mixer/交互因子，保留 `both(action, condition)` 接口 |
| IQL | `omsd/agents.py:48` 的 `IQLCritic` | 接入新双 Q，先保持 IQL 损失及数据协议 |
| 行为条件 | `omsd/agents.py:277` 的 `_behavior_condition` | 新增共同联合历史条件的结构化行为模型，保留原配置作对照 |
| 行为预训练 | `update_behavior` / `pretrain_behavior.py` | 新增显式 `log_prob` 的条件密度训练入口 |
| Actor 更新 | `omsd/agents.py:343` 的 `update_policy` | 完整联合目标、随机 Actor、更新后刷新动作、相同 Q/正则样本 |
| 数据与环境 | `omsd/data.py`、`omsd/environments.py` | 复用已有 MPE / MADiff-OG-MARL 数据适配，逐项核对动作与观察格式 |
| 执行 | `omsd/agents.py:400` 的 `act` | 只使用各自局部观察；历史扩展作为独立改动 |

[固定版本代码](https://github.com/qiaodan-cuhk/OMSD/tree/6290d437437a7a5f49d6a5d9adb6712c5dbcd2a2)；[FACMAC 非单调 mixer](https://github.com/oxwhirl/facmac/blob/main/src/modules/mixers/qmix_ablations.py)。

FACMAC 本身是在线 Actor–Critic 工程，提供有价值的分解与梯度实现参考，但离线数据、行为模型和预训练流程需额外加入。OMIGA 的现有推导与 mixer 限制更强，不适合直接承载本轮非 IGM 假设。MADiff 继续作为独立轨迹规划路线；把本方案强行接入其逆动力学规划流程会增加无关改动。

### 6.2 必须先记录的实现差异

当前官方 `omsd/agents.py:348–388` 在一轮更新开始缓存各 Actor 动作；`optimizer.step()` 之后把步前 forward 得到的 `action.detach()` 写回缓存，没有重新运行更新后的 Actor。这与论文 v3 关于使用本轮已更新前序动作的表述有差异。此结论来自静态源代码，尚未用训练复现量化影响。

当前 `_behavior_condition` 使用本地观察加前序动作，而不是每个因子共同条件于整个联合观察。该模型作为条件 DAG 仍可正规化，但相对于给定完整联合观察的真实行为链，包含额外的条件信息删减假设。

因此分别保留“发布代码 baseline”和“动作刷新/共同条件改动”的单项对照，避免把实现行为差异混入模型创新收益。新模型不默认加载原结构 checkpoint。

### 6.3 训练阶段

1. 数据审计：确定动作类型与范围、同步 agent-id、即时团队奖励、真实终止与截断、历史 padding、同 episode 切分。训练/验证按 episode 切分。
2. 预训练非单调双 Q 和 V；在同一数据上拟合树形行为模型。预训练互相独立，Actor 阶段冻结。
3. 初始化局部随机 Actor；原确定性 OMSD 同时保留。可用局部 BC 作初始化，不能把 BC 初始联合支持当作已保证。
4. 用同一当前联合动作评价 Q 和完整联合密度，按坐标更新并刷新动作。行为因子图和 Actor 更新顺序分开设置。
5. 闭环评估严格本地输入，记录回报、碰撞、覆盖、联合模式、跨种子稳定性、计算时间。

## 7. 最小实验与失败判据

先做两智能体单步任务：准确 Q 对照学习 Q，独立边缘约束对照树形完整约束，局部截断梯度对照完整梯度。加入不等价模式、样本比例偏斜及不同初始点，不能只测试完全对称双峰。

再做三智能体 parity，验证模型已知表达边界。最后进入 MPE simple_spread 的一致动作版本和固定数据导出，至少覆盖两种质量，并使用多训练种子。原数据为离散/one-hot 动作时不能直接用连续 Gaussian Actor 运行，需单独设计 categorical 版本或使用匹配的连续数据。

| 对照 | 要回答的问题 |
| --- | --- |
| 发布代码 / 更新后刷新动作 | 最新队友动作本身是否改变结果？ |
| 原 Actor / 同结构随机 Actor | 完整 KL 的 Actor 改动是否构成主要混杂？ |
| 原 Q / 参数量匹配的非单调 Q | 分解结构是否比单纯容量增加更有效？ |
| 独立行为 / 树形行为 / 全条件行为 | 依赖表达与规模之间的折中是否存在？ |
| 自身条件 score / 完整 log-density 梯度 | 子因子梯度是否影响协同与回报？ |
| 单体 Q / 分解 Q × 独立约束 / 联合约束 | Q 与约束是否存在互补收益？ |

先做消融定位，不要求立即把所有组合铺成大规模训练。每个比较保持数据、信息、种子、更新数及超参数搜索预算一致；模型参数量与实际耗时另行报告。

行为似然只能说明模型拟合，不能代替环境回报或可靠 OOD 检测。准确 Q 下仍失败，优先审查优化/协同/可实现性；只在学习 Q 下失败，优先审查 Critic 外推。树形模型在 parity 失败属于结构边界；不能靠调参隐去。

本文方案不保证任意数据分布下回报单调提升、无 OOD、全局最优或顺序不敏感。现阶段可争取的贡献是验证受控复杂度的联合正则如何与非 IGM Critic 的信用信号对齐，并说明适用条件。

如果相同本地历史对应的不同联合情境要求互斥动作，局部 Actor 的可实现性本身也会受限。集中式行为模型或更强 Critic 不能赋予 Actor 执行时缺失的信息。保留本地历史及固定角色编码有时有帮助，但不构成普遍解决。

## 8. 本轮验证与交付

配套 `research/checks/brpo_design_checks.py` 使用枚举与有限差分验证：

1. 非单调 mixer 可表达 XOR。
2. 联合 KL 的个体条件分解成立，示例两侧误差约 \(10^{-16}\)。
3. 正规化树形因子联合质量之和为 1。
4. 完整联合动作梯度含子因子贡献：示例为 -0.824，自身因子单独为 -0.2；有限差分吻合完整值。
5. 同一正则化评分的反事实中心化条件期望为 0。
6. parity 的真实 pairwise 边缘不足以恢复联合分布，树重建总变差为 0.5。

这些是公式与反例检查，不是算法训练实验，也不支持性能提升声明。

## 一手来源

- [FACMAC](https://arxiv.org/abs/2003.06709)：非单调因子 Critic 与集中式策略梯度先例。
- [DCG](https://proceedings.mlr.press/v119/boehmer20a.html)：局部交互 payoff 分解。
- [COMA](https://arxiv.org/abs/1705.08926)：反事实 baseline 的信用分配思想。
- [IQL](https://arxiv.org/abs/2110.06169)：数据内价值学习与策略提取。
- [OMIGA](https://arxiv.org/abs/2307.11620)：特定结构下全局到局部正则。
- [AlberDICE](https://arxiv.org/abs/2311.02194)：占用分布正则与交替策略优化。
- [CFCQL](https://arxiv.org/abs/2309.12696)：Critic 的反事实保守约束。
- [OMSD v3](https://arxiv.org/html/2505.05968v3)：条件 score 正则和本文 baseline。
- [Copula-MAIL](https://arxiv.org/abs/2107.04750)：边缘行为与依赖结构的分离建模。
