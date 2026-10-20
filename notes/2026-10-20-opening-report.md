## 一 研究目的与意义

多机器人协作、智能交通与分布式控制等任务需要多个决策主体在局部信息条件下共同完成团队目标。多智能体强化学习（Multi-Agent Reinforcement Learning，MARL）通过与环境交互学习协作策略，集中式训练与去中心化执行（Centralized Training with Decentralized Execution，CTDE）允许训练阶段利用联合信息，执行阶段仅依赖本地输入，为合作决策提供了常用技术框架。MADDPG与MAPPO等方法展示了该范式在多智能体控制中的有效性[1,2]。

在线MARL通常依赖大量环境交互与策略试错。在真实机器人、工业系统等场景中，采样成本、安全要求和多主体同步运行限制了反复在线训练。离线强化学习利用预先收集的固定数据训练策略，为复用历史经验提供了途径；但策略优化可能查询数据分布之外的动作，使价值误差被放大[3,4]。这一困难在多智能体环境中进一步表现为联合动作分布偏移：每个动作单独合理，组合后却可能违背数据中的协同关系。

现有离线数据往往由不同训练阶段、不同质量和不同分工模式的团队策略混合构成。分别模仿各智能体的边缘行为，可能把不同模式中的局部动作重新组合为低价值联合行为。只依赖保守价值估计，也难以充分回答智能体应当如何适应正在更新的队友。本课题拟研究严格CTDE条件下的行为依赖表示与价值引导问题，使局部Actor的更新同时考虑团队收益、当前队友策略及离线数据的联合支持。

研究将围绕稀疏树形联合行为建模、非单调非线性价值因式分解及二者的联合优化展开。理论上，尝试厘清个体行为约束、联合支持与信用分配的关系；方法上，探索表达能力与学习复杂度之间的折中；应用上，为固定数据条件下的协作控制提供可复现的算法与评测依据。预期收益需要通过对照实验验证，不预设模型结构变化必然带来回报提升。

## 二 国内外研究现状与发展趋势

## 1 在线协作学习与价值分解

在线合作MARL的研究主要围绕非平稳性、部分可观测性和信用分配展开。COMA通过集中式Critic和反事实baseline评价个体动作的团队贡献[5]；VDN将团队价值表示为个体效用之和[6]；QMIX利用非负混合权重建立局部效用与联合价值的单调关系，使局部贪心动作与联合最大化相容[7]。这些方法分别从梯度估计和价值结构出发支持去中心化执行。

价值分解的表达能力取决于所施加的结构限制。多最优模式或非单调交互任务中，要求整个联合最优集合由局部最优集合的笛卡尔积表示，可能造成信息损失。Actor–Critic的执行动作由局部Actor直接输出，因此Critic可以采用更一般的分解。FACMAC在集中式策略梯度框架中研究了非单调因子Critic[8]，表明不强制IGM的价值表示具有方法依据，但此类结构仍需解决优化和数据覆盖问题。

## 2 从单智能体离线学习到离线MARL

单智能体离线学习形成了行为约束、保守价值学习和数据内学习等路径。BRAC系统研究了对策略偏离行为分布施加正则的设计[9]。CQL通过保守价值正则降低策略利用高估值的风险[10]；IQL通过数据内动作的expectile价值学习和优势加权策略提取减少对未见动作的查询[11]。这些设计为离线MARL提供了基础，但联合动作空间随智能体数量增长，误差、模式错配与队友策略变化会相互影响，直接套用单智能体方案存在局限。

MA-ICQ通过隐式约束将价值估计重点放在数据支持的状态—动作对上[12]；OMAR结合一阶策略梯度与零阶搜索缓解保守价值函数上的局部优化困难[13]。CFCQL固定数据中的队友动作，仅替换一个智能体动作，构造反事实保守惩罚，以减轻联合保守估计的过度悲观[14]。这些工作主要改善价值学习或策略搜索，不能仅凭个体保守性保证最新Actor产生的联合动作相容。

OMIGA从全局隐式价值正则推导局部学习目标，强调价值结构与策略提取的一致性；其等价转换依赖特定的正权混合及行为分解条件[15]。AlberDICE对联合状态—动作占用分布施加正则，并通过条件队友模型和交替优化适应当前队友策略[16]；ComaDICE进一步研究合作离线学习中的占用分布偏移与价值分解[17]。占用分布正则同时涉及状态访问，与固定状态上的动作KL约束应区别讨论。

BRUD相关研究指出，对数据中历史队友的最佳响应不一定适用于更新后的队友，协同失败具有独立于简单价值高估的优化成因[18]。因此，当前研究逐步从“各智能体动作是否接近数据”转向“联合策略是否遵循可行协作模式，以及如何适应当前队友”。即使Critic准确，互不相容的模式选择仍可能降低团队回报。

## 3 扩散模型在离线决策中的不同作用

去噪扩散概率模型通过逐步加噪与学习反向去噪过程拟合复杂分布[19]，基于score的生成建模将这一过程与密度梯度联系起来[20]，classifier-free guidance则为条件引导提供了常用机制[21]。离线数据中的行为分布常具有多峰结构，这使扩散模型成为表达多种可行行为的重要工具。其在决策系统中的作用可按模型位置分为以下几类。

第一类将扩散模型用作轨迹规划器。Diffuser通过生成轨迹开展规划[22]；Decision Diffuser以回报等条件生成状态序列，并结合逆动力学恢复动作[23]。MADiff将注意力引入多智能体扩散，研究集中式控制与去中心化规划两种执行形式[24]。这条路径可以表达时间与智能体间依赖，但面临采样开销、队友建模偏差以及生成计划与实际执行之间的误差。

第二类把扩散模型作为动作策略。Diffusion-QL利用扩散策略表达多模态行为，并结合Q引导进行策略优化[25]。第三类仅在训练中利用扩散行为模型提供score正则。SRPO将行为score用于约束策略梯度，提取推理成本较低的Actor[26]。两类方法的区别在于执行动作是否需要迭代去噪，研究设计应分别报告训练和执行成本。

第四类将扩散模型作为数据生成或增强模块。CODA根据当前联合策略生成合成轨迹，以促进离线学习中的策略相互适应[27]。近期OMSD则以条件扩散模型估计顺序分解的行为score，结合集中式Q引导局部策略更新[28]。其结果说明条件行为依赖对模式协调具有价值；但有限数据下的条件估计误差、因子梯度处理及大规模团队的复杂度仍值得研究。扩散表达能力本身不构成联合OOD被消除的保证。

## 4 结构化行为依赖与分布价值表示

在概率建模中，Chow–Liu依赖树以低阶依赖近似高维离散联合分布[29]，条件Chow–Liu模型进一步研究给定外部信息时的结构化分布表示[30]。Copula-MAIL将个体边缘行为与智能体之间的统计依赖分开学习[31]。这些工作为结构化行为克隆提供了基础，但把条件树用于离线MARL的Actor约束及其闭环性能，仍需专门论证。

价值表示方面，协调图以局部和交互payoff因子组织团队价值，提供了表达能力与复杂度之间的折中[32]。分布强化学习通过分位回归描述回报随机性[33]，DFAC研究了多智能体回报分布的因子化表示[34]。行为动作分布与回报价值分布是两个不同对象：前者描述如何行动，后者描述行动可能产生的长期收益。本课题拟在非单调Critic中组织回报分布，并从同一联合行为约束目标提取个体改进信号。

## 三 主要研究内容与拟解决的问题

本研究面向固定离线数据、完全合作及执行无通信的场景。训练期可读取同步联合观察或可用全局状态；执行期每个智能体只读取本地观察及允许的历史。研究重点是：如何用受控复杂度保留行为依赖，如何以非单调结构表示团队价值与信用，以及如何让Q引导和行为约束作用于同一策略更新。

## 1 基于稀疏条件树的联合行为克隆

以智能体为节点，构造具有少量动作依赖的树形行为图，分别学习根动作密度和子节点条件动作密度。模型拟采用共享状态编码器和局部条件密度头；连续动作先使用可计算似然的混合密度或flow验证完整目标，再与扩散score实现比较。树结构首先固定，后续依据训练数据依赖度与留出集拟合质量选择，避免用测试回报选择结构。

统一以s表示理论推导中的集中式条件，包含训练时可用的状态或联合观察；这不意味着局部Actor可以读取全部s。令智能体数量为N，个体动作为$a_i$，联合动作为a=($a_1$,…,$a_N$)，联合行为模型定义为：

```latex
\hat{\mu}_{\psi,\mathcal{T}}(\mathbf{a}\mid s)=p_{\psi_r}(a_r\mid s)\prod_{j\ne r}p_{\psi_j}(a_j\mid s,a_{\operatorname{pa}(j)})
```

式中，μ̂为联合行为密度估计，ψ为全部行为模型参数，$\mathcal{T}$为有根树，r为根智能体索引，j为非根智能体索引，pa(j)为父节点索引，各p均为正规化条件密度。使用同一episode、同一环境时间步的数据动作训练，负对数似然目标为：

```latex
\mathcal{L}_{\mu}(\psi)=-\mathbb{E}_{(s,\mathbf{a})\sim\mathcal{D}}[\log\hat{\mu}_{\psi,\mathcal{T}}(\mathbf{a}\mid s)]
```

其中，$\mathcal{D}$为固定离线数据集，$\mathcal{L}_{\mu}$为行为克隆损失，期望对数据中的状态—联合动作取样。与边缘克隆相比，该模型保留部分动作条件依赖；与完整前序条件相比，其因子动作条件维度受控。树形假设可能遗漏高阶协同，需要通过三智能体奇偶任务及真实环境检验，并在必要时扩展到有界父集结构。

## 2 非单调非线性因子Critic与团队回报分布

各智能体的局部观察—动作编码器输出特征，集中式非线性mixer聚合这些特征，并输出团队回报分位点。该结构不对智能体效用施加非负混合权重；局部动作由Actor产生，因此不依赖局部贪心值与联合最大值一致。分位点维度上的有序约束与智能体维度上的IGM是不同约束。

```latex
z_i=E_{\phi_i}(o_i,a_i),\quad g=E_{\phi_g}(s),\quad q_{\phi,m}(s,\mathbf{a})=M_{\phi_m}(g,z_1,\ldots,z_N)
```

式中，$o_i$为智能体i的局部观察，E为编码器，$z_i$为个体特征，g为集中式特征，φ为Critic参数集合，M为非单调mixer，m=1,…,K为分位点索引，K为输出分位点数量；$q_{\phi,m}$估计团队折扣回报分布$Z_\phi$在分位水平$\tau_m=(m-1/2)/K$处的值。均值价值采用：

```latex
Q_{\phi}^{\mathrm{tot}}(s,\mathbf{a})\approx\frac{1}{K}\sum_{m=1}^{K}q_{\phi,m}(s,\mathbf{a})
```

其中，$Q^{\mathrm{tot}}$为团队期望动作价值，分位点平均用于近似期望。拟采用有序分位点头或分位交叉惩罚保持输出有效性，不把任意局部随机价值的非线性组合直接当作正确联合回报分布。先用标量IQL版本检验非单调分解，再扩展分位回归，分别识别结构与分布信息的增益。

分布Critic拟使用带目标网络的分位Bellman回归：

```latex
y_n=r_k+\gamma(1-d_k)q_{\bar{\phi},n}(s_{k+1},\mathbf{a}^{\prime}),\quad\mathbf{a}^{\prime}\sim\pi_{\bar{\theta}}(\cdot\mid s_{k+1})
```

式中，k为环境时间步，$r_k$为即时团队奖励，γ为折扣率，$d_k$为禁止bootstrap的真实终止标记，$s_{k+1}$为下一条件，n为目标分位点索引，$y_n$为训练目标，$\bar{\phi}$和$\bar{\theta}$为延迟更新的Critic及Actor参数，a′为目标Actor根据各自下一局部输入产生的联合动作。通过分位Huber损失拟合预测分位点与目标样本。时间上限截断需单独处理；目标联合动作仍可能超出数据支持，拟通过行为约束及独立保守正则消融控制风险。回报分布的离散程度不直接解释为模型认知不确定性。

## 3 价值信用与联合行为约束的一致优化

令$\pi_{\theta_i}(a_i\mid o_i)$为智能体i的局部随机Actor，$θ_i$为其参数，θ为全体参数。为简化集中式推导，联合策略统一记为$\pi_\theta(\mathbf{a}\mid s)$，其实现仍按本地输入分解。候选优化目标采用行为正则策略优化（BRPO）：

```latex
J(\theta)=\mathbb{E}_{s\sim\mathcal{D}}[\mathbb{E}_{\mathbf{a}\sim\pi_{\theta}}Q_{\phi}^{\mathrm{tot}}(s,\mathbf{a})-\lambda D_{\mathrm{KL}}(\pi_{\theta}(\cdot\mid s)\Vert\hat{\mu}_{\psi,\mathcal{T}}(\cdot\mid s))]
```

其中，J为Actor最大化目标，$D_{\mathrm{KL}}$为reverse KL，λ>0为行为正则系数。KL两侧均定义在同一联合动作空间；更新单个Actor时，采用给定当前队友动作的自身条件行为分布，而不是把自身策略与队友动作分布比较。连续动作使用可计算密度的随机Actor；确定性版本作为代理目标对照，不能将Dirac策略的密度KL直接视为有限值。

为使价值与行为约束使用一致的信用标准，定义联合正则评分及反事实贡献：

```latex
S(s,\mathbf{a})=Q_{\phi}^{\mathrm{tot}}(s,\mathbf{a})+\lambda\log\hat{\mu}_{\psi,\mathcal{T}}(\mathbf{a}\mid s)
```

```latex
A_i^{\mathrm{reg}}(s,\mathbf{a})=S(s,\mathbf{a})-\mathbb{E}_{\tilde{a}_i\sim\rho_i}[S(s,\tilde{a}_i,\mathbf{a}_{-i})]
```

式中，S为价值与行为相容性的合成评分，$A_i^{\mathrm{reg}}$为正则化反事实贡献，$\mathbf{a}_{-i}$为队友联合动作，ã_i为替代自身动作，$ρ_i$为与本次实际自身动作无关的参考分布，可取冻结的当前自身策略。替换动作后同时重新评价Q和行为密度。该贡献依赖模型与参考分布，不宣称是唯一的因果信用。

随机Actor的统一目标还需保留熵项：

```latex
J(\theta)=\mathbb{E}_{s\sim\mathcal{D},\mathbf{a}\sim\pi_{\theta}}[S(s,\mathbf{a})-\lambda\sum_{i=1}^{N}\log\pi_{\theta_i}(a_i\mid o_i)]
```

式中各符号同前，负对数策略密度项对应熵奖励。实现优先对这一标量目标重参数化求导，反事实贡献用于信用诊断及梯度估计器对照。Q与行为网络可独立训练，但更新时必须使用相同状态、相同当前联合动作与相同动作尺度。

对树形模型，自身动作也可能是其他因子的条件，完整动作梯度应包含子因子的影响：

```latex
\nabla_{a_i}\log\hat{\mu}=\nabla_{a_i}\log p_{\psi_i}+\sum_{j\in\operatorname{ch}(i)}\nabla_{a_i}\log p_{\psi_j}
```

其中，$\nabla_{a_i}$表示对自身动作求导，ch(i)为节点i的子节点集合；式中$p_{\psi_i}$及$p_{\psi_j}$均为第一式中在相同s和当前动作处评价的完整因子。根节点使用根密度。冻结网络参数时仍保留动作输入梯度，避免把固定队友动作误解为条件因子对自身动作的导数为零。

## 四 研究方法 技术路线与实施安排

技术路线采用数据审计、模型预训练、联合策略优化和闭环评测四个环节。首先检查episode边界、agent身份、动作编码、奖励与终止语义，并按episode划分训练及验证数据。随后独立训练行为模型与Critic；Actor更新时从各自最新策略获得队友动作，更新一个Actor后刷新其动作。行为依赖树的结构顺序与Actor优化顺序分别配置。

实验先在两智能体双峰协作及XOR任务中比较准确Critic与学习Critic，再加入三阶依赖反例，区分表示误差与优化误差。环境评测以MPE simple_spread为起点，使用与数据匹配的动作和观测版本；根据适配情况扩展至MaMuJoCo，并借助PettingZoo及SMAC形成不同任务类型的补充验证[35,36]。离散与连续策略分别实现，不混用动作接口。

对照包括发布版OMSD、非单调Critic单项、结构化行为约束单项与完整框架；在同一随机Actor设置下进一步比较独立边缘约束、树形约束和完整条件约束，并设置参数量匹配网络、完整与截断梯度、均值与分布Critic、不同树及更新顺序的消融。评价团队回报、协同模式、覆盖与碰撞、跨种子方差、留出集拟合和完整训练/执行耗时，采用一致数据、种子及调参预算。

前期已完成相关文献阅读、MADiff算法复现及研究记录整理，具备继续开展离线数据与扩散模型实验的基础；本课题的新结构尚待实现和验证。代码拟以OMSD三阶段工程为起点，参考FACMAC的非单调Critic接口。按照文献与复现核查、模块实现、联合消融、结果分析及论文撰写分阶段推进，使用GitHub记录配置、代码版本和正负结果，保证结论可追溯。

可能的难点包括树形依赖不足、条件模型在新队友动作处失准、分布Critic外推及局部信息不可实现性。拟通过有界高阶因子对照、保守价值消融、有限更新步长和不同信息条件评测定位原因。若某模块只改善拟合而未改善闭环回报，应收缩相应创新主张。预期形成具有明确适用边界的候选算法、可复现实验及学位论文。

## 主要参考文献

[1] Ryan Lowe, Yi Wu, Aviv Tamar, et al. Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments[C]//Advances in Neural Information Processing Systems. 2017, 30.

[2] Chao Yu, Akash Velu, Eugene Vinitsky, et al. The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games[C]//Advances in Neural Information Processing Systems. 2022, 35.

[3] Sergey Levine, Aviral Kumar, George Tucker, et al. Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems[EB/OL]. arXiv:2005.01643, 2020[2026-10-20]. https://arxiv.org/abs/2005.01643.

[4] Scott Fujimoto, David Meger, Doina Precup. Off-Policy Deep Reinforcement Learning without Exploration[EB/OL]. arXiv:1812.02900, 2019[2026-10-20]. https://arxiv.org/abs/1812.02900.

[5] Jakob Foerster, Gregory Farquhar, Triantafyllos Afouras, et al. Counterfactual Multi-Agent Policy Gradients[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2018.

[6] Peter Sunehag, Guy Lever, Audrunas Gruslys, et al. Value-Decomposition Networks For Cooperative Multi-Agent Learning[EB/OL]. arXiv:1706.05296, 2017[2026-10-20]. https://arxiv.org/abs/1706.05296.

[7] Tabish Rashid, Mikayel Samvelyan, Christian Schroeder de Witt, et al. QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning[C]//Proceedings of the 35th International Conference on Machine Learning. 2018.

[8] Bei Peng, Tabish Rashid, Christian A. Schroeder de Witt, et al. FACMAC: Factored Multi-Agent Centralised Policy Gradients[C]//Advances in Neural Information Processing Systems. 2021, 34.

[9] Yifan Wu, George Tucker, Ofir Nachum. Behavior Regularized Offline Reinforcement Learning[EB/OL]. arXiv:1911.11361, 2019[2026-10-20]. https://arxiv.org/abs/1911.11361.

[10] Aviral Kumar, Aurick Zhou, George Tucker, et al. Conservative Q-Learning for Offline Reinforcement Learning[C]//Advances in Neural Information Processing Systems. 2020, 33.

[11] Ilya Kostrikov, Ashvin Nair, Sergey Levine. Offline Reinforcement Learning with Implicit Q-Learning[C]//International Conference on Learning Representations. 2022.

[12] Yiqin Yang, Xiaoteng Ma, Chenghao Li, et al. Believe What You See: Implicit Constraint Approach for Offline Multi-Agent Reinforcement Learning[C]//Advances in Neural Information Processing Systems. 2021, 34.

[13] Ling Pan, Longbo Huang, Tengyu Ma, et al. Plan Better Amid Conservatism: Offline Multi-Agent Reinforcement Learning with Actor Rectification[EB/OL]. arXiv:2111.11188, 2022[2026-10-20]. https://arxiv.org/abs/2111.11188.

[14] Jianzhun Shao, Yun Qu, Chen Chen, et al. Counterfactual Conservative Q Learning for Offline Multi-agent Reinforcement Learning[C]//Advances in Neural Information Processing Systems. 2023, 36.

[15] Xiangsen Wang, Haoran Xu, Yinan Zheng, et al. Offline Multi-Agent Reinforcement Learning with Implicit Global-to-Local Value Regularization[C]//Advances in Neural Information Processing Systems. 2023, 36.

[16] Daiki E. Matsunaga, Jongmin Lee, Jaeseok Yoon, et al. AlberDICE: Addressing Out-Of-Distribution Joint Actions in Offline Multi-Agent RL via Alternating Stationary Distribution Correction Estimation[C]//Advances in Neural Information Processing Systems. 2023, 36.

[17] The Viet Bui, Thanh Hong Nguyen, Tien Mai. ComaDICE: Offline Cooperative Multi-Agent Reinforcement Learning with Stationary Distribution Shift Regularization[C]//International Conference on Learning Representations. 2025.

[18] Callum Rhys Tilbury, Claude Formanek, Louise Beyers, et al. Coordination Failure in Cooperative Offline MARL[EB/OL]. arXiv:2407.01343, 2024[2026-10-20]. https://arxiv.org/abs/2407.01343.

[19] Jonathan Ho, Ajay Jain, Pieter Abbeel. Denoising Diffusion Probabilistic Models[C]//Advances in Neural Information Processing Systems. 2020, 33.

[20] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, et al. Score-Based Generative Modeling through Stochastic Differential Equations[C]//International Conference on Learning Representations. 2021.

[21] Jonathan Ho, Tim Salimans. Classifier-Free Diffusion Guidance[EB/OL]. arXiv:2207.12598, 2022[2026-10-20]. https://arxiv.org/abs/2207.12598.

[22] Michael Janner, Yilun Du, Joshua B. Tenenbaum, et al. Planning with Diffusion for Flexible Behavior Synthesis[EB/OL]. arXiv:2205.09991, 2022[2026-10-20]. https://arxiv.org/abs/2205.09991.

[23] Anurag Ajay, Yilun Du, Abhi Gupta, et al. Is Conditional Generative Modeling all you need for Decision-Making?[C]//International Conference on Learning Representations. 2023.

[24] Zhengbang Zhu, Minghuan Liu, Liyuan Mao, et al. MADiff: Offline Multi-agent Learning with Diffusion Models[C]//Advances in Neural Information Processing Systems. 2024, 37.

[25] Zhendong Wang, Jonathan J Hunt, Mingyuan Zhou. Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning[C]//International Conference on Learning Representations. 2023.

[26] Huayu Chen, Cheng Lu, Zhengyi Wang, et al. Score Regularized Policy Optimization through Diffusion Behavior[C]//International Conference on Learning Representations. 2024.

[27] Marcel Hedman, Kale-ab Abebe Tessera, Juan Claude Formanek, et al. CODA: Coordination via On-Policy Diffusion for Multi-Agent Offline Reinforcement Learning[EB/OL]. arXiv:2604.23308, 2026[2026-10-20]. https://arxiv.org/abs/2604.23308.

[28] Dan Qiao, Wenhao Li, Shanchao Yang, et al. Offline Multi-agent Reinforcement Learning via Sequential Score Decomposition[C]//Proceedings of the 43rd International Conference on Machine Learning. 2026: 101269–101297.

[29] C. Chow, C. Liu. Approximating discrete probability distributions with dependence trees[J]. IEEE Transactions on Information Theory, 1968, 14(3): 462–467. DOI: 10.1109/TIT.1968.1054142.

[30] Sergey Kirshner, Padhraic Smyth, Andrew Robertson. Conditional Chow-Liu Tree Structures for Modeling Discrete-Valued Vector Time Series[C]//Proceedings of the 20th Conference on Uncertainty in Artificial Intelligence. 2004.

[31] Hongwei Wang, Lantao Yu, Zhangjie Cao, et al. Multi-Agent Imitation Learning with Copulas[EB/OL]. arXiv:2107.04750, 2021[2026-10-20]. https://arxiv.org/abs/2107.04750.

[32] Wendelin Böhmer, Vitaly Kurin, Shimon Whiteson. Deep Coordination Graphs[C]//Proceedings of the 37th International Conference on Machine Learning. 2020.

[33] Will Dabney, Mark Rowland, Marc G. Bellemare, et al. Distributional Reinforcement Learning with Quantile Regression[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2018.

[34] Wei-Fang Sun, Cheng-Kuang Lee, Simon See, et al. A Unified Framework for Factorizing Distributional Value Functions for Multi-Agent Reinforcement Learning[J]. Journal of Machine Learning Research, 2023, 24(220): 1–32.

[35] J. K. Terry, Benjamin Black, Nathaniel Grammel, et al. PettingZoo: Gym for Multi-Agent Reinforcement Learning[EB/OL]. arXiv:2009.14471, 2021[2026-10-20]. https://arxiv.org/abs/2009.14471.

[36] Mikayel Samvelyan, Tabish Rashid, Christian Schroeder de Witt, et al. The StarCraft Multi-Agent Challenge[EB/OL]. arXiv:1902.04043, 2019[2026-10-20]. https://arxiv.org/abs/1902.04043.

## 拟创新点与特色

1 面向协同模式的稀疏条件行为克隆

拟将树形概率分解用于离线MARL联合行为建模，以局部条件因子保留队友动作依赖，研究相较独立边缘克隆的模式相容性，以及相较完整条件模型的复杂度折中。树形概率模型本身已有理论基础，创新拟体现在与当前队友策略约束的结合、适用边界与实验验证。

2 不依赖IGM的非线性分布Critic

拟以个体观察—动作特征和非单调mixer表示集中式团队价值，并扩展有序分位点头描述回报分布。通过标量与分布版本对照，研究交互表达与回报分布信息对信用分配的作用。非单调分解和分布强化学习已有先例，本课题重点是其在相关离线行为约束下的组织与验证。

3 联合行为与价值信用的一致更新

拟由同一BRPO目标推导Q引导、完整联合行为梯度和正则化反事实贡献，使二者在相同当前联合动作上评价个体更新；研究子因子梯度、动作刷新及结构近似对协同选择的影响，在严格本地执行条件下检验性能和稳定性。上述内容均为拟研究方案，实际创新程度与效果以实验和进一步文献核查为准。
