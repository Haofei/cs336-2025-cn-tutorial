# Stanford CS336 Lecture 16 中文教程：Alignment 中的强化学习（一）

本讲是后训练（post-training）部分的第二讲，主题从传统 RLHF 过渡到“可验证奖励上的强化学习”（reinforcement learning from verifiable rewards）。核心问题是：为什么语言模型对齐需要 RL？PPO、GRPO 这类算法到底在优化什么？为什么训练会不稳定，工程上又有哪些关键细节？

## 1. 从 RLHF 到可验证奖励

RLHF（Reinforcement Learning from Human Feedback）通常从人类偏好数据开始：给定同一个 prompt 的两个回答，人类标注哪一个更好。目标是训练一个语言模型 policy，使它更倾向产生人类喜欢的回答。

这里的 policy 指模型在给定上下文后对输出序列的概率分布。与预训练或 SFT（supervised fine-tuning）不同，RLHF 不是单纯做“数据分布拟合”：模型生成什么会改变它获得的 reward，因此目标中包含“从当前模型采样”的过程。这使优化比普通最大似然更难。

上一讲提到的 DPO（Direct Preference Optimization）把偏好优化转化成一种类似监督学习的目标：不显式训练 reward model，也不跑完整 RL loop，而是通过偏好对直接调整 policy。DPO 的直觉很简单：提高 chosen response 的概率，降低 rejected response 的概率；当模型的隐含 reward 判断错得越多，更新越大。DPO 因为实现简单，一度成为开源模型 post-training 的主流方法。

但 DPO 也有局限：它天然适合 pairwise preference，不太适合“数学题答对/答错”这类只有标量奖励的任务；它通常也是离线的，即先收集一批偏好对，再在其上训练。对于推理模型，研究者更希望在模型不断生成新解答的过程中，直接根据可验证结果进行在线优化。

## 2. RLHF 的两个风险：过优化与校准变差

RLHF 中最重要的经验现象之一是 overoptimization（过优化）。reward model 只是人类偏好的代理模型，带有噪声和误差。训练初期，优化代理 reward 往往能提升真实人类偏好；但继续优化后，模型可能开始“钻 reward model 的空子”，代理 reward 继续上升，真实 win rate 却停滞甚至下降。这与监督学习中的 train-test gap 类似：训练集上的 reward model 不等于真实偏好 oracle。

另一个现象是 calibration（校准）变差。预训练模型可被看作概率生成模型，而 RLHF 后的模型更像为了某个 reward 调整过的 policy。若 reward 没有鼓励“表达不确定性”，模型就可能变得更自信、更迎合、更少说“不知道”。因此不要把 RLHF 模型的输出概率直接理解为可靠的真实概率估计。

这些问题说明：人类偏好很有价值，但难以大规模、低噪声、稳定地优化。于是一个自然方向是寻找 reward 更清晰的任务，例如数学、代码、形式化证明、可执行测试等。在这些领域，答案是否正确可以自动验证，reward 更接近真实目标，也更不容易被 reward hacking。

## 3. RL 基础：policy、reward、value、advantage

在语言模型 RL 中，一个样本通常包含 prompt 和模型生成的 response。可以把生成完整 response 看成一次 rollout。常见术语如下：

- Policy：当前语言模型 πθ，即给定 prompt 后生成 token 序列的概率分布。
- Reward：对生成结果的评分。RLHF 中可能来自 reward model；数学/代码中可能来自答案匹配、单元测试、格式检查等。
- Value function：估计某个状态或部分生成未来能获得多少 reward 的函数，常用于降低 policy gradient 的方差。
- Advantage：某个动作/输出比基线好多少。直觉上，advantage 为正就提高该输出概率，为负就降低该输出概率。

最基本的 policy gradient 思想是：对高 reward 的输出增加 log probability，对低 reward 的输出减少 log probability。许多 RL 算法本质上都可理解为“upweight good stuff, downweight bad stuff”，区别在于：如何定义好坏、如何做方差降低、如何避免 policy 一步走太远。

语言模型 RL 还有一个特点：很多任务更像 contextual bandit。模型看到 prompt，生成完整回答，然后得到一个终局 reward；没有传统游戏环境里复杂的状态转移。但训练时仍常把 KL penalty 等正则项分摊到 token 级别，而把“答对/答错”这样的任务 reward 放在最后一个 token 或序列级别。

## 4. PPO：强大但工程复杂

PPO（Proximal Policy Optimization）是 RLHF 早期最重要的算法之一。它从 policy gradient 出发，引入两个关键机制。

第一，重要性采样和旧 policy。纯 on-policy 方法要求每次更新都用当前 policy 新生成样本，代价很高，因为 rollout 很慢。PPO 允许先用旧 policy 采样一批数据，再对同一批 rollout 做多次梯度更新。

第二，clipping。PPO 不希望新 policy 相比旧 policy 变化过大，因此使用概率比值 ratio，并把它裁剪在 1-ε 到 1+ε 之间，例如 0.8 到 1.2。这样即使某个样本 reward 很高，模型也不会无限制地把概率推高，从而提升训练稳定性。

PPO 通常还需要 value model 来估计 advantage，例如使用 GAE（Generalized Advantage Estimation）。这能降低梯度方差，但代价是工程复杂：要维护 policy model、reward model、value model，有时还要处理不同 tokenizer、KL shaping、value loss、policy loss、clip norm、rollout 与训练 worker 的同步等。实际 PPO 有大量实现细节，稍有差异就可能影响结果。

在大语言模型上，value model 尤其昂贵：它往往和 policy 一样大，显存和计算成本接近翻倍。因此人们希望找到一种保留 PPO 稳定性、但去掉 value model 的方法。

## 5. GRPO：用组内基线替代 value model

GRPO（Group Relative Policy Optimization）可以看作 PPO 的简化变体，也是 DeepSeek Math / R1 系列中的关键算法。它保留 policy gradient、KL regularization、ratio clipping 等思想，但去掉了 value function 和复杂的 GAE。

GRPO 的核心做法是：对同一个问题 q，一次采样 G 个回答，形成一个 group。每个回答都有 reward。然后用组内 reward 的均值和标准差构造 advantage：

A_i = (r_i - mean(r_1, ..., r_G)) / std(r_1, ..., r_G)

也就是说，不再问“这个回答的绝对 reward 多高”，而是问“它比同一问题下的其他回答好多少”。这很自然：不同题目难度不同，简单题平均 reward 高，难题平均 reward 低；组内均值可以作为题目难度的 baseline。这样就不需要额外训练 value model。

如果只对每批 rollout 做一步在线更新，GRPO 甚至可以非常接近普通 policy gradient：高于组内平均的回答被上调，低于平均的回答被下调。实现上只需：生成多个回答、计算 reward、按组归一化、加 KL penalty、做梯度更新。

但 GRPO 也有微妙问题。标准差归一化并不是严格 policy gradient 推导中允许的普通 baseline。它会放大 reward 方差很小的组：例如所有回答都错或都对的题目。这可能把训练重点放在“太难”或“太容易”的问题上，而不是最有学习信号的中等难度问题。

另一个问题是长度归一化。如果把序列 reward 除以输出长度，那么答错时模型可能通过生成更长内容来稀释负 reward；答对时则倾向更短。这会诱导模型在不确定时输出很长的 chain-of-thought，看起来像“思考更久”，但可能只是目标函数偏差。后续 Dr. GRPO 等分析认为，去掉某些长度归一化能在保持 reward 的同时减少无界变长。

## 6. 为什么可验证奖励推动 reasoning models

以 DeepSeek R1 为例，训练流程展示了一个非常简单但有效的范式：在数学、代码等可验证任务上，用 outcome reward（最终答案对不对）进行 RL。R1-Zero 几乎直接从基础模型出发做 RL，reward 主要包括 accuracy reward 和 format reward。format reward 要求模型把推理放在特定 think tags 中，虽然看似只是格式约束，但实践中对稳定训练很重要。

R1 的重要结论是：不一定需要复杂的 MCTS search 或 PRM（Process Reward Model）。PRM 能给推理中间步骤打分，理论上反馈更丰富，但很难构建可靠的过程监督器。R1 发现，简单的 outcome-based reward 加 GRPO 就能得到强推理能力。

实际可发布模型通常不会只做 RL。更常见流程是：先做少量 long chain-of-thought SFT，让模型学会可读的推理格式；再做 verifiable reward RL，提高数学/代码正确率；最后再做通用 instruction tuning 和 RLHF，恢复聊天、写作、安全等通用能力。这说明 SFT 和 RL 是互补的：SFT 提供初始行为模式，RL 则让模型针对真实目标继续优化。

Kimi K1.5 和 Qwen3 也体现了类似思路。Kimi 强调数据筛选和长度控制：用 best-of-N 过滤太容易的问题，构造课程学习，并在训练后期加入 length reward，避免推理链过长导致推理成本失控。Qwen3 则加入 thinking mode fusion：同一模型支持 think 与 no-think 模式，并可通过 token budget 控制测试时思考长度，实现 inference-time scaling。

## 7. 训练稳定性与工程注意事项

LLM RL 的难点不只在算法，还在系统。rollout 需要自回归生成，远比普通 teacher-forcing 训练慢；训练 worker 更新权重后，还要同步到 inference worker；长 chain-of-thought 会造成 batch 长度不均，降低 GPU 利用率。许多系统会把训练和推理分成不同 worker，并用 vLLM 等推理引擎生成样本。

稳定训练通常依赖以下技巧：

1. KL regularization：限制新 policy 不要偏离 reference policy 太远，避免语言质量崩坏。
2. Clipping 或显式正则：控制单次 policy update 幅度。
3. 合理 baseline：降低 policy gradient 方差，例如 PPO 的 value function 或 GRPO 的组内均值。
4. Reward shaping：把格式、语言一致性、长度等辅助目标以加权 reward 形式加入，但权重需要经验调参。
5. 数据难度控制：太容易没有学习信号，太难全错也没有信号；best-of-N 过滤和 curriculum 能改善训练效率。
6. 长度控制：长推理可能提升性能，也可能只是被目标函数诱导；需要在正确率与推理成本之间权衡。

## 8. SFT、RL 与 RLHF 的分工

把这一讲放回整个 alignment pipeline，可以看到三类训练各有角色。SFT 负责给模型示范“应该怎样回答”，例如遵循指令、写出长推理链、使用固定格式、避免明显有害输出。它的优点是稳定、便宜、容易调试；缺点是只能模仿数据中已有的行为，不能直接鼓励模型探索比示范更好的解法。

RL 负责把模型从“会模仿”推向“会优化目标”。在数学和代码中，模型可以尝试许多不同解法，只要最终答案或测试通过，就获得正 reward。这种探索能发现 SFT 数据没有覆盖的行为模式，也能把正确答案概率持续推高。RLHF 则把优化目标从可验证任务扩展到人类偏好，例如有用性、礼貌性、安全性和风格一致性；但由于偏好 reward 噪声更大，所以更需要 KL、早停、评测集和人工检查来防止过优化。

因此，一个实用的顺序通常是：先用 SFT 建立可控的初始 policy；再在高质量、可验证、难度合适的数据上做 RL，提升推理和解题能力；最后用偏好优化或 RLHF 修正通用聊天体验。若顺序反过来，直接从很弱或格式混乱的模型开始 RL，reward 可能太稀疏，训练会不稳定；若只做 SFT 不做 RL，模型又可能停留在“看起来像会推理”，而不是在真实可验证目标上取得最高正确率。

评估时也要区分不同目标：数学榜单提升不代表通用助手更好，通用偏好提升也不代表推理更强。可靠的训练流程需要同时监控任务正确率、回答长度、KL 距离、格式违规率、拒答率、人工偏好和安全指标。只有这些曲线一起合理，才能说明 RL 真正在改善模型，而不是仅仅利用了某个评测或 reward 的漏洞。

## 9. 小结

本讲的主线是：RLHF 证明了 RL 可以用于语言模型对齐，但人类偏好 reward 噪声大、易过优化；可验证奖励提供了更清晰、更可扩展的训练信号。PPO 是经典且强大的 RLHF 算法，但 value model 和大量实现细节使它成本高、难调。GRPO 用同题多回答的组内相对 reward 替代 value model，大幅简化训练，因此成为 reasoning model 后训练的重要工具。

从 R1、Kimi K1.5、Qwen3 的经验看，成功 recipe 往往包含：少量高质量 long CoT SFT、可验证任务上的 RL、KL/长度/格式等稳定化约束、再加通用 RLHF 或指令微调。最终目标不是让模型“无限思考”，而是在可控成本下，把 policy 推向更高正确率、更好对齐和更稳定的行为。