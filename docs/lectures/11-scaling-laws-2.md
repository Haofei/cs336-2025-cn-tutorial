# CS336 第 11 讲教程：Scaling Laws（二）

上一讲介绍了 scaling laws 的基本思想：用小规模实验拟合 loss、参数量、数据量和 compute 之间的关系，再外推到大模型。本讲更接近真实大模型训练流程：公开论文中的团队到底怎样用 scaling laws 选学习率、batch size、模型大小、训练 token 和架构？这些拟合什么时候可靠，什么时候只是看起来像一条直线？

核心结论可以先概括为一句话：scaling laws 不是单个公式，而是一套降低大规模训练风险的实验方法。它通常包括小模型代理实验、超参数迁移、IsoFLOP 分析、WSD 学习率日程、μP 参数化，以及对外推结果的保守验证。

## 1. 真实训练中的 Scaling 问题

训练一个前沿语言模型前，团队必须回答几个昂贵问题：

1. 给定训练 FLOPs，应该用多大的模型、多少 token？
2. batch size 应随规模怎样变化？
3. 学习率是否需要随模型变大而降低？
4. 小模型上调好的超参数能否迁移到大模型？
5. 新架构在小规模看起来不错，放大后是否仍然划算？

这些问题不能靠在 70B 或 400B 模型上反复试错。公开前沿实验中常见的策略是：先训练一批 10M、100M、1B 级别的代理模型，拟合趋势，然后只在目标规模做少数验证或最终训练。Cerebras-GPT、MiniCPM、DeepSeek LLM、Llama 3、Hunyuan、MiniMax-01 都以不同方式展示了这种流程。

## 2. Cerebras-GPT：用 μP 让超参数更稳定

Cerebras-GPT 训练了从约 0.1B 到 13B 的模型，并重点验证 μP（mu-parameterization，最大更新参数化）。标准参数化下，模型越宽，最优学习率通常会向更小值移动；如果直接把小模型学习率搬到大模型，训练可能不稳定或 loss 偏高。μP 的目标是重新缩放初始化和逐层学习率，使“同一个基础学习率”在不同宽度下都接近最优。

实践上，μP 的做法大致是：非 embedding 权重按宽度缩放初始化；使用 Adam/AdamW 时，不同层的学习率也按 fan-in 或宽度缩放，而不是所有参数共享一个全局学习率。这样做的好处是，小模型上可以进行密集网格搜索，找到学习率、初始化、宽深比等设置；放大模型时，这些设置更可能保持有效。

Cerebras-GPT 的实验显示，μP 曲线更贴近预期 scaling law，标准参数化则更容易在不同规模出现振荡或偏离。这不是说没有 μP 就训练不了大模型，而是 μP 把“每个规模都重新调学习率”的问题，转化为“在小规模调好后迁移”的问题。

## 3. MiniCPM：小模型、长训练与 WSD 日程

MiniCPM 的目标是训练很强的小模型，例如 1B 到 2B 级别，但用大量数据充分训练。它的 scaling 实验包含三个重要工具。

第一，仍然使用 μP 来稳定学习率迁移。MiniCPM 在几十 M 参数的小代理模型上搜索超参数，再放大到数百 M 或 1B 级别。实验中，不同模型大小的最优学习率基本落在同一位置，这支持了 μP 的实用价值。

第二，拟合 critical batch size。临界 batch size 可以理解为“继续增大 batch 开始收益递减”的位置。模型越大、目标 loss 越低，通常可以使用更大的 batch。MiniCPM 通过不同 batch size 的训练曲线，拟合目标 loss 与最优 batch size 的 log-log 关系，再把它外推到目标训练规模。

第三，也是本讲非常重要的一点：使用 WSD（warmup-stable-decay）学习率日程来降低 Chinchilla 式数据/模型 scaling 的实验成本。

常规 cosine schedule 的问题是：如果总训练 token 不同，整条 cosine 曲线都不同。一个训练到 1T token 的模型，其中 100B token 处的 checkpoint，并不等价于“从头训练 100B token 并完整 cooldown”的模型。因此，不能简单拿长训练的中间 checkpoint 当作短数据量实验点。

WSD 把学习率分成三段：

```text
warmup -> stable plateau -> decay/cooldown
```

它的好处是 stable 阶段可以复用。想估计较短数据量的最终 loss 时，可以从中间 checkpoint 回退出来，单独接一个 decay 阶段。这样，一个长训练加若干短 cooldown，就能近似获得多个数据量终点，大幅节省重复训练成本。

MiniCPM 用 WSD 做 Chinchilla 风格分析，并用两种方法估计模型/数据最优比例：一种是取训练曲线下包络线，另一种是直接拟合二维 loss surface：

```text
L(N, D) = E + A / N^alpha + B / D^beta
```

它得到的 token/parameter 比例很高，约 192:1。这个数字未必应被当作通用定律；更重要的启发是，Chinchilla 的 20:1 不是不可突破的硬规则。现代高质量数据、改进架构和更强优化可能让“更多 token、更小模型”的方案更有吸引力，尤其当推理成本也被纳入目标函数时。

## 4. DeepSeek LLM：直接拟合学习率、Batch Size 和 IsoFLOP

DeepSeek LLM 的公开论文很有价值，因为它展示了另一种更直接的 scaling 方法：不依赖 μP，而是显式拟合 batch size 和 learning rate 随规模的变化。

DeepSeek 先在小模型上对 batch size 与学习率做网格搜索，找到每个规模下的最优点或近似最优区域。然后把这些最优 batch size、学习率与训练 FLOPs 放到 log-log 图上拟合趋势，再外推到 7B 和 67B 模型。

这里有一个实践判断：batch size 的 scaling 往往比较干净，学习率的 scaling 更噪声、更可疑。学习率曲线有时看起来也可以用水平线解释。因此，拟合学习率更多是为了得到正确数量级，而不是得到精确公式。大模型训练通常有较宽的“可用盆地”，只要学习率不差一个数量级，训练可能仍然可行。

DeepSeek 同样做了 Chinchilla / IsoFLOP 分析。IsoFLOP 的流程是：固定多个 compute budget；在每个 budget 下训练不同模型大小，小模型看更多 token，大模型看更少 token；找到每条固定 FLOPs 曲线的最低 loss 点；再拟合最优参数量和最优 token 数如何随 FLOPs 增长。相比学习率拟合，这类 IsoFLOP 曲线通常更稳定、更可信。

DeepSeek 还使用 WSD 式 cooldown 来减少重复训练，并最终从约 10^20 FLOPs 量级外推到约 10^24 FLOPs 量级，较准确地预测了 7B 和 67B 模型的 loss。这说明，在训练制度、数据和架构保持一致时，loss scaling 的外推确实可以成为大训练前的风险控制工具。

这类成功外推也说明了为什么许多团队愿意在正式训练前投入大量“小实验预算”。这些实验本身可能已经很贵，但如果能避免一次目标规模训练失败，成本就是值得的。更现实地说，scaling law 不一定要精确预测最终所有 benchmark；只要它能提前发现学习率数量级错误、batch size 不合适、token/parameter 比例明显偏离，或者某个架构在放大后没有优势，就已经能节省大量算力。

## 5. 近期模型中的趋势：比例在变，方法在复用

Llama 3、Hunyuan 和 MiniMax-01 的论文没有像 MiniCPM 或 DeepSeek 那样给出大量 scaling 细节，但仍显示几个趋势。

Llama 3 重新做了 IsoFLOP / Chinchilla 分析，得到的最优 token/parameter 比例大约是 40:1，高于 Chinchilla 的 20:1。它还尝试把训练 loss 或 negative log likelihood 映射到下游 benchmark accuracy，例如用 sigmoid 拟合 loss 到 MMLU 等任务表现的关系。这样做的动机很明确：团队真正关心的不是 log loss 本身，而是下游能力。但 benchmark 分数更噪声、更容易饱和，因此通常还是先预测 loss，再辅助预测 benchmark。

Hunyuan 的分析得到更高的 active-parameter token 比例，例如约 96:1。这里要注意 MoE 或稀疏模型中的“总参数”和“激活参数”不同，比例不能直接与稠密模型混为一谈。

MiniMax-01 则把 scaling laws 用于架构选择。它比较 softmax attention、linear attention 和 hybrid attention 的 loss-compute 曲线，观察它们在下包络线和最优模型/token 趋势上是否接近。如果线性注意力在相同 compute 下的 scaling 曲线没有明显变差，就能支持把它用于长上下文模型。这说明 scaling laws 不只用于选大小，也能用于判断新架构是否值得放大。

## 6. μP 的直觉：控制激活和更新的尺度

μP 背后的数学直觉可以简化为两个条件。

第一，模型变宽时，每个坐标的 activation 不应爆炸或消失。若一层是矩阵乘法 `h_l = W_l h_{l-1}`，为了让输出 activation 的尺度稳定，常见初始化需要按 fan-in 的平方根缩放，类似：

```text
W_l ~ 1 / sqrt(fan_in)
```

这与 Kaiming/Xavier 初始化的直觉一致。

第二，做一次梯度更新后，activation 的变化量也不应随宽度爆炸或消失。这个条件会约束学习率如何随层宽变化。对 SGD 推导可得到类似 fan-out/fan-in 的比例；对 Adam/AdamW，由于自适应归一化改变了梯度尺度，常见 μP 规则会让学习率按 fan-in 或宽度缩放。

因此，μP 的重点不只是初始化，而是“初始化 + 逐层学习率 + 某些前向缩放”共同保证更新尺度稳定。对 Transformer，还可能涉及 attention logits 的缩放；一些 μP 实现会用 `1/d` 而不是传统 `1/sqrt(d)` 的注意力缩放，以满足更新稳定性要求。

经验研究显示，μP 对很多变化是稳健的：换 ReLU、SwiGLU、Squared ReLU，或在一定范围内改变 batch size，学习率迁移仍然可行。但它也不是万能的：可学习 norm gain、强 weight decay、Lion 这类 sign-gradient 风格优化器，都可能破坏 μP 的迁移假设。换句话说，μP 是针对特定优化器和参数化设计的工程工具，不是对所有训练配方自动成立的定理。

## 7. 实验拟合的实践流程

一个较完整的 scaling 实验可以按以下步骤执行：

1. 固定数据、tokenizer、架构族、优化器和训练代码，减少混杂变量。
2. 选若干小到中等规模模型，覆盖至少几个数量级的参数或 FLOPs。
3. 用小规模网格搜索学习率、batch size、warmup、weight decay 等关键超参数。
4. 若使用 μP，则在小模型上调参并验证学习率能跨宽度迁移；若不用 μP，则显式拟合学习率和 batch size 随规模变化。
5. 用 WSD 或等价方法收集不同 token 终点的真实 cooldown loss，避免误用 cosine 中间 checkpoint。
6. 做 IsoFLOP 或二维 `L(N,D)` 拟合，估计给定 compute 下的最优模型大小和训练 token。
7. 在比最终规模小但明显大于拟合点的中等规模上做验证，检查外推是否命中。
8. 最终训练时保留监控：如果 loss 偏离预测，要尽早排查数据、优化器、batch、学习率 schedule 和实现 bug。

## 8. 局限性与常见误区

第一，log-log 直线不等于真理。Scaling law 是经验拟合，依赖数据分布、模型族、优化器、训练 schedule 和评估集。外推越远，不确定性越大。

第二，训练 loss 最稳定，下游能力不一定稳定。困惑度下降通常是好信号，但数学推理、工具使用、长上下文、指令跟随等能力可能有阈值效应或评测噪声。

第三，token/parameter 比例没有统一常数。20:1、40:1、96:1、192:1 都出现在不同论文中，差异来自数据质量、架构、是否 MoE、训练目标和部署成本。Chinchilla 给的是训练 FLOPs 最优基准，不是产品总成本最优答案。

第四，学习率 scaling 比 loss scaling 更脆弱。Batch size 和 IsoFLOP 往往更容易拟合；学习率曲线可能很平、噪声很大，只能作为数量级参考。

第五，不能把长训练的中间 checkpoint 当作短训练终点。没有 cooldown 的 checkpoint loss 往往偏高，会系统性污染数据 scaling 拟合。WSD 的价值正是解决这个问题。

## 小结

本讲展示了 scaling laws 在真实模型训练中的用法。Cerebras-GPT 和 MiniCPM 用 μP 稳定超参数迁移；MiniCPM 和 DeepSeek 用 WSD 降低 Chinchilla 分析成本；DeepSeek 直接拟合 batch size、学习率和 IsoFLOP，并成功预测大模型 loss；Llama 3、Hunyuan、MiniMax-01 则说明现代团队仍在复用 IsoFLOP 分析，只是最优 token/parameter 比例和架构问题会随目标变化而变化。

真正可靠的 scaling 工作不是“画一条线然后相信它”，而是用小规模实验系统排除风险：让学习率、batch size、模型大小、数据量和架构选择都在放大前有证据支撑，同时承认外推的局限，并用中等规模验证点校准预测。它的价值在于把大模型训练从豪赌变成可管理的工程决策：先用便宜实验缩小搜索空间，再把昂贵训练押在最有证据的方案上。
