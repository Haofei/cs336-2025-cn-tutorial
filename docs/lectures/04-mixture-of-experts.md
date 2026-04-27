# Stanford CS336 第 4 讲教程：Mixture of Experts（MoE）

## 学习目标

学完本讲，你应该能够：

1. 解释 Mixture of Experts（MoE，专家混合）相对 dense model（稠密模型）的核心优势：用接近相同的激活计算量获得更多总参数。
2. 理解 router / gating（路由器/门控）、expert（专家）、top-k routing（Top-K 路由）、load balancing（负载均衡）等关键术语。
3. 写出 MoE 前向传播的基本公式与伪代码。
4. 分析 MoE 在训练和推理中的成本：FLOPs、显存、通信、负载不均、token dropping。
5. 理解现代 MoE 系统（如 DeepSeek、Mixtral、Grok、Llama 4）常见的工程权衡。

## 先修知识

本讲默认你已经了解：Transformer 的基本结构，尤其是 self-attention 和 FFN/MLP；softmax、top-k、残差连接（residual connection）；语言模型训练中的 next-token prediction；以及基本的 GPU/TPU 并行训练概念。

## 本讲地图

MoE 的主线可以概括为一句话：把 Transformer 中昂贵的 FFN 层替换成多个“专家”FFN，并让每个 token 只激活其中少数几个专家。

本讲依次讨论：

- 为什么 MoE 在相同 FLOPs 下通常优于 dense model；
- router 如何为 token 选择 expert；
- expert 应该多大、多少个，是否需要 shared expert；
- 为什么 load balancing 是训练 MoE 的关键；
- MoE 的系统代价：通信、显存、并行、token dropping；
- DeepSeek 系列如何把这些想法组合成现代大规模 MoE 架构。

## 核心概念

### 1. Dense model 与 MoE 的区别

在普通 Transformer 中，每一层通常包含 attention 和 FFN。dense model 的 FFN 是一个固定的大 MLP：每个 token 都经过同一个 FFN。

MoE 把这个 FFN 替换成多个 expert。每个 expert 通常也是一个 FFN，但每个 token 不会经过所有 expert，而是由 router 选择其中 K 个。若每个 token 只激活 1 或 2 个 expert，则计算量主要取决于“activated parameters”（被激活参数），而不是模型总参数。

因此，MoE 的优势是：

- 总参数更多：模型有更大容量，可以记忆和表达更多模式；
- 激活参数较少：每个 token 只用一小部分参数，FLOPs 不随专家总数线性增长；
- 天然适合 expert parallelism（专家并行）：不同 expert 可放在不同设备上。

但它也带来复杂性：路由是离散选择，不易优化；不同 expert 负载可能严重不均；跨设备发送 token 会产生通信开销。

### 2. Expert 并不一定是“语义专家”

“Mixture of Experts”这个名字容易误导。它并不保证一个专家负责代码、一个负责数学、一个负责中文。expert 更准确地说是若干可被稀疏激活的子网络。它们可能形成某种 specialization（专门化），但这种专门化通常不是人工可解释的领域划分。

更好的心智模型是：MoE 在每一层提供了多条可选的非线性变换路径。router 根据当前 hidden state 选择其中几条路径。由于 hidden state 已经包含上下文、位置、前面层的计算结果，所以同一个表面 token 在不同语境下也可能被送往不同 expert。比如 “Python” 在编程上下文和动物上下文中，路由结果可能不同；但这并不意味着某个 expert 可以被简单命名为“编程专家”。

### 3. Router / Gating

router 是一个很轻量的模块，输入 token 的 hidden state，输出该 token 对每个 expert 的 affinity score（亲和分数）。常见做法是一个线性投影加 softmax 或 sigmoid，然后选择分数最高的 K 个 expert。

router 的输出也常被称为 gating weights（门控权重），用于加权组合多个 expert 的输出。

### 4. Top-K Routing

现代大规模 MoE 基本收敛到 token choice top-k routing：每个 token 自己选择分数最高的 K 个 expert。

也存在其他方案：

- expert choice：每个 expert 选择自己要处理的 token，天然负载均衡，但可能不是 token 的最佳选择；
- global assignment：解一个全局匹配/最优传输问题，更优雅但计算代价高；
- hashing routing：用哈希函数固定分配 token，意外地也能带来增益，但不如学习式路由灵活；
- RL routing：用强化学习处理离散选择，原则上合理，实践中成本和方差太高。

## 逐步教程

### 第一步：把 FFN 替换成专家池

普通 FFN 可写成：

```text
h_out = h + FFN(h)
```

MoE 层则写成：

```text
h_out = h + sum_{i in TopK(router(h))} g_i(h) * Expert_i(h)
```

其中：

- `h` 是当前 token 的 hidden state；
- `Expert_i` 是第 i 个 FFN；
- `router(h)` 给出 token 对所有 expert 的分数；
- `TopK` 只保留 K 个 expert；
- `g_i(h)` 是门控权重，可归一化，也可不完全归一化，取决于实现。

如果 K=1，计算量接近一个 dense FFN；如果 K=2，大致相当于激活两个 FFN，FLOPs 约翻倍。但模型总参数可以远大于 dense model。

### 第二步：理解专家数量与大小

早期 MoE 常把 dense FFN 复制成多个同样大小的 expert。后来 DeepSeek 等系统发现 fine-grained experts（细粒度专家）非常有效：把每个 expert 做得更小，但数量更多。

例如，原本一个 FFN 的中间维度是 hidden size 的 4 倍。细粒度 MoE 可以把每个 expert 的中间维度切成原来的 1/2、1/4 甚至更小，然后激活更多个 expert。这样可以增加路由组合的灵活性，同时控制 FLOPs。

另一个设计是 shared expert（共享专家）：无论 router 选择什么，每个 token 都经过一个或几个共享 FFN。动机是让模型保留一部分通用处理能力，而不是所有计算都依赖稀疏路由。DeepSeek 系列使用过 shared expert，但其他模型的消融实验显示其收益并不总是稳定，因此这是一个工程选择。

### 第三步：为什么 load balancing 必不可少

如果没有约束，router 很容易陷入坏的局部最优：所有 token 都被送到少数几个 expert，其他 expert 几乎不训练，成为 dead experts（死亡专家）。这不仅浪费显存，也使 MoE 退化成一个小得多的模型。

因此训练 MoE 时通常加入 auxiliary load balancing loss（辅助负载均衡损失）。Switch Transformer 中常见形式是：

```text
L_balance = alpha * N * sum_i f_i * p_i
```

其中：

- `N` 是 expert 数量；
- `f_i` 是实际被路由到 expert i 的 token 比例；
- `p_i` 是 router 分配给 expert i 的平均概率；
- `alpha` 是损失权重。

直觉是：如果某个 expert 已经拿到太多 token，它对应的路由概率会被压低，从而鼓励 token 分散到其他 expert。

负载均衡不仅是系统优化，也是建模优化。即使不考虑 GPU 利用率，也需要它来避免专家坍缩。

### 第四步：训练中的稳定性问题

MoE 难训练的原因主要有三点：

1. top-k 是离散选择，未被选中的 expert 没有梯度；
2. router 可能过早偏向少数 expert；
3. softmax router 在低精度训练中可能引入数值不稳定。

常见稳定化技巧包括：

- router 计算使用 float32；
- 对 router logits 加 z-loss，约束 softmax normalizer；
- 加 load balancing loss；
- 有时加入噪声或 jitter 促进探索；
- 微调时使用更多数据，避免 MoE 因总参数巨大而过拟合。

DeepSeek V3 提出 auxiliary-loss-free balancing：为每个 expert 维护一个偏置 `b_i`。如果 expert i 最近拿到的 token 太少，就增加 `b_i`；如果太多，就减少 `b_i`。这个偏置只用于路由选择，不作为最终 gating weight。实践中 DeepSeek V3 仍保留了 sequence-wise auxiliary loss，用于控制单条序列内的负载不均。

### 第五步：训练与推理成本

MoE 的 FLOPs 很有吸引力，但真实系统成本不只看 FLOPs。

训练成本包括：

- 激活 expert 的矩阵乘法成本；
- 存储所有 expert 参数的显存成本；
- router 与 load balancing 的额外计算；
- token dispatch 和 combine 的通信成本；
- 负载不均导致的设备空等。

推理成本也类似。虽然每个 token 只激活少数 expert，但所有 expert 的权重必须在某处存放。若 expert 分布在多 GPU 上，就需要 all-to-all communication：先把 token 发送到对应 expert 所在设备，计算后再把结果送回。

如果某个 expert 收到太多 token，系统可能触发 token dropping：超过容量的 token 不经过该 expert，直接靠 residual connection 传下去。这会造成训练或推理结果依赖 batch 中其他请求，从而引入看似奇怪的非确定性。

### 第六步：Expert Parallelism

expert parallelism 是 MoE 的重要系统优势。因为 expert 是天然分块的，可以把不同 expert 放在不同设备上。流程大致是：

1. 每个设备上有一部分 token hidden states；
2. router 决定每个 token 要去哪些 expert；
3. all-to-all 把 token 发到对应设备；
4. expert 执行 FFN；
5. all-to-all 把输出发回原位置；
6. 按 gating weight 合并结果。

这给大模型并行增加了一条轴：除了 data parallelism、tensor/model parallelism、pipeline parallelism，还可以用 expert parallelism。但通信拓扑、batch size、expert 数量、capacity factor 都会影响效率。

一个重要判断标准是：expert 的计算必须“足够厚”，才能掩盖 all-to-all 通信成本。如果每个 expert 太小、每次只收到很少 token，GPU 上会出现许多小矩阵乘法和通信碎片，实际 wall-clock time 可能并不理想。因此现代实现会使用 fused kernels、block-sparse matrix multiplication、MegaBlocks 之类的库，把多个专家计算组织成更适合硬件执行的批量操作。

### 第七步：DeepSeek 系列的演化

DeepSeek MoE 是本讲反复引用的现代案例。早期 DeepSeek MoE 已经采用了两个关键设计：fine-grained experts 和 shared experts，并使用 top-k routing 加辅助负载均衡。DeepSeek V2 在总体结构上变化不大，但规模扩大到数百亿激活参数，并加入 top-M device selection：先限制 token 可以访问的设备集合，再在这些设备内部选 expert，以降低跨设备通信。

DeepSeek V3 继续沿用 MoE 主体，但调整了 router：使用更温和的 sigmoid 风格分数，并引入按 expert 负载在线更新的 bias，实现所谓 auxiliary-loss-free balancing。同时它仍保留 sequence-wise auxiliary loss，以防推理时单条异常序列压垮少数 expert。这个演化说明：MoE 的基本架构并不复杂，真正的进步往往来自训练稳定性、通信控制和负载均衡细节。

## 公式与伪代码

### Top-K MoE 前向传播

```python
# h: [tokens, d_model]
# W_router: [d_model, n_experts]
# experts: list of FFN modules
# k: number of active experts

scores = h @ W_router              # [tokens, n_experts]
probs = softmax(scores, dim=-1)    # or sigmoid + normalization
indices = topk(probs, k)           # selected experts per token

output = zeros_like(h)
for token t:
    for expert i in indices[t]:
        weight = probs[t, i]
        output[t] += weight * experts[i](h[t])

h_next = h + output
```

### Load balancing loss

```text
f_i = fraction of tokens actually routed to expert i
p_i = average router probability assigned to expert i
L_balance = alpha * N * sum_i f_i p_i
```

### DeepSeek V3 风格偏置更新

```text
if load_i < target_load:
    b_i = b_i + gamma
else:
    b_i = b_i - gamma

routing_score_i = router_score_i + b_i
```

注意：`b_i` 用于决定 top-k，但最终合并 expert 输出时通常仍使用原始 gating score。

## 常见误区

1. “Expert 就是人类可解释的领域专家。”
   不一定。expert 是稀疏激活的子网络，可能有专门化，但通常不是清晰的代码/数学/中文专家。

2. “MoE 参数更多，所以一定更贵。”
   总参数更多，但每个 token 只激活少量参数。FLOPs 取决于 activated parameters，而显存取决于 total parameters。训练或部署 MoE 时，必须同时报告总参数、激活参数、每 token 激活 expert 数量，否则很容易误读模型规模。

3. “只要有很多 expert 就会更好。”
   更多 expert 会增加显存和通信复杂度。如果路由不均，很多 expert 会死亡；如果 expert 太细，通信碎片化也会变严重。真正有意义的是“更多可用且被充分训练的 expert”，而不是配置文件里写着更多 expert。

4. “softmax 后 top-k 可以去掉。”
   不能随便去掉。若所有 expert 都参与计算，MoE 就失去稀疏计算优势，训练和推理成本会爆炸。softmax 的主要作用是产生可比较、可加权的分数；top-k 的作用是强制稀疏激活，两者解决的是不同问题。

5. “Load balancing 只是为了 GPU 利用率。”
   不只是。它还防止 expert collapse，让所有参数都得到训练。没有负载均衡时，模型可能看起来仍在下降 loss，但实际上只训练了少数 expert，浪费了大部分容量。

6. “MoE 推理一定比 dense 快。”
   不一定。若模型太大导致权重分布在多机多卡上，通信和调度可能抵消稀疏计算的收益。MoE 适合在足够大规模、足够成熟的推理系统中发挥优势；在单卡小模型场景下，dense model 可能更简单、更稳定。

## 实践练习

1. 手写一个 toy MoE 层：输入二维向量，4 个 expert，每次 top-2 激活，观察不同 token 的路由。
2. 关闭 load balancing loss，记录每个 expert 的 token 数量，看看是否出现 expert collapse。
3. 比较 K=1 与 K=2：训练 loss、计算量、expert 利用率有何变化？
4. 实现 capacity factor：每个 expert 最多接收固定数量 token，超过则 drop，观察输出是否受 batch 组成影响。
5. 阅读 DeepSeek V3 技术报告，找出其 MoE 之外的两个关键设计：MLA（Multi-head Latent Attention）和 MTP（Multi-token Prediction）。

## 总结

MoE 是现代高性能语言模型的重要架构。它的核心思想很简单：用 router 为每个 token 选择少数 expert，从而在不同比例上解耦 total parameters 与 activated parameters。这样可以在相同训练 FLOPs 下获得更低 loss，或者在相似推理计算下使用更大容量模型。

真正困难的是工程与优化：离散 top-k 路由难以训练；expert 可能负载不均；跨设备通信会成为瓶颈；推理时还可能因 token dropping 引入非确定性。现代系统通过 top-k routing、fine-grained experts、shared experts、load balancing loss、expert parallelism、auxiliary-loss-free balancing 等技巧，使 MoE 成为大规模模型训练和部署的主流选择。

## 下一讲衔接

本讲主要讨论 MoE 架构与训练。下一步需要更深入理解大模型系统层面的并行策略：data parallelism、tensor parallelism、pipeline parallelism、expert parallelism 如何组合，以及通信带宽、显存、batch size 如何共同决定真实训练吞吐。理解这些系统细节后，才能真正判断 MoE 在某个硬件集群上是否“划算”。如果你继续学习系统部分，建议始终把“数学上的 FLOPs”和“机器上的时间”分开思考：MoE 的论文曲线常常展示前者，而工程成败往往取决于后者。
