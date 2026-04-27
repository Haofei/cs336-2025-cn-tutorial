# Stanford CS336 Lecture 7 中文教程：Parallelism 1

## 学习目标

读完本讲，你应该能够：

1. 解释为什么训练大语言模型（LLM）必须从单 GPU 扩展到多 GPU、多节点甚至整个数据中心。
2. 区分 data parallelism、model parallelism、tensor parallelism、pipeline parallelism 与 activation/sequence parallelism 的核心思想。
3. 理解 all-reduce、reduce-scatter、all-gather 等 collective communication 在分布式训练中的作用。
4. 判断不同并行策略在计算、显存、通信带宽和 batch size 之间的取舍。
5. 根据硬件拓扑给出一个合理的 LLM 训练并行配置思路。

## 本讲地图

本讲讨论的不是“如何让单个 kernel 更快”，而是“模型太大、训练太慢时，如何把训练任务切到很多机器上”。主线如下：

- 先看动机：单 GPU 的 FLOPs 和显存都不够，必须使用 multi-machine parallelism。
- 再看通信基础：GPU 之间并不是等价连接，节点内 NVLink/NVSwitch 很快，跨节点 InfiniBand 或以太网更慢。
- 然后介绍 data parallelism：复制模型，切分数据，并同步梯度；再用 ZeRO/FSDP 降低显存占用。
- 接着介绍 model parallelism：不再复制完整模型，而是把模型本身切开，包括 pipeline parallelism 和 tensor parallelism。
- 最后讨论 activation memory、sequence parallelism、通信/计算权衡，以及实际大模型训练中的组合规则。

## 核心概念

### 1. 为什么 LLM 训练需要并行？

大模型训练有两个硬约束：compute 和 memory。

Compute 指训练所需的总浮点运算量。模型越大、token 越多，训练所需 FLOPs 越高。单个 GPU 即使每代都变快，也无法立刻满足前沿模型的训练需求。

Memory 指 GPU 显存。参数本身只是显存的一部分。训练时还要存：

- parameters：模型参数，通常 BF16/FP16，约 2 bytes/parameter；
- gradients：梯度，约 2 bytes/parameter；
- optimizer states：Adam 的一阶、二阶矩，以及 master weights，常常占更多；
- activations：前向传播中为反向传播保存的中间结果。

一个常见估算是，使用 Adam 训练时，单个参数可能需要约 16 bytes 的训练状态。因此，7B、70B、甚至更大模型无法简单放进一张 GPU。

### 2. 通信原语：collective communication

分布式训练大量依赖 collective communication：

- all-reduce：每个 rank 都有一份张量，先做求和/平均等 reduce，再把结果发回所有 rank。数据并行同步梯度常用它。
- reduce-scatter：先 reduce，再把结果按 shard 分给不同 rank。
- all-gather：每个 rank 有一个 shard，把所有 shard 收集起来，使每个 rank 都得到完整张量。
- broadcast：一个 rank 的数据复制给所有 rank。

重要等价关系：

```text
all-reduce ≈ reduce-scatter + all-gather
```

在 bandwidth-bound 场景下，二者通信量近似等价。这个事实解释了 ZeRO 为什么能在不明显增加通信量的情况下节省显存。

### 3. 硬件拓扑决定并行策略

并行算法不能脱离硬件。典型 NVIDIA 训练节点中，一台机器可能有 8 张 GPU，节点内通过 NVLink/NVSwitch 高速互连；跨机器则通过 InfiniBand 等网络连接，带宽和延迟明显更差。

因此经验规则是：

- bandwidth-hungry 的 tensor parallelism 通常放在节点内；
- data parallelism 可以跨更慢的网络，因为每个 step 只同步一批梯度或参数 shard；
- pipeline parallelism 通信的是 activation，且多为 point-to-point，有时适合跨节点或跨较慢链路。

## 逐步教程

开始设计并行方案时，可以先问三个问题。第一，单张卡是否能放下模型参数、梯度、优化器状态和峰值 activation；如果不能，必须先切模型或切训练状态。第二，通信链路在哪里最快；节点内高速互连适合频繁同步，跨节点慢链路应尽量少同步或只做点到点传输。第三，当前训练允许的有效 batch size 有多大；如果 batch 已经接近临界值，继续用数据并行堆 GPU 可能只会增加通信而不提升收敛速度。

一个实用思路是把并行策略看成不同维度的切分。Data parallelism 切的是样本维度，优先提高吞吐；tensor parallelism 切的是每层矩阵宽度，优先解决单层太宽、参数太大的问题；pipeline parallelism 切的是网络深度，优先解决层数太多、整网放不下的问题；sequence parallelism 切的是序列维度，专门处理 activation 中难以被 tensor parallelism 降下来的部分。实际系统通常不是四选一，而是把它们叠加成所谓 3D/4D parallelism。

## Step 1：Data Parallelism：复制模型，切分数据

Data parallelism（数据并行）是最自然的并行方式：每张 GPU 保存完整模型参数，但处理不同的数据样本。

设全局 batch size 为 B，GPU 数为 M，则每张 GPU 处理 B/M 个样本。每张 GPU 独立做 forward/backward，得到本地梯度，然后用 all-reduce 对梯度求平均，最后每张 GPU 执行相同的 optimizer step。

SGD 更新可写作：

```text
θ_{t+1} = θ_t - η · (1/B) · Σ_{i=1}^B ∇ℓ(x_i; θ_t)
```

数据并行只是把这个求和拆给不同 GPU 做。

优点：

- 计算扩展性好：batch 足够大时，更多 GPU 可以处理更多样本。
- 实现简单，对模型结构不敏感。

缺点：

- 显存不省：每张 GPU 都存完整参数、梯度和 optimizer state。
- 通信量与参数量相关：每个 step 需要同步梯度。
- 受 batch size 限制：GPU 数不能无限超过有效 batch size。

### Batch size 是一种资源

当 batch size 很小时，数据并行无法继续扩展，因为每张 GPU 至少要有可用样本。即使能增大 batch，优化效果也存在 critical batch size：超过某个点后，继续增大 batch 对训练速度的收益递减，因为瓶颈从“梯度噪声”变成了“梯度更新步数”。

所以，batch size 不是随便可用的无限资源。它要在 data parallelism、pipeline parallelism 和 gradient accumulation 之间分配。

## Step 2：ZeRO/FSDP：让数据并行也节省显存

朴素数据并行浪费显存，因为每张 GPU 都复制完整训练状态。ZeRO（Zero Redundancy Optimizer）逐步切分这些状态。

### ZeRO Stage 1：切 optimizer state

每张 GPU 仍保存完整 parameters 和 gradients，但 Adam 的 optimizer states 被切成 shards。每张 GPU 只负责更新自己拥有的参数 shard。

流程：

1. 每张 GPU 计算完整梯度。
2. 用 reduce-scatter 把梯度按参数 shard 汇总到对应 GPU。
3. 每张 GPU 用自己的 optimizer state 更新对应参数 shard。
4. 用 all-gather 把更新后的参数 shard 收集回每张 GPU。

通信上，reduce-scatter + all-gather 近似等价于原来的 all-reduce，因此 Stage 1 几乎是“免费”的显存优化。

### ZeRO Stage 2：再切 gradients

Stage 2 不再保留完整梯度。反向传播时，每算出一层梯度，就立刻 reduce 到负责该参数 shard 的 GPU，并释放本地临时梯度。

这样避免在显存中实例化完整 gradient vector。通信总量仍接近原始数据并行，但调度更复杂。

### ZeRO Stage 3 / FSDP：连 parameters 也切

FSDP（Fully Sharded Data Parallel）基本对应 ZeRO Stage 3：parameters、gradients、optimizer states 全部 sharded。

核心思想是“按需 all-gather 参数”：

1. 某层 forward 前，all-gather 该层参数。
2. 完成该层计算后释放参数。
3. backward 时再次按需 all-gather 参数。
4. 算出梯度后 reduce-scatter 到负责该 shard 的 GPU。

通信量从约 2×parameters 增加到约 3×parameters，但通过 overlap communication and computation（通信与计算重叠）、prefetch（预取）等技术，实际开销可接受。

FSDP 的优势是通用：它不需要深入理解 Transformer 结构，常作为大模型训练的默认显存优化手段。

## Step 3：Model Parallelism：把模型本身切开

当模型或 activation 仍然放不下时，需要 model parallelism（模型并行）。与 FSDP 不同，模型并行的目标不是“临时收集完整参数”，而是让模型的不同部分固定生活在不同 GPU 上，通信的主要对象变成 activations。

本讲重点介绍两种：pipeline parallelism 和 tensor parallelism。

## Step 4：Pipeline Parallelism：按层切模型

Pipeline parallelism（流水线并行）沿深度方向切模型：例如 GPU0 放前几层，GPU1 放中间层，GPU2 放后几层。forward 时 activation 从前往后传，backward 时 activation gradients 从后往前传。

朴素做法会产生严重 bubble：某个时刻只有一张 GPU 在工作，其他 GPU 空闲。为减少 bubble，通常把 batch 切成多个 micro-batches，让不同 micro-batch 像工厂流水线一样同时处在不同 stage。

bubble 开销的常见近似：

```text
bubble_ratio ≈ (pipeline_stages - 1) / micro_batches
```

因此 micro-batch 越多，pipeline 越满。但这又消耗 batch size 资源。

优点：

- 参数和部分 activation 按层分布，显存扩展好。
- 通信多为相邻 stage 的 point-to-point activation 传输。
- 适合跨较慢网络链路使用。

缺点：

- 调度复杂，尤其是 1F1B、interleaved pipeline、zero-bubble pipeline 等高级策略。
- bubble 会降低 GPU 利用率。
- 工程实现非常难，常需要深度介入 autograd 和 runtime 调度。

Zero-bubble pipeline 的一个技巧是把 backward 拆成两类工作：

- B：反传 activation gradient，存在严格依赖；
- W：计算 weight gradient，依赖较少，可以挪到 bubble 里执行。

这样可用原本空闲的时间计算参数梯度，提高利用率。

## Step 5：Tensor Parallelism：按矩阵宽度切模型

Tensor parallelism（张量并行）沿宽度方向切矩阵乘法。Transformer 的主要计算来自大矩阵乘法，因此可以把权重矩阵切成多个子矩阵，由多张 GPU 分别计算 partial results，再通过 collective communication 合并。

例如 MLP 中：

```text
Y = GeLU(XA)
Z = YB
```

可以把 A、B 切成 A1/A2、B1/B2。每张 GPU 处理一部分矩阵乘法，必要时 all-reduce 合并结果。

优点：

- 不消耗 batch size。
- 没有 pipeline bubble。
- 对 Transformer 这类以矩阵乘法为主的模型很自然。

缺点：

- 每层都可能有同步屏障。
- 通信的是 activation，频率高、带宽需求大。
- 通常只适合节点内高速互连，例如 8 张 GPU 的 NVLink/NVSwitch 环境。

经验规则：tensor parallel size 常设为节点内 GPU 数，例如 8。超过节点后跨慢链路做 tensor parallel，吞吐可能明显下降。

## Step 6：Activation 与 Sequence Parallelism

前面主要处理了参数、梯度和 optimizer state，但 activation memory 也会成为大问题。它的峰值通常出现在反向传播早期：此时许多前向 activation 还没释放，梯度又开始累积。序列越长、batch 越大，这部分越明显；长上下文训练尤其容易被 activation 卡住。

Transformer 每层 activation 近似包含两类项：

```text
activation_memory_per_layer ≈ S · B · H · 34 + 5 · A · S² · B
```

其中 S 是 sequence length，B 是 batch size，H 是 hidden size，A 是 attention heads。右侧 S² 项来自 attention softmax 等二次复杂度部分，可通过 FlashAttention 和 recomputation 大幅降低。

Tensor parallelism 能切分许多矩阵乘相关 activation，但 layer norm、dropout、残差流输入等 pointwise operations 仍可能保留完整 activation。Sequence parallelism 的做法是沿 sequence dimension 切这些 pointwise activation：不同 GPU 负责不同 token position。

这会引入额外 all-gather / reduce-scatter，但可进一步降低 activation memory。结合 activation recomputation，常能用更多计算换取更低显存，从而允许更大 batch 或更大模型。

## 公式与伪代码速查

### 数据并行训练伪代码

```python
for batch in data:
    local_batch = shard(batch, rank)
    loss = model(local_batch)
    loss.backward()
    all_reduce(model.gradients)   # 同步所有 rank 的梯度
    optimizer.step()
    optimizer.zero_grad()
```

### FSDP 思想伪代码

```python
for layer in model.layers:
    weights = all_gather(layer.weight_shards)
    activation = layer.forward(activation, weights)
    free(weights)

for layer in reversed(model.layers):
    weights = all_gather(layer.weight_shards)
    grad = layer.backward(grad, weights)
    reduce_scatter(layer.grad_shards)
    free(weights)

optimizer.step_on_local_shards()
```

### 并行策略组合规则

```text
先保证模型能放进显存：
    1. 节点内优先用 tensor parallelism
    2. 仍放不下时，用 FSDP/ZeRO-3 或 pipeline parallelism

模型能放下后：
    3. 剩余 GPU 用 data parallelism 扩展吞吐
    4. 如果通信太频繁，用 gradient accumulation 增大有效 batch
```

## 常见误区

1. “GPU 越多越快。”
   错。通信、同步、pipeline bubble 和 batch size 限制都会让扩展效率下降。

2. “Data parallelism 能解决显存问题。”
   朴素 DDP 不能。只有 ZeRO/FSDP 这类 sharding 技术才能显著减少参数相关显存。

3. “FSDP 就是 model parallelism。”
   不完全是。FSDP 虽然切参数，但计算时会按需 all-gather 参数；model parallelism 更强调参数固定分布，主要传 activation。

4. “Tensor parallelism 可以随便跨节点扩展。”
   通常不行。它每层都通信，对带宽和延迟极敏感，最好放在节点内高速互连上。

5. “Pipeline parallelism 概念简单，所以实现简单。”
   恰好相反。高效 pipeline 调度、micro-batch、1F1B、zero-bubble、autograd 介入都很复杂。

6. “Activation memory 不重要。”
   对长序列、大 batch、大模型训练，activation 可能成为主要显存瓶颈。

## 练习

1. 假设一个模型有 P 个参数，使用 Adam 训练，每个参数训练状态约 16 bytes。估算 7B 参数模型仅参数相关训练状态需要多少显存。

2. 用自己的话解释为什么：

```text
all-reduce ≈ reduce-scatter + all-gather
```

并说明这个等价关系如何帮助 ZeRO Stage 1。

3. 一个 8 GPU 节点内 NVLink 很快，跨节点网络较慢。你会把 tensor parallelism、pipeline parallelism、data parallelism 分别放在哪些范围？为什么？

4. 如果 pipeline stages = 8，micro-batches = 32，估算 bubble_ratio。micro-batches 减半后会发生什么？

5. 解释 FSDP 与 tensor parallelism 的主要区别：它们分别通信什么？分别依赖什么硬件特性？

## 总结

LLM 训练并行的本质是同时管理四种稀缺资源：显存、计算、通信带宽/延迟和 batch size。Data parallelism 简单且适合扩展吞吐，但受 batch size 和显存复制限制；ZeRO/FSDP 通过 sharding optimizer states、gradients、parameters 让数据并行也能节省显存；pipeline parallelism 沿层切模型，通信较温和但有 bubble 和复杂调度；tensor parallelism 沿矩阵宽度切模型，不消耗 batch size，但需要高速互连；sequence parallelism 和 activation recomputation 则进一步处理 activation memory。

实际训练中没有单一最佳方案。常见做法是：节点内用 tensor parallelism，必要时结合 sequence parallelism；模型仍放不下时加入 FSDP 或 pipeline parallelism；最后用 data parallelism 吃掉剩余 GPU，并用 gradient accumulation 调节通信频率。理解这些策略的通信对象和硬件需求，是设计高效 LLM 训练系统的关键。

从工程角度看，好的并行方案不是追求某个术语，而是让昂贵 GPU 尽量少等待。通信可以被预取和重叠，显存可以用分片和重计算换取，batch size 可以用来摊薄同步成本，但每个选择都会把压力转移到别的资源上。Lecture 7 的核心结论正是：大规模训练是系统设计问题，算法、硬件拓扑、优化器状态、activation 生命周期和调度策略必须一起考虑。
