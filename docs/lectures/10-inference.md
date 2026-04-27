# Stanford CS336 Lecture 10：LLM Inference 中文教程

本讲讨论的是 **inference（推理/服务时生成）**：给定一个已经训练好的固定模型，根据用户的 prompt 生成 response。训练通常是一次性的大成本，而推理会在聊天、代码补全、批处理、模型评测、test-time compute、RL 采样等场景中被反复调用，因此推理效率往往直接决定产品成本和用户体验。

## 1. 推理要优化什么？

衡量 LLM inference 主要看三个指标：

- **TTFT（Time To First Token，首 token 延迟）**：用户从提交 prompt 到看到第一个输出 token 的等待时间。它主要由 prompt 的处理时间决定，对交互式应用非常重要。
- **Latency（延迟）**：生成开始后 token 到达的速度，常理解为每个 token 需要多久，或用户感受到的流式输出速度。
- **Throughput（吞吐）**：系统单位时间能生成多少 token，通常以 tokens/s 计。批处理任务更关心吞吐，而聊天产品还要兼顾低延迟。

低 latency 和高 throughput 经常冲突：为了提高吞吐，系统倾向于把更多请求合成大 batch；但 batch 越大，单个用户等待本轮计算完成的时间也可能越长。

## 2. 为什么推理不同于训练？

Transformer 训练时，整段序列的 token 已知，可以在 sequence 维度上并行计算；矩阵乘法足够“大”，容易把 GPU/TPU 的算力打满。

推理时，尤其是自回归生成（autoregressive generation），第 t 个 token 必须依赖前面已经生成的 token，因此 decode 阶段只能一步一步来。每一步通常只为每个序列生成 1 个 token，这会让很多计算退化成较“瘦”的矩阵乘法或矩阵-向量乘法，难以充分利用硬件算力。

理解这一点需要一个关键词：**arithmetic intensity（算术强度）**，即每读取/写入一个 byte 内存能做多少 FLOPs。若算术强度高，通常是 compute-bound（受算力限制）；若算术强度低，通常是 memory-bound（受内存带宽限制）。H100 这类 GPU 的算力很强，但 HBM memory bandwidth 仍有限，所以如果每次只读大量参数和 KV cache，却做不了多少计算，GPU 就会“等内存”。

在训练或 prefill 中，batch size × sequence length 足够大，算术强度高；在逐 token decode 中，T=1，算术强度显著下降，推理就容易变成 memory bandwidth bottleneck。

## 3. Prefill 与 Decode：推理的两个阶段

LLM 推理可分为两个阶段：

### 3.1 Prefill（提示词预填充）

给定 prompt，模型一次性处理所有 prompt token，计算每层 attention 所需的 key/value，并得到下一个 token 的 logits。这个阶段类似训练的 forward pass：序列维度可并行，通常 compute-bound，速度相对较快。TTFT 很大程度上来自 prefill，特别是 prompt 很长时。

### 3.2 Decode / Generation（逐 token 生成）

模型每次根据已有上下文生成一个新 token，然后把它追加到上下文中，再继续下一步。这个阶段顺序性强，T=1，通常 memory-bound，是推理系统最难优化的部分。

## 4. KV Cache：避免重复计算的核心机制

最朴素的生成方法是：每生成一个 token，就把完整上下文重新送进 transformer。这会重复计算所有历史 token 的 key/value，复杂度很差。

**KV cache（Key-Value cache）** 的思想是：在 causal transformer 中，历史 token 的 key/value 不会因为新 token 到来而改变，所以可以缓存起来。Prefill 时为 prompt 建立 KV cache；decode 时只为新 token 计算新的 K/V，并把它追加进 cache。这样每一步不再重算完整前缀，而是读取历史 KV cache 来做 attention。

KV cache 的形状大致与以下因素成正比：

- batch size：同时服务多少序列；
- sequence length：每条序列已经有多少 token；
- number of layers：每层都要存；
- number of KV heads：key/value head 数量；
- head dimension：每个 head 的维度；
- K 和 V 两份数据，以及数据精度（如 BF16）。

因此 KV cache 很容易成为显存大户。长上下文、多并发、大 batch 都会迅速增加显存占用。

## 5. Memory Bandwidth 为什么是瓶颈？

在 MLP 层中，不同请求会共享同一组模型权重。batch 越大，读一次权重可以服务更多 token，所以 batch 可以提高算术强度。

但 attention 的 decode 更麻烦：每条序列都有自己的 KV cache。即使 batch 变大，每个请求仍要读取自己独有的历史 K/V，无法像 MLP 权重那样在 batch 内大量复用。因此 attention decode 的算术强度接近常数级，常常低到 memory-bound。

这解释了推理优化中的一个核心原则：**减少需要从 HBM 读取/写入的数据量，往往比单纯减少 FLOPs 更重要。**

## 6. Batch Size、Latency 与 Throughput 的权衡

增大 batch size 可以提高吞吐：一次 decode step 同时为 B 个请求各生成 1 个 token。但代价是：

1. 每一步要处理更多序列，单步 latency 可能上升；
2. 每条序列都要保留 KV cache，显存占用随 batch 增大；
3. 吞吐提升存在边际递减，且最终受显存容量限制。

如果只服务一个用户，latency 可以很低，但 GPU 利用率差；如果聚合很多用户，throughput 更好，但用户等待可能变长。这就是 LLM serving 的基本张力。

一种简单但有效的扩展方式是 **replication（复制模型）**：在多张 GPU 上各放一份模型，不需要训练时那样复杂的梯度同步，latency 基本不变，总吞吐近似随副本数增加。若模型太大，才需要 tensor parallelism、pipeline parallelism 或 KV cache sharding 等更复杂策略。

## 7. 减小 KV Cache 的架构技巧

因为 decode 主要受 KV cache 读写限制，很多现代架构改动都可以理解为“让 KV cache 更小”。

### 7.1 GQA：Grouped-Query Attention

传统 multi-head attention 中，query heads、key heads、value heads 数量相同。**MQA（Multi-Query Attention）** 极端地让所有 query 共享一组 K/V，但可能表达能力不足。**GQA（Grouped-Query Attention）** 是折中：多个 query heads 共享较少的 KV heads。

这样不会减少 query 的表达能力太多，却能显著减少 KV cache 的 head 数量。KV cache 变小后，显存占用降低，memory transfer 减少，latency 和 throughput 都会改善；同时还能允许更大的 batch size。

### 7.2 MLA：Multi-head Latent Attention

DeepSeek 系列提出的 **MLA（Multi-head Latent Attention）** 不一定减少 KV head 个数，而是把 K/V 投影到更低维的 latent space 中缓存。也就是说，不缓存完整高维 K/V，而缓存压缩后的表示，需要时再恢复或参与计算。它从“维度”上缩小 KV cache，目标同样是降低显存和带宽压力。

### 7.3 CLA：Cross-Layer Attention

**CLA（Cross-Layer Attention）** 在层之间共享 K/V 表示。GQA 是跨 head 共享，CLA 是跨 layer 共享。由于 KV cache 原本要为每一层保存一份，跨层共享可以进一步减少 cache size，但需要在模型质量和效率之间做权衡。

### 7.4 Local / Sliding Window Attention

**Local attention（局部注意力）** 或 **sliding window attention（滑动窗口注意力）** 只关注最近 K 个 token。这样生成很长序列时，超出窗口的 KV 可以丢弃，cache 不再随总序列长度线性增长，而近似随窗口大小固定。

问题是纯局部注意力会损害长程依赖能力。因此实际模型常用 hybrid design：大多数层用 local attention，少数层保留 full/global attention；或者结合 KV sharing、GQA 等技巧。

## 8. 更激进的方向：改变 Transformer

如果 full attention 的 KV cache 是根本瓶颈，一个方向是减少 full attention 层，甚至换掉部分 transformer 结构。

- **State Space Models（SSM）/ Mamba**：用类似 RNN 的状态表示替代随序列增长的 KV cache，使推理状态接近常数大小。挑战是保持语言建模能力，尤其是 associative recall 这类需要精确检索远处信息的任务。
- **Linear Attention（线性注意力）**：通过核函数或特征映射改写 attention，使复杂度从二次变成线性，并具有类似 recurrent state 的实现形式。现代模型常把 linear/local/full attention 混合使用。
- **Diffusion language models（扩散式语言模型）**：不再严格自回归逐 token 生成，而是并行生成一段文本并反复 refine。这样更容易打满硬件，但文本质量和通用性仍是研究问题。

这些方法说明：推理优化不只是系统工程，也会反过来推动模型架构设计。

## 9. Quantization、Pruning 与 Distillation

**Quantization（量化）** 通过降低数值精度减少显存和带宽。例如从 BF16 降到 FP8、INT8，甚至 INT4。由于推理常 memory-bound，减少每个参数或 KV 元素的 byte 数能直接改善速度和容量。但低精度会带来误差，尤其是大模型中存在 outliers（异常大的激活或权重）时。常见做法包括 post-training quantization、对 outliers 单独保留高精度、activation-aware quantization 等。

**Pruning（剪枝）** 则是删除不重要的层、head 或 hidden dimensions，让模型结构本身更小。剪枝后模型通常会变差，因此常结合 **distillation（蒸馏）**：用原始大模型作为 teacher，把能力迁移到被剪过或更小的 student 模型中。

这些方法通常是 lossy 的：速度和成本更好，但需要验证质量是否可接受。

## 10. Speculative Decoding：用小模型加速大模型

**Speculative decoding / speculative sampling（投机解码/投机采样）** 的关键观察是：验证一串给定 token 比逐个生成它们更快。因为验证可以像 prefill 一样并行，而生成必须自回归。

流程如下：

1. 用一个便宜的 **draft model（草稿模型）** 先自回归生成 K 个候选 token；
2. 用昂贵的 **target model（目标模型）** 并行计算这些 token 的概率；
3. 按照接受-拒绝规则决定保留多少草稿 token；
4. 若某个 token 被拒绝，则从 target model 的校正分布中采样，然后继续。

数学上，这种方法可以保证输出分布与直接从 target model 采样一致，即“exact sampling from target model”，前提是接受-拒绝步骤实现正确。加速效果取决于 draft model 与 target model 的接近程度：draft 越准，接受率越高，速度越快。Medusa、EAGLE 等方法都是围绕更好的 draft 或并行草稿生成展开。

## 11. Serving Systems：真实流量下的系统问题

训练时 batch 通常是整齐的 dense token block；服务时请求是动态的：到达时间不同、prompt 长度不同、生成长度不同、有些共享 prefix、有些很快结束。这要求 serving system 动态调度。

### 11.1 Continuous Batching

**Continuous batching（连续批处理）** 不等待一个 batch 全部完成再接新请求，而是在每个 decode step 后把控制权交回 scheduler：完成的请求退出，新来的请求加入。这样能减少 GPU 空转，提高吞吐。

### 11.2 Selective Batching

不同请求长度不同，attention 部分很难完全整齐地 batch；但 MLP 部分不依赖序列之间交互，可以把不同长度的 token flatten 到 batch 维度一起算。这就是 **selective batching（选择性批处理）** 的思路。

### 11.3 PagedAttention 与 vLLM

KV cache 动态增长会导致显存碎片：请求生成长度未知，提前分配会浪费；请求结束后又留下不连续空洞。**PagedAttention** 借鉴操作系统虚拟内存，把 KV cache 切成固定大小 block/page，请求的逻辑连续上下文可以映射到物理上不连续的显存块。这样减少碎片，提高显存利用率。

如果多个请求共享 prefix，还可以使用 **copy-on-write（写时复制）**：共享相同 KV block，只有在后续生成分叉时才复制，进一步节省内存。

## 12. 采样与解码策略

前面的内容主要讨论“如何更快地算出下一个 token 的 logits”，但真正生成文本还需要 **decoding strategy（解码策略）**：从 logits 或概率分布中选择 token。

最简单的是 **greedy decoding（贪心解码）**，每一步都选概率最高的 token。它稳定、便宜、可复现，但容易输出模板化文本，遇到多步推理或创意写作时可能过早锁死路径。**beam search（束搜索）** 会同时保留多个候选序列，在传统机器翻译中常见；但对开放式 LLM 对话，它可能产生重复、保守的答案，且会增加计算和内存压力。

更常用的是随机采样。**temperature（温度）** 会缩放 logits：温度低，分布更尖锐，输出更确定；温度高，分布更平坦，输出更多样。**top-k sampling** 只在概率最高的 k 个 token 中采样，截掉长尾低概率 token；**top-p / nucleus sampling（核采样）** 则选择累计概率达到 p 的最小 token 集合，动态决定候选集合大小。实际服务中还常加入 repetition penalty、frequency penalty、stop sequences、最大输出长度等规则，以控制重复、终止和成本。

这些策略本身不改变 transformer 的主要计算瓶颈，但会影响生成长度、接受率和用户感知质量。例如 speculative decoding 的速度取决于 draft token 被 target model 接受的概率；温度越高、采样越随机，小模型草稿越难预测大模型，接受率可能下降。因此推理系统往往需要把模型、采样参数和服务目标一起调优。

## 13. 总结

LLM inference 的核心难点来自自回归 decode：每次只生成一个 token，难以并行，且需要反复读取模型权重和每条序列专属的 KV cache，因此通常受 memory bandwidth 限制。Prefill 相对容易并行，decode 才是主要瓶颈。

优化路径可以分为几类：

- 系统层面：continuous batching、selective batching、PagedAttention、模型复制与并行；
- 架构层面：GQA、MLA、CLA、local attention、SSM、linear attention、diffusion models；
- 模型压缩：quantization、pruning、distillation；
- 解码算法：speculative decoding，用小模型草稿和大模型验证换取无损加速。

最终目标不是“把某个固定 transformer 跑得更快”这么窄，而是在给定 latency、throughput、显存和成本预算下，交付尽可能高质量的模型输出。推理效率已经成为现代 LLM 架构、算法和系统共同设计的核心驱动力。
