# Stanford CS336 Lecture 8：并行训练（二）中文教程

本讲继续讨论大模型训练的系统层问题：当单张 GPU 已经无法容纳模型、优化器状态或足够大的 batch 时，如何把计算和数据分布到多张 GPU、甚至多个节点上，并尽量避免通信成为瓶颈。核心原则仍然和单 GPU 优化一致：让昂贵的计算单元尽可能忙起来，提高算术强度，减少不必要的数据搬运。

## 1. 多 GPU 训练的硬件视角

可以把训练集群理解成一个层次化存储与通信系统：

- GPU 内部：SM 执行计算，L1/shared memory 很快但很小，HBM 容量更大但更慢。
- 单节点多 GPU：GPU 之间通过 PCIe 或 NVLink/NVSwitch 连接。NVLink 是 NVIDIA 为 GPU 间高带宽通信设计的专用链路，通常远快于 PCIe。
- 多节点：节点之间还要经过网卡、交换机等网络设备，带宽更低、延迟更高。

因此，跨 GPU 通信比访问本地 HBM 更贵，跨节点通信又比同节点通信更贵。分布式训练工程的目标不是“完全避免通信”，而是：

1. 减少通信量；
2. 把通信安排在合适的位置；
3. 尽量与计算重叠；
4. 根据硬件拓扑选择并行策略。

在 NVIDIA 生态中，底层通信通常由 NCCL 负责。NCCL 会根据 GPU 拓扑选择通信路径，并把高层 collective 操作编译成底层 CUDA kernel 和数据传输。PyTorch 的 `torch.distributed` 则在 Python 层提供更方便的接口，例如 `all_reduce`、`all_gather`、`reduce_scatter` 等。

## 2. Collective 通信原语

分布式训练里经常使用 collective operations。假设一共有 `world_size` 个进程或设备，每个设备编号为一个 `rank`。

常见原语包括：

- `broadcast`：一个 rank 上的数据复制到所有 rank。
- `scatter`：一个 rank 上的不同切片分发到不同 rank。
- `gather`：多个 rank 的数据收集到一个 rank。
- `reduce`：多个 rank 的数据先做求和、平均、最大值等归约，再放到一个 rank。
- `all_gather`：每个 rank 都得到所有 rank 的数据拼接结果。
- `reduce_scatter`：先对各 rank 的输入做归约，再把归约后的不同切片分给不同 rank。
- `all_reduce`：每个 rank 都得到所有 rank 数据归约后的完整结果。

一个重要等价关系是：

```text
all_reduce = reduce_scatter + all_gather
```

例如数据并行训练中，每张 GPU 在不同数据切片上算出梯度，然后通过 `all_reduce` 求平均，使所有 GPU 的参数更新保持一致。很多高级策略会把 `all_reduce` 拆成 `reduce_scatter` 和 `all_gather`，以便减少峰值显存或更好地与计算重叠。

使用这些原语时要特别注意同步关系。collective 操作要求参与同一个 process group 的 rank 都以一致的顺序调用；如果某个 rank 少调用一次 `all_reduce`，其他 rank 可能永久等待，表现为程序 hang 住。

## 3. Benchmark：不要只看理论带宽

H100 的 NVLink 理论带宽很高，但实际训练中能达到多少，取决于张量大小、rank 数量、通信模式、NCCL 算法、节点拓扑和是否跨节点。课程中用大张量测试 `all_reduce`，得到的实际带宽明显低于硬件标称值；`reduce_scatter` 的表现也可能与简单估算不完全一致。

工程上应养成习惯：

- 对目标集群实际 benchmark，而不是只读产品规格；
- 分别测试同节点和跨节点通信；
- 关注不同消息大小下的吞吐和延迟；
- 用 warmup、`torch.cuda.synchronize()`、barrier 等方式避免计时误差；
- 区分“算法需要传输的字节数”和“墙钟时间”。

通信性能很难仅凭公式精确预测，因为 NCCL 会使用 ring、tree、分层通信、网络内归约等实现细节。因此实际调优必须依赖 profiling 和 benchmark。

## 4. 数据并行 DDP：切 batch，同步梯度

数据并行是最直观的并行方式：每张 GPU 保存完整模型，但处理不同的 batch 切片。每个 rank 独立执行 forward 和 backward，得到本地梯度，然后对所有参数梯度做 `all_reduce` 平均。

训练步骤可以概括为：

1. 把全局 batch 按 rank 切成 local batch；
2. 每个 rank 用同一份模型参数处理自己的 local batch；
3. backward 得到本地梯度；
4. 对每个参数的梯度执行 `all_reduce(mean)`；
5. 每个 rank 用相同梯度执行 optimizer step。

这样每个 rank 的 loss 可能不同，因为数据不同；但同步梯度后，参数仍保持一致。

DDP 的优点是实现简单、计算扩展性好；缺点是每张 GPU 都要保存完整参数、梯度和优化器状态。当模型变大时，显存首先成为瓶颈。另一个工程要点是，`all_reduce` 本身也是同步点；如果某些 rank 速度较慢，其他 rank 会等待，这就是 straggler 问题。

## 5. ZeRO 与 FSDP：切参数、梯度和优化器状态

为了突破 DDP 的显存限制，可以对模型状态做分片。ZeRO（Zero Redundancy Optimizer）把训练状态拆成几个层级：

- ZeRO-1：分片 optimizer states，例如 Adam 的一阶、二阶动量；
- ZeRO-2：进一步分片 gradients；
- ZeRO-3：连 parameters 也分片。

FSDP（Fully Sharded Data Parallel）可以看作 PyTorch 中类似 ZeRO-3 的实现：每个 rank 只常驻一部分参数，需要计算某层时，通过 `all_gather` 临时收集该层完整参数；反向传播结束后，用 `reduce_scatter` 把梯度归约并重新切片。

FSDP 的基本权衡是：

- 好处：显著降低每张 GPU 的常驻显存，使更大模型可以训练；
- 代价：forward/backward 期间要频繁 `all_gather` 参数和 `reduce_scatter` 梯度；
- 工程重点：合理设置 wrapping 粒度、prefetch、bucket size，避免通信碎片化。

如果切得太细，每层通信开销和调度开销会变大；如果切得太粗，峰值显存又会上升。实际训练中常按 Transformer block 作为 FSDP 单元，并结合混合精度、activation checkpointing 和 CPU/offload 策略。

## 6. Activation checkpointing：用重算换显存

反向传播需要 forward 期间保存 activations。长序列、大 batch、深层 Transformer 都会让 activation 显存非常大。Activation checkpointing 的思路是：只保存部分中间结果，backward 时重新计算缺失的 activations。

这是一种典型的“计算换存储”策略：

- 不 checkpoint：计算少，但保存很多 activation；
- checkpoint 全部或部分 block：显存低，但 backward 需要额外 forward 重算。

实践中不应盲目重算所有东西，而是选择合适粒度。通常可以在 Transformer block 级别 checkpoint；对于 matmul 后紧跟的简单 pointwise 操作，可能没必要保存所有中间值，重算成本很低。Checkpointing 经常与 FSDP/ZeRO 搭配，因为参数、梯度、优化器状态被分片后，activation 可能成为新的显存瓶颈。

## 7. Tensor Parallel：切隐藏维度，频繁 collective

张量并行切的是模型内部的矩阵维度，而不是 batch。以 MLP 的线性层为例，权重矩阵可以按列或按行切到多个 rank 上。每个 rank 只保存一部分权重并计算一部分输出。

课程中的简化例子是：每个 rank 拥有每一层的一部分隐藏维度。计算出局部 activation 后，需要通过 `all_gather` 把所有 rank 的 activation 拼回完整隐藏向量，再进入下一层。

Tensor Parallel 的特点：

- 优点：单层超大矩阵可以分布到多张 GPU 上；
- 缺点：每层或每几个算子都可能需要 collective；
- 适用硬件：强依赖高速互联，通常优先放在同节点 NVLink/NVSwitch 内。

Transformer 中更常见的做法是对 attention heads、MLP intermediate dimension 或 vocabulary projection 做切分。不同切法对应不同 collective：有时 forward 需要 `all_reduce`，有时 backward 需要；有时可以用 `reduce_scatter` + `all_gather` 优化显存和通信。

## 8. Pipeline Parallel：切层，处理 pipeline bubbles

流水线并行沿“深度”切模型：rank 0 保存前几层，rank 1 保存后几层，依次类推。forward 时，前一阶段把 activation 发送给后一阶段；backward 时，梯度反向传回。

朴素 pipeline 的问题是 bubble：如果一次只送一个完整 batch，那么早期阶段在后面阶段计算时会空闲，后期阶段在等待输入时也会空闲。解决方法是把 batch 切成多个 microbatch，让不同 microbatch 在不同阶段同时流动。

bubble 的直观规律是：

- pipeline stage 越多，填充和排空开销越大；
- microbatch 数越多，bubble 占比越小；
- 但 microbatch 太多会增加调度开销，并影响 batch norm/optimizer 等行为。

真实系统还要设计 forward/backward 调度，例如 GPipe 的先全 forward 后全 backward，或 1F1B（一前一后）调度。为了减少等待，应使用异步 `isend/irecv`，让通信与后续 microbatch 的计算重叠。否则同步 send/recv 会让 GPU 经常阻塞。

## 9. 组合并行与工程调优

大模型训练通常不是只用一种并行，而是组合使用：

- 数据并行：跨更多节点扩展吞吐；
- FSDP/ZeRO：降低模型状态显存；
- Tensor Parallel：处理单层太宽、单卡放不下或单层计算太大的情况；
- Pipeline Parallel：处理模型太深、跨层切分；
- Activation checkpointing：降低 activation 显存；
- Sequence/context parallel：在长上下文训练中切序列维度。

一个常见经验是：把通信最频繁、最细粒度的 tensor parallel 放在同节点高速互联内；把较粗粒度的数据并行扩展到跨节点；必要时再加入 pipeline parallel。选择策略时要同时看显存、计算利用率、通信带宽、延迟和代码复杂度。

工程调优清单：

1. 先确认瓶颈：用 profiler 判断是 compute-bound、memory-bound 还是 communication-bound。
2. 调整 batch 与 microbatch：增大 microbatch 可提高矩阵乘利用率，但会增加 activation 显存。
3. 调整 FSDP bucket/prefetch：让 `all_gather` 和 `reduce_scatter` 尽量与计算重叠。
4. 避免过多小 collective：小消息延迟主导，合并 bucket 往往更快。
5. 匹配拓扑：同节点用 tensor parallel，跨节点少做高频通信。
6. 检查同步点：barrier、loss logging、checkpoint 保存都可能引入等待。
7. 保持确定性和一致性：所有 rank 必须按相同顺序进入 collective；随机种子、数据切分和 dropout 也要正确处理。
8. 定期保存完整 checkpoint：分片训练下 checkpoint 可能分布在多个 rank，需要明确保存和恢复格式。

## 10. 一个实用的选型流程

如果从零开始为一个模型选择并行方案，可以按下面顺序思考。第一步先估算单卡显存：参数、梯度、优化器状态和激活值分别占多少。如果完整模型加优化器状态已经放不下，优先考虑 FSDP 或 ZeRO；如果主要是长序列或大 microbatch 导致激活值过大，优先打开 activation checkpointing。第二步看单层是否过大：如果注意力头、MLP 中间层或词表投影单卡计算和显存都吃紧，就需要 tensor parallel，并尽量把同一个 tensor parallel group 放在同一台机器的高速互联内。第三步看模型深度：如果层数很多，而且只靠 FSDP 与张量并行仍不够，可以再用 pipeline parallel 按层切分。第四步才是扩展数据并行，把多个模型副本放到更多节点上提高总吞吐。

调参时不要只看每秒 token 数，还要同时看 GPU 利用率、通信时间占比、显存峰值和 step time 的方差。若 GPU 利用率低而通信时间高，说明并行切得太碎或跨节点通信太频繁；若显存接近上限但通信不高，可以增加 checkpointing 或更细的分片；若 pipeline 阶段负载不均，某些 rank 会长期等待，需要重新划分层或调整 microbatch 数。真正的分布式训练往往是反复测量、定位瓶颈、修改切分策略的过程。

## 11. 常见故障信号

分布式程序最常见的问题不是报错，而是卡住。遇到长时间无输出时，先检查所有 rank 是否进入了同样的 collective，张量形状是否一致，发送和接收的源、目标是否匹配。若训练能跑但速度忽快忽慢，要检查数据加载、日志写入、checkpoint 保存和跨节点网络拥塞。若某个 rank 显存明显更高，通常说明模型切分不均、pipeline stage 负载不均，或某些 activation 没有被释放。调试时可以先用较小模型和较少 rank 复现，再逐步扩大规模。任何一次改动后都应重新测量，避免凭直觉判断瓶颈和优化效果是否真正可靠。

## 12. 总结

本讲的核心是：分布式训练的本质是在数据、参数、梯度、优化器状态和 activation 之间做切分，并在计算、显存和通信之间做权衡。DDP 简单但显存冗余；ZeRO/FSDP 通过分片降低显存但增加通信；activation checkpointing 用重算换显存；tensor parallel 适合切大矩阵但需要高速 collective；pipeline parallel 能切深层模型但必须处理 bubbles 和调度问题。

硬件会继续进步，但模型规模也会继续逼近硬件极限。因此，层次化内存、通信瓶颈、重算与分片这些系统问题不会消失。训练大模型时，真正的性能来自算法、模型结构、并行策略和硬件拓扑的共同设计。