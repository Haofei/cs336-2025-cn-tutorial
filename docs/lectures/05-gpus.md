# Stanford CS336 Lecture 5：GPU 中文教程

大语言模型之所以能训练到今天的规模，核心原因之一是硬件吞吐量的持续增长，尤其是 GPU 的普及。本讲的重点不是把 CUDA API 背下来，而是建立一个性能直觉：GPU 很擅长并行矩阵乘法，但常常不是“算不动”，而是“数据搬不动”。理解 GPU 架构、执行模型和内存层次，才能解释为什么同一个矩阵乘法在某些维度上飞快、在另一些维度上突然变慢，也才能理解 FlashAttention 这类算法为什么有效。

## 1. CPU 与 GPU 的设计目标

CPU 优化的是低延迟。它通常有复杂的控制逻辑、分支预测、缓存体系和较高的单线程性能，目标是让一个任务尽快完成。GPU 优化的是高吞吐。它牺牲单个线程的灵活性和低延迟，把芯片面积更多用于大量算术单元，让成千上万个相似任务同时推进。

这正好适合深度学习：训练 Transformer 时，大量工作都是矩阵乘法、逐元素运算、归约和张量变换。这些操作结构规则、数据量巨大，可以拆成很多相似的小任务并行执行。随着 Dennard scaling 和单线程性能增长放缓，深度学习的扩展越来越依赖并行硬件，而 GPU 正是这种并行扩展的代表。

## 2. GPU 的基本架构：SM、SP 与 Tensor Core

可以把一块 GPU 看成由许多 Streaming Multiprocessor（SM）组成。SM 类似 GPU 上的基本执行单元：每个 SM 有自己的调度和控制逻辑、寄存器、共享内存以及大量执行单元。SM 内部有许多更细粒度的处理单元，可以对不同数据执行同一条指令。

现代 NVIDIA GPU 还包含 Tensor Core，这是一类专门为矩阵乘法设计的硬件单元。从 V100 开始，矩阵乘法吞吐量与普通浮点运算吞吐量之间出现巨大差距：如果你的神经网络大部分时间都能落在矩阵乘法上，就能吃到 Tensor Core 的红利；如果设计了大量非矩阵乘法的复杂操作，即使理论 FLOPs 不高，也可能跑得很慢。

这也是 LLM 架构偏爱线性层、注意力中的 QK^T 和 PV、MLP 中大矩阵乘法的原因：这些操作与 GPU 的硬件能力高度匹配。

## 3. SIMT、thread、warp 与 block

GPU 的执行模型通常称为 SIMT：Single Instruction, Multiple Threads。也就是说，同一组线程在同一时刻执行同一条指令，但处理不同的数据。

CUDA 编程中常见三个层次：

- thread：最小的逻辑执行单位。
- warp：一组通常为 32 个连续线程，它们一起执行同一条指令。
- block：一组线程，通常被分配到某个 SM 上执行。

这个模型带来一个重要限制：同一个 warp 内最好不要出现严重分支分歧。例如 32 个线程中一半走 if 分支，另一半走 else 分支，GPU 不能真正同时执行两条不同路径，而是先让一部分线程执行、另一部分暂停，再反过来执行。这样会降低有效利用率。因此，高性能 GPU kernel 通常追求规则的数据访问和规则的控制流。

## 4. 内存层次：性能优化的主战场

GPU 的计算单元非常快，但内存速度差异巨大。可以按从快到慢理解：

- register：每个线程私有，最快，适合保存临时标量。
- shared memory / L1：位于 SM 内部，延迟很低，线程块内可以共享。
- L2 cache：在芯片上，但不在单个 SM 内，速度较慢。
- global memory / HBM：位于芯片外的高带宽显存，容量大但延迟高。

访问 shared memory 可能只需几十个 cycle，而访问全局显存可能需要数百个 cycle。如果 kernel 不断从 HBM 读写中间结果，计算单元就会等待数据，吞吐量很难上去。

一个优秀 GPU 算法的基本原则是：尽量少访问 global memory；一旦把数据搬进 SM，就在 shared memory 或寄存器中做尽可能多的计算；最后只把必要结果写回 HBM。

## 5. Roofline 模型：算力瓶颈还是内存瓶颈

Roofline 模型用来判断程序性能受什么限制。横轴常理解为 arithmetic intensity，即每读取一个字节数据能做多少 FLOPs；纵轴是实际吞吐。

当 arithmetic intensity 很低时，程序在左侧的 memory-bound 区域：算术单元还没吃饱，性能主要由内存带宽决定。很多逐元素操作就是这样，例如 ReLU、加法、LayerNorm 的某些部分，它们读写很多数据，但每个元素只做少量计算。

当 arithmetic intensity 足够高时，程序进入 compute-bound 区域：矩阵乘法足够大，数据复用充分，Tensor Core 被充分利用，吞吐接近硬件峰值。

LLM 训练中，大矩阵乘法通常更容易 compute-bound，而小 batch、小矩阵、逐元素操作、归约、频繁写回中间张量则容易 memory-bound。优化的核心就是把更多工作推向 roofline 的右上方。

## 6. 矩阵乘法为什么需要 tiling

朴素矩阵乘法 C = A × B 中，每个 C[i,j] 需要读取 A 的一行和 B 的一列。如果每个线程都直接从 global memory 读取所需元素，会产生大量重复读取：同一个 A 元素会被多个输出复用，同一个 B 元素也会被多个输出复用。若每次都从 HBM 取，显然浪费。

Tiling 的思路是把 A、B、C 切成小块。一个 block 负责计算 C 的一个 tile。它先把 A 和 B 的对应 tile 从 global memory 搬到 shared memory，然后在 shared memory 中反复使用这些数据，累加 partial sum。处理完当前 tile 后，再加载下一个 tile。

这样做有两个收益：

1. 全局内存读取次数减少。tile 大小为 T 时，理想情况下可以把部分 global memory 访问减少约 T 倍。
2. 访问模式更规则。加载 tile 时可以安排连续线程读取连续地址，便于 memory coalescing。

不过 tile 大小不是越大越好。它受 shared memory 容量、寄存器数量、warp 调度、Tensor Core 形状和矩阵维度整除性的共同限制。矩阵维度如果刚好是 tile 大小、warp 大小或 burst section 的倍数，通常会更快；如果多出 1 个元素，可能需要额外 tile，导致很多 SM 处理“稀疏边角块”，吞吐突然下降。

## 7. Memory coalescing、padding 与奇怪的性能波动

DRAM 通常不是一次只返回一个标量，而是按连续块读取。若同一 warp 的线程访问相邻地址，硬件可以把这些访问合并成更少的内存事务，这叫 memory coalescing。若线程访问分散地址，就会触发多次读取，带宽利用率下降。

这解释了很多看似玄学的现象：矩阵按行访问还是按列访问，性能可能差很多；词表大小、hidden size、batch size 是否是 8、16、32、64、128 的倍数，也会影响吞吐。Karpathy 曾提到把 nanoGPT 的 vocab size padding 到 64 的倍数能带来明显加速，本质上就是让矩阵形状更适合 GPU 的 tile、warp 和内存对齐。

另一个现象是 wave quantization。假设 A100 有 108 个 SM，如果一次矩阵乘法被切成 98 个 tile，那么一波即可让大部分 SM 工作；如果维度略增导致 tile 数变成 120，前 108 个 tile 先跑，剩下 12 个 tile 再单独跑一小波，后半段 SM 利用率很低。于是矩阵只增大一点，性能却可能掉一大截。

## 8. 降低精度、融合与重计算

为了减少内存压力，常用三类技巧。

第一是低精度。FP16、BF16、FP8 或 int8 会减少每个元素的字节数，使同样的带宽能搬运更多数据，也能使用更快的 Tensor Core。训练中通常采用 mixed precision：输入和权重用 16 位，乘法累加用 FP32 accumulator，以兼顾速度和数值稳定性。

第二是 operator fusion。若代码先计算 sin(x)，写回 HBM，再读出计算平方，再写回，内存往返会非常多。融合 kernel 会把多个逐元素操作放在一次 kernel 中完成，中间值保存在寄存器或 shared memory，只在最后写回结果。torch.compile、Triton 和手写 CUDA kernel 都常用于这种优化。

第三是 recomputation。反向传播需要前向激活。朴素做法是把所有激活存到 HBM，反向时再读回来。但如果某些激活计算便宜、读取昂贵，可以在反向时重新计算它们，用额外 FLOPs 换更少的内存读写。这不仅能省显存，也能在 memory-bound 场景下提速。

## 9. FlashAttention：把这些思想组合起来

标准 attention 包含 QK^T、softmax、再乘 V。问题是注意力矩阵大小为 n × n，如果把完整 score 和 softmax 结果都 materialize 到 HBM，长上下文时内存读写会非常昂贵。

FlashAttention 的关键不是减少 attention 的数学计算量，而是减少 HBM 访问。它使用 tiling：把 Q、K、V 分块搬到 SRAM/shared memory，在块内计算 QK^T 和后续累加。难点是 softmax 是行级全局操作，必须知道整行的最大值和归一化分母。FlashAttention 使用 online softmax：按块维护当前行的 running max 和归一化和，每来一个 tile 就更新这些统计量，因此不需要把完整 n × n 矩阵写回显存。

反向传播中，FlashAttention 还会对 softmax 相关量做重计算，避免保存 n × n 的中间激活。于是它结合了本讲的多个核心技巧：tiling、shared memory 复用、operator fusion、online softmax 和 recomputation。结果是精确 attention，但 HBM 访问显著下降，长序列 Transformer 训练和推理都更快。

## 10. 总结：LLM 训练为何依赖 GPU

LLM 训练依赖 GPU，不只是因为 GPU FLOPs 高，更因为 Transformer 的主要计算形式天然适合 GPU：大规模矩阵乘法、规则张量操作、可批处理的并行数据流。Tensor Core 让矩阵乘法成为“被硬件祝福”的操作，混合精度进一步放大吞吐。

但现代 GPU 的真正瓶颈越来越多地来自内存移动，而不是纯计算。高性能实现要关注：是否让 warp 访问连续内存；是否避免不必要的 HBM 读写；是否用 tiling 提高数据复用；矩阵维度是否对齐；是否能 fusion；是否能用重计算换内存；是否让 tile 数和 SM 数匹配良好。

因此，理解 GPU 的核心不是记住某个型号的参数，而是形成一套判断性能瓶颈的思维：计算是否足够密集？数据是否被重复从 HBM 搬运？warp 是否分歧？访问是否合并？tile 是否对齐？这些细节共同决定了 LLM 训练能否真正把昂贵的 GPU 用满。

## 11. 实践检查清单：看到慢代码时先问什么

当你在训练或推理 LLM 时发现 GPU 利用率不高，可以按下面顺序排查。第一，看是不是 CPU 在拖后腿：数据加载、tokenization、日志打印、频繁 `.item()` 都可能让 GPU 等待。第二，看是不是 kernel 太碎：如果一个 Transformer block 中出现大量很短的小 kernel，说明许多逐元素操作没有融合，kernel launch 和 HBM 往返会吞掉时间。第三，看矩阵形状是否适合 Tensor Core：hidden size、vocab size、batch×sequence 是否对齐到硬件喜欢的倍数。第四，看显存占用是否迫使 batch 太小：batch 太小会降低矩阵乘法算术强度，也会让并行度不够。第五，用 profiler 确认瓶颈，而不是只看 `nvidia-smi` 的利用率；后者只能给粗略信号，不能告诉你到底是哪个 kernel 慢。

## 12. 一个贯穿例子：为什么“同样的模型”速度可以差很多

假设两个实现都训练同一个 Transformer，参数量、batch size 和 dtype 完全相同。实现 A 直接用许多 PyTorch 小操作拼接 attention、MLP、residual 和 normalization；实现 B 使用 fused layer norm、FlashAttention、fused optimizer，并把词表和 hidden size padding 到更友好的维度。两者数学上几乎等价，但实现 B 会少写回大量中间张量，减少 CPU/GPU 同步，让矩阵乘法更稳定地落到高效 tile 上。结果可能不是提升 5%，而是提升数十个百分点甚至更多。

这就是本讲想建立的工程直觉：模型架构论文里的一行公式，落到 GPU 上会变成很多 kernel、很多内存事务和很多调度决策。做 LLM Infra 不能只懂模型，也要能把公式翻译成硬件成本：哪些张量会被 materialize，哪些中间值可以重算，哪些操作应该融合，哪些维度会触发额外一波 tile。掌握这套直觉后，后面的 Triton、自定义 kernel、分布式训练和推理服务才会真正连起来。

最后给一个最实用的判断标准：如果优化能减少 HBM 读写、提高 Tensor Core 利用率、降低同步次数或提升 batch/sequence 的有效并行度，它通常值得尝试；如果只是把代码写得更底层但没有改变数据移动路径，收益往往有限。
