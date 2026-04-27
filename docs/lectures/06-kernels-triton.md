# Stanford CS336 2025 第 6 讲教程：Kernel、Triton 与 LLM 算子优化

本讲进入大模型训练的底层性能世界：如何理解 GPU 上的 kernel，如何用 CUDA/Triton 写自定义算子，以及为什么 FlashAttention 能带来巨大加速。核心思想很简单：GPU 擅长计算，但从显存搬数据很贵；高性能代码要让数据在离计算单元更近的地方多用几次，减少无意义的读写和 kernel 启动开销。

## 1. GPU 执行模型：SM、block 到 warp

一张 A100/H100 GPU 由许多 SM（Streaming Multiprocessor）组成。每个 SM 内有计算单元、寄存器、共享内存和缓存。程序提交到 GPU 的基本单位叫 kernel：一个 kernel 会启动很多线程，这些线程被组织成 thread block；多个 block 组成 grid。

可以用三层结构理解：

```text
grid = 很多 thread block
thread block = 会被调度到某个 SM 上执行的一组线程
thread = 真正执行指令、处理元素的最小单位
```

同一个 block 内的线程可以通过共享内存快速通信和同步；不同 block 之间通信很贵，也通常不能在一个 kernel 内同步。因此设计 kernel 时，要把需要共享的数据尽量放在同一个 block/SM 内完成。

GPU 还会把线程按 32 个一组组织成 warp。一个 warp 内的线程以 SIMD/SIMT 风格一起执行。这样可以减少控制逻辑，把更多芯片面积用于计算。代价是：如果 warp 内线程走不同分支，或工作量不均衡，就会降低效率。写 kernel 时通常希望 block 数足够多，能填满所有 SM；同时每个 warp 的工作尽量均匀。

## 2. 性能瓶颈：算还是搬？

判断一个算子快不快，不能只看 FLOPs，还要看 arithmetic intensity：每搬运 1 byte 数据能做多少计算。

```text
算术强度 = FLOPs / 内存搬运字节数
```

矩阵乘法如果做得好，数据块可以被重复使用，算术强度高，通常是 compute-bound。很多 elementwise 操作、softmax、归一化和简单激活函数则经常是 memory-bound：每个元素只做少量计算，却要从 HBM 读出再写回。

LLM 训练里有两类优化特别重要：

1. 让矩阵乘法走高性能库或硬件路径，例如 cuBLAS/CUTLASS/Tensor Core。
2. 对 memory-bound 算子做 fusion、tiling 和减少中间结果写回。

## 3. Benchmark 与 profiling

课程反复强调：不要凭感觉优化。benchmark 告诉你端到端运行多久；profiling 告诉你时间花在哪些 kernel 上。

做 GPU benchmark 有两个常见坑。

第一，要 warm up。第一次运行 PyTorch/CUDA 代码时，可能会触发 kernel 编译、库加载、缓存初始化等开销。真正关心的是稳定状态下的速度。

第二，要显式同步。CPU 提交 CUDA kernel 后通常不会等待 GPU 完成，而是继续向前执行。因此直接用 Python 的时间函数包住 GPU 操作，可能测到的只是“提交任务”的时间。正确做法是在计时前后调用：

```python
torch.cuda.synchronize()
```

Profiler 则能看到更底层的事件。例如 Python 里一个 `a + b`，底下会有 PyTorch C++ 接口、CUDA kernel launch 和真正的 elementwise kernel。矩阵乘法也不是固定实现：不同 shape、dtype、硬件会调度到不同的 cuBLAS/CUTLASS kernel。Nsight Systems 还能画出 CPU 线程和 GPU stream 的时间线：CPU 通常提前排队提交 kernel，GPU 在后面按队列执行。

这解释了为什么 Python 训练代码并不必然慢：只要 CPU 能足够快地排任务，瓶颈仍在 GPU。反过来，频繁 `print(loss)`、`.item()` 或把 tensor 搬回 CPU，会强制同步，使 CPU/GPU 流水线断开。

## 4. Kernel fusion：少读写一次就是巨大收益

假设要计算 GELU 的近似公式：

```text
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

如果用普通 PyTorch 表达式手写，可能会分解成多次乘法、加法、`tanh`、幂运算。每一步都是一个 kernel：从 HBM 读 `x`，计算，写中间结果；下一步再读中间结果，再写回。即使每个操作本身很简单，总体也会被内存读写和 kernel launch 拖慢。

Kernel fusion 的目标是把这些操作合成一个 kernel：每个元素从 HBM 读入一次，在寄存器里完成全部计算，最后写回一次。课程中的例子里，朴素手写 GELU 比 PyTorch 内置 fused GELU 慢很多；自定义 CUDA/Triton kernel 能把多个 kernel 压成一个，速度接近内置算子。

这个思路对 LLM 很重要。Transformer 里除了大矩阵乘法，还有 bias add、activation、dropout、residual add、layer norm、softmax、mask 等小操作。如果每个都单独读写 HBM，显存带宽会成为瓶颈。现代框架会自动做一部分 fusion，但复杂结构仍可能需要手写 kernel。

## 5. CUDA kernel 基本写法

CUDA 是直接编程 GPU 的 C++ 接口。写一个 elementwise GELU kernel 时，通常分两部分。

第一部分是 CPU 侧 wrapper：检查输入是否在 CUDA 上、是否 contiguous；分配输出 `torch.empty_like(x)`；计算 block size 和 block 数；最后启动 kernel。

第二部分是 GPU 侧 kernel：每个线程根据自己的位置计算全局索引。

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
    out[i] = gelu_formula(in[i]);
}
```

这里有几个工程细节值得记住。

1. `empty_like` 比先分配再清零更好，因为输出马上会被覆盖。
2. block 数要向上取整，保证尾部元素也被处理。
3. 最后一个 block 可能越界，所以必须检查 `i < n`。
4. 自定义 kernel 常假设输入 contiguous，否则索引逻辑会复杂很多。transpose/view 可能产生非连续 tensor，必要时要在外层调用 `.contiguous()`，但这会产生复制成本。
5. 调试 CUDA 时可设置 `CUDA_LAUNCH_BLOCKING=1`，让错误更容易定位，但会影响性能。

CUDA 的优点是控制力强；缺点是样板代码多，要手动管理线程、block、共享内存、同步和边界条件。

## 6. Triton：以 block 为中心的 GPU 编程

Triton 是 OpenAI 开发的 GPU 编程 DSL。它允许你在 Python 中写 kernel，但抽象层级比 CUDA 高：CUDA 通常让你思考“每个线程做什么”，Triton 更鼓励你思考“每个 program/block 处理一块数据”。

一个 Triton elementwise kernel 大致是这样：

```python
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n
x = tl.load(x_ptr + offsets, mask=mask)
y = gelu_formula(x)
tl.store(y_ptr + offsets, y, mask=mask)
```

`tl.arange` 生成一个向量化 offset，因此一个 Triton program 一次处理一整个 block。Triton 编译器会把它降到更底层的 GPU 指令，处理很多繁琐细节，例如 memory coalescing、寄存器使用、部分共享内存管理等。

Memory coalescing 是 GPU 性能关键。GPU 从 HBM 取数据时，更喜欢连续地址访问；相邻线程读取相邻元素，硬件就能合并成高效内存事务。Triton 的向量化 load/store 使这种模式更自然。查看生成的 PTX 可以看到，编译器会把连续值成组加载到寄存器中，再执行乘法、指数、tanh 等操作，最后成组写回。

Triton 的价值在于折中：比 PyTorch 表达式更接近硬件，比 CUDA 更容易写和调试。对于新模型中的特殊算子，Triton 往往是最实用的自定义 kernel 工具。

## 7. Tiling：让数据在近处多用几次

Tiling 是高性能 GPU 算子的核心模式：把大张量切成小块，让一个 block/SM 负责一个 tile。这样可以把 tile 读入寄存器或共享内存，在局部完成尽可能多的计算，再写回全局内存。

矩阵乘法的高性能实现高度依赖 tiling。不是每个输出元素都独立从 HBM 读取整行整列，而是把 A、B 的小块加载进共享内存，多个线程协作复用这些数据。这样同一份数据被多次参与乘加，算术强度提高。

Softmax 也是典型例子。如果一行长度能放进一个 block，就可以让每个 block 处理一整行：先读入一行，减去最大值保证数值稳定，计算 exp，做行内 sum reduction，再除以 sum，最后写回。这样中间结果不必反复落到 HBM。

```text
一行 softmax：
load row -> max -> exp(row - max) -> sum -> normalize -> store row
```

当序列很长或矩阵很大时，一个 tile 放不下全部数据，就需要分块 reduction、跨 tile 合并统计量，复杂度会上升。这正是 FlashAttention 的核心动机之一。

## 8. FlashAttention：为什么自定义 kernel 有必要

标准 attention 计算：

```text
scores = QK^T / sqrt(d)
probs = softmax(scores)
out = probs V
```

朴素实现会显式构造 `scores` 和 `probs`，形状是 `[batch, heads, seq, seq]`。当 seq 很长时，这个中间矩阵极大，读写 HBM 的成本非常高。注意力本身的数学没有变，但实现方式浪费了大量内存带宽。

FlashAttention 的关键是 IO-aware：把 Q、K、V 按 tile 分块，只在 SRAM/寄存器中维护当前块的 softmax 统计量和输出累积，不把完整 attention matrix 写到 HBM。它使用 online softmax，在扫描 K/V 时维护每行最大值、归一化分母和输出累积，从而得到与标准 attention 等价的结果。

这类优化很难靠简单 fusion 自动发现，因为它改变了计算调度和中间状态位置。FlashAttention 2/3 还进一步利用硬件特性，例如更好的并行划分、Tensor Core 和 H100 新能力。因此，当算子具有复杂 reduction、数据复用或特殊硬件路径时，手写 Triton/CUDA kernel 仍然有价值。

## 9. torch.compile：自动优化

课程也提醒：不要把所有东西都手写成 CUDA kernel。PyTorch 的 `torch.compile` 已经能自动做很多优化，包括简单的 kernel fusion、shape-specialized 优化、为矩阵乘法选择更合适的底层 kernel。示例里，手写 GELU 经过 `torch.compile` 后会生成 fused Triton kernel，性能接近甚至超过课堂手写 Triton 版本。

实践建议是：

1. 先写清晰正确的 PyTorch 版本。
2. 用 benchmark 和 profiler 找真正瓶颈。
3. 先尝试 `torch.compile`、官方 fused op、xFormers/FlashAttention 等成熟实现。
4. 如果仍然发现某个特殊算子占时高、内存读写多、自动编译器无法处理，再考虑 Triton/CUDA。

## 10. LLM 算子优化原则

大模型性能优化不是“把 Python 改成 C++”这么简单，而是围绕 GPU 内存层级重新组织计算：

- 尽量使用高性能矩阵乘法库，让 Tensor Core 工作。
- 对 memory-bound 操作做 fusion，减少中间 tensor 写回。
- 用 tiling 提高数据复用，把热点数据留在寄存器/共享内存。
- 保证连续访问和 memory coalescing，避免零散读写。
- 避免不必要的 CPU/GPU 同步，例如频繁 `.item()`、`print`、拷贝到 CPU。
- 用 profiler 验证每一次优化，而不是凭直觉猜。

本讲的核心 takeaway 是：LLM 的速度很大程度上取决于算子的实现方式。数学相同，性能可能差一个数量级。Triton 给研究者提供了实用入口：当 PyTorch 表达式太慢、CUDA 又太繁琐时，可以用接近 Python 的方式写出接近底层性能的自定义 kernel。

## 11. 什么时候值得写自定义 kernel

并不是所有慢代码都应该手写 kernel。一个实用判断是：如果某个操作在 profiler 中占比很高，而且它不是标准大矩阵乘法，那么才值得深入。标准矩阵乘法通常已经由 cuBLAS、CUTLASS、FlashAttention 等成熟库处理得很好；自己重写反而容易更慢。更适合自定义的场景包括：多个 elementwise 操作反复读写同一个大张量；一个算子需要特殊 mask 或特殊 reduction；中间张量很大但其实可以边算边丢；模型结构新颖，框架还没有 fused 实现。

写之前还要估算收益上限。如果一个 kernel 只占总训练时间 1%，即使优化 10 倍，端到端也只提升不到 1%。如果 attention 在长上下文下占 30%，并且 profiler 显示大量 HBM 读写，那么 FlashAttention 这种 IO-aware 重排就可能显著改变整体速度。优化应从端到端瓶颈开始，而不是从最有趣的底层代码开始。

## 12. 从 PyTorch 到 Triton 的推荐工作流

实践中可以采用四步法。第一，写最清晰的 PyTorch reference implementation，并用小 tensor 做 correctness test。第二，用 `torch.compile` 或已有 fused op 获取一个强 baseline，同时用 profiler 找热点。第三，如果仍然需要自定义 kernel，先写 Triton 版本，因为它更容易表达 block 级逻辑，也更容易和 Python 测试框架结合。第四，再根据必要性决定是否下探到 CUDA/CUTLASS，例如需要更细控制 shared memory、warp-level primitive、Tensor Core 指令或多阶段流水线。

每次改 kernel 都要同时验证三件事：数值是否一致，速度是否在多种 shape 下都更快，显存读写是否真的减少。很多 kernel 在一个 shape 上很快，换 batch size、sequence length 或 head dimension 就退化；也有些优化牺牲了数值稳定性，在长序列或低精度下才暴露问题。因此 LLM 算子优化不是单点技巧，而是 correctness、benchmark、hardware constraints 三者一起迭代。

## 13. 与后续课程的连接

这一讲是后面分布式训练和推理系统的底座。单卡 kernel 优化解决的是“一个 GPU 内部如何少搬数据、多做有效计算”；并行训练解决的是“多个 GPU 之间如何切分参数、激活和 batch，尽量少通信”；推理系统解决的是“在线请求如何 batching、调度 KV cache，并在 latency 和 throughput 间取舍”。三者使用同一套思维：先找到稀缺资源，再重排计算和数据流。稀缺资源可能是 HBM 带宽、SM 算力、显存容量、PCIe/NVLink 带宽，也可能是用户请求的延迟预算。理解 kernel 后，你会更容易看懂为什么 FlashAttention、FSDP、tensor parallel、PagedAttention 都是在围绕数据移动做文章。
