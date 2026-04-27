# Stanford CS336 2025 Lecture 6 Tutorial: Kernels, Triton, and LLM Operator Optimization

> Adaptation note: This tutorial is translated and adapted from the Chinese tutorial version. It preserves the original structure, code snippets, formulas, and technical terminology while making the explanations natural for English-speaking learners.

This lecture enters the low-level performance world of large-model training: how to understand kernels on GPUs, how to write custom operators with CUDA/Triton, and why FlashAttention can bring huge speedups. The core idea is simple: GPUs are good at computation, but moving data from GPU memory is expensive; high-performance code should reuse data near the compute units as many times as possible, while reducing meaningless reads, writes, and kernel launch overhead.

## 1. GPU execution model: from SM and block to warp

An A100/H100 GPU consists of many SMs (Streaming Multiprocessors). Each SM has compute units, registers, shared memory, and cache. The basic unit submitted to the GPU is called a kernel: a kernel launches many threads, organized into thread blocks; multiple blocks form a grid.

You can understand the structure in three levels:

```text
grid = many thread blocks
thread block = a group of threads scheduled onto an SM
thread = the smallest unit that actually executes instructions and processes elements
```

Threads inside the same block can communicate and synchronize quickly through shared memory. Communication between different blocks is expensive, and usually different blocks cannot synchronize inside one kernel. Therefore, when designing a kernel, you should try to place data that must be shared within the same block/SM.

The GPU also groups threads into warps of 32. Threads inside one warp execute together in a SIMD/SIMT style. This reduces control logic and leaves more chip area for computation. The cost is that if threads inside a warp take different branches or have unbalanced work, efficiency drops. When writing kernels, we usually want enough blocks to fill all SMs, and we want each warp to have uniform work.

## 2. Performance bottleneck: compute or move data?

To judge whether an operator is fast, do not look only at FLOPs. Also look at arithmetic intensity: how much computation is done per byte of data moved.

```text
arithmetic intensity = FLOPs / bytes moved
```

If implemented well, matrix multiplication reuses data blocks many times, has high arithmetic intensity, and is usually compute-bound. Many elementwise operations, softmax, normalization, and simple activation functions are often memory-bound: each element involves little computation but must be read from and written back to HBM.

Two types of optimization are especially important in LLM training:

1. Make matrix multiplication use high-performance libraries or hardware paths such as cuBLAS/CUTLASS/Tensor Core.
2. Apply fusion, tiling, and reduced intermediate writes to memory-bound operators.

## 3. Benchmarking and profiling

The course repeatedly emphasizes: do not optimize by feeling. A benchmark tells you how long end-to-end execution takes; profiling tells you which kernels consume the time.

GPU benchmarking has two common pitfalls.

First, warm up. The first run of PyTorch/CUDA code may trigger kernel compilation, library loading, cache initialization, and other overheads. What we care about is steady-state speed.

Second, synchronize explicitly. After the CPU submits a CUDA kernel, it usually does not wait for the GPU to finish, but continues executing. Therefore, directly wrapping a GPU operation with Python timing functions may measure only “task submission” time. The correct approach is to call before and after timing:

```python
torch.cuda.synchronize()
```

A profiler shows lower-level events. For example, a Python expression `a + b` involves the PyTorch C++ interface, CUDA kernel launch, and the actual elementwise kernel underneath. Matrix multiplication is not a fixed implementation either: different shapes, dtypes, and hardware dispatch to different cuBLAS/CUTLASS kernels. Nsight Systems can also draw a timeline of CPU threads and GPU streams: the CPU usually queues kernels ahead of time, and the GPU executes them later in order.

This explains why Python training code is not necessarily slow: as long as the CPU can submit work fast enough, the bottleneck remains on the GPU. Conversely, frequent `print(loss)`, `.item()`, or moving tensors back to CPU forces synchronization and breaks the CPU/GPU pipeline.

## 4. Kernel fusion: one fewer read/write is a huge gain

Suppose we need to compute the GELU approximation:

```text
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

If written as an ordinary PyTorch expression, it may be decomposed into multiple multiplications, additions, `tanh`, and power operations. Each step is a kernel: read `x` from HBM, compute, write an intermediate result; then the next step reads the intermediate result and writes again. Even if each operation itself is simple, the whole computation is slowed by memory reads/writes and kernel launches.

The goal of kernel fusion is to merge these operations into one kernel: each element is read once from HBM, all computation is performed in registers, and the result is written back once. In the course example, naive handwritten GELU is much slower than PyTorch’s built-in fused GELU; a custom CUDA/Triton kernel can compress multiple kernels into one and achieve speed close to the built-in operator.

This idea is important for LLMs. Besides large matrix multiplications, Transformers include many small operations: bias add, activation, dropout, residual add, layer norm, softmax, mask, and so on. If each one separately reads and writes HBM, memory bandwidth becomes the bottleneck. Modern frameworks automatically perform some fusion, but complex structures may still require handwritten kernels.

## 5. Basic CUDA kernel pattern

CUDA is a C++ interface for programming GPUs directly. When writing an elementwise GELU kernel, there are usually two parts.

The first part is the CPU-side wrapper: check that the input is on CUDA and contiguous; allocate output with `torch.empty_like(x)`; compute the block size and number of blocks; finally launch the kernel.

The second part is the GPU-side kernel: each thread computes its global index based on its position.

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
    out[i] = gelu_formula(in[i]);
}
```

Several engineering details are worth remembering.

1. `empty_like` is better than allocating and zeroing, because the output will immediately be overwritten.
2. The number of blocks should be rounded up so that tail elements are processed.
3. The last block may go out of bounds, so the check `i < n` is required.
4. Custom kernels often assume contiguous input; otherwise indexing logic becomes much more complex. `transpose`/`view` may produce non-contiguous tensors, so `.contiguous()` may be needed at the outer layer, but it creates a copy cost.
5. When debugging CUDA, `CUDA_LAUNCH_BLOCKING=1` can make errors easier to locate, but it affects performance.

CUDA gives strong control, but it also requires a lot of boilerplate: you manually manage threads, blocks, shared memory, synchronization, and boundary conditions.

## 6. Triton: block-centered GPU programming

Triton is a GPU programming DSL developed by OpenAI. It lets you write kernels in Python, but the abstraction level is higher than CUDA: CUDA usually makes you think “what does each thread do,” while Triton encourages you to think “what block of data does each program/block handle.”

A Triton elementwise kernel roughly looks like this:

```python
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n
x = tl.load(x_ptr + offsets, mask=mask)
y = gelu_formula(x)
tl.store(y_ptr + offsets, y, mask=mask)
```

`tl.arange` generates a vector of offsets, so one Triton program processes a whole block at once. The Triton compiler lowers this to lower-level GPU instructions and handles many tedious details, such as memory coalescing, register use, and some shared-memory management.

Memory coalescing is crucial for GPU performance. When a GPU fetches data from HBM, it prefers contiguous address access; if neighboring threads read neighboring elements, the hardware can combine them into efficient memory transactions. Triton’s vectorized load/store makes this access pattern natural. Looking at the generated PTX, you can see that the compiler loads consecutive values into registers in groups, then performs multiply, exp, tanh, and other operations, and finally writes values back in groups.

Triton’s value is the trade-off: it is closer to hardware than PyTorch expressions, but easier to write and debug than CUDA. For special operators in new models, Triton is often the most practical custom kernel tool.

## 7. Tiling: reuse nearby data several times

Tiling is a core pattern in high-performance GPU operators: split a large tensor into small tiles, and let one block/SM handle one tile. The tile can be loaded into registers or shared memory, as much local computation as possible can be performed, and the result is then written back to global memory.

High-performance matrix multiplication relies heavily on tiling. Instead of having every output element independently read an entire row and column from HBM, small blocks of A and B are loaded into shared memory, and multiple threads cooperate to reuse them. The same data participates in multiple multiply-adds, increasing arithmetic intensity.

Softmax is another typical example. If one row fits into one block, each block can process a whole row: first read the row, subtract the maximum for numerical stability, compute exp, perform a row-wise sum reduction, divide by the sum, and finally write back. Intermediate results do not need to be repeatedly stored in HBM.

```text
one row of softmax:
load row -> max -> exp(row - max) -> sum -> normalize -> store row
```

When the sequence is very long or the matrix is very large, one tile cannot hold all data. Then block-wise reductions and cross-tile merging of statistics are needed, increasing complexity. This is one of the core motivations for FlashAttention.

## 8. FlashAttention: why custom kernels are necessary

Standard attention computes:

```text
scores = QK^T / sqrt(d)
probs = softmax(scores)
out = probs V
```

A naive implementation explicitly constructs `scores` and `probs`, with shape `[batch, heads, seq, seq]`. When seq is long, this intermediate matrix is enormous and the cost of reading and writing HBM is very high. The mathematics of attention has not changed, but the implementation wastes a large amount of memory bandwidth.

The key of FlashAttention is IO-awareness: split Q, K, and V into tiles, maintain current-block softmax statistics and output accumulation only in SRAM/registers, and never write the full attention matrix to HBM. It uses online softmax: while scanning K/V, it maintains each row’s maximum, normalization denominator, and output accumulator, producing results equivalent to standard attention.

This kind of optimization is hard for simple fusion to discover automatically, because it changes the computation schedule and where intermediate state lives. FlashAttention 2/3 further exploit hardware features such as better parallel partitioning, Tensor Cores, and new H100 capabilities. Therefore, when an operator has complex reductions, data reuse, or special hardware paths, handwritten Triton/CUDA kernels are still valuable.

## 9. torch.compile: automatic optimization

The course also reminds us: do not write everything by hand as a CUDA kernel. PyTorch’s `torch.compile` can already perform many optimizations automatically, including simple kernel fusion, shape-specialized optimization, and choosing better low-level kernels for matrix multiplication. In the examples, a handwritten GELU compiled with `torch.compile` generates a fused Triton kernel and performs close to or even better than the hand-written Triton version from class.

Practical advice:

1. First write a clear and correct PyTorch version.
2. Use benchmarks and profilers to find the real bottleneck.
3. First try `torch.compile`, official fused ops, xFormers/FlashAttention, and other mature implementations.
4. If a special operator still takes a lot of time, has many memory reads/writes, and cannot be handled by the automatic compiler, then consider Triton/CUDA.

## 10. Principles for LLM operator optimization

Large-model performance optimization is not simply “rewrite Python as C++.” It is about reorganizing computation around the GPU memory hierarchy:

- Use high-performance matrix multiplication libraries whenever possible so Tensor Cores work.
- Fuse memory-bound operations and reduce intermediate tensor writes.
- Use tiling to improve data reuse and keep hot data in registers/shared memory.
- Ensure contiguous access and memory coalescing; avoid scattered reads and writes.
- Avoid unnecessary CPU/GPU synchronization, such as frequent `.item()`, `print`, or copying to CPU.
- Use a profiler to validate every optimization instead of guessing by intuition.

The core takeaway is: LLM speed depends heavily on how operators are implemented. The mathematics may be the same, but performance can differ by an order of magnitude. Triton provides a practical entry point for researchers: when PyTorch expressions are too slow and CUDA is too cumbersome, Triton can be used to write custom kernels close to low-level performance in a Python-like way.

## 11. When is it worth writing a custom kernel?

Not all slow code should become a handwritten kernel. A practical criterion is: if an operation accounts for a large fraction of time in the profiler and it is not a standard large matrix multiplication, then it is worth investigating. Standard matrix multiplication is usually already handled well by mature libraries such as cuBLAS, CUTLASS, and FlashAttention; rewriting it yourself is often slower. Better candidates for custom kernels include: multiple elementwise operations repeatedly reading and writing the same large tensor; an operator requiring a special mask or special reduction; a large intermediate tensor that can actually be discarded as computation proceeds; or a novel model structure for which the framework has no fused implementation.

Before writing a kernel, estimate the upper bound of the gain. If one kernel takes only 1% of total training time, even a 10× speedup improves end-to-end time by less than 1%. If attention takes 30% under long contexts and the profiler shows heavy HBM reads and writes, then an IO-aware rearrangement such as FlashAttention may significantly change overall speed. Optimization should start from end-to-end bottlenecks, not from the most interesting low-level code.

## 12. Recommended workflow from PyTorch to Triton

In practice, a four-step workflow is useful. First, write the clearest PyTorch reference implementation and run correctness tests on small tensors. Second, use `torch.compile` or existing fused ops to obtain a strong baseline, while using a profiler to find hotspots. Third, if a custom kernel is still needed, write the Triton version first, because it expresses block-level logic more easily and integrates better with Python testing frameworks. Fourth, decide whether to go down to CUDA/CUTLASS only if necessary, such as when you need finer control over shared memory, warp-level primitives, Tensor Core instructions, or multi-stage pipelines.

Every kernel change must verify three things at the same time: numerical consistency, speed improvement across multiple shapes, and actual reduction in memory reads/writes. Many kernels are fast for one shape but degrade when batch size, sequence length, or head dimension changes; some optimizations sacrifice numerical stability and only fail under long sequences or low precision. Therefore, LLM operator optimization is not a single trick, but an iteration over correctness, benchmarking, and hardware constraints.

## 13. Connection to later lectures

This lecture is the foundation for later distributed training and inference systems. Single-GPU kernel optimization answers “how can one GPU move less data internally and perform more useful computation”; parallel training answers “how can multiple GPUs split parameters, activations, and batches while minimizing communication”; inference systems answer “how can online requests be batched, how should KV cache be scheduled, and how should latency and throughput be traded off.” All three use the same mindset: first identify the scarce resource, then reorganize computation and data flow. The scarce resource may be HBM bandwidth, SM compute, memory capacity, PCIe/NVLink bandwidth, or the user’s latency budget. After understanding kernels, it becomes easier to see why FlashAttention, FSDP, tensor parallelism, and PagedAttention are all about data movement.
