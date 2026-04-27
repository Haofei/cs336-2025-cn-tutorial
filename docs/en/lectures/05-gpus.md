# Stanford CS336 Lecture 5 Tutorial: GPUs

> Adaptation note: This tutorial is translated and adapted from the Chinese tutorial version. It preserves the original structure and technical terms while making the explanations natural for English-speaking learners.

One core reason large language models can be trained at today’s scale is the continuous growth of hardware throughput, especially the widespread use of GPUs. The point of this lecture is not to memorize the CUDA API, but to build performance intuition: GPUs are very good at parallel matrix multiplication, yet the bottleneck is often not “unable to compute,” but “unable to move data fast enough.” Understanding GPU architecture, the execution model, and the memory hierarchy helps explain why the same matrix multiplication can be extremely fast for some dimensions and suddenly slow for others, and why algorithms such as FlashAttention are effective.

## 1. Design goals of CPUs and GPUs

CPUs optimize for low latency. They usually have complex control logic, branch prediction, cache hierarchies, and high single-thread performance. The goal is to finish one task as quickly as possible. GPUs optimize for high throughput. They sacrifice the flexibility and low latency of a single thread and devote more chip area to many arithmetic units, allowing thousands of similar tasks to make progress at the same time.

This matches deep learning well. When training Transformers, most work consists of matrix multiplication, elementwise operations, reductions, and tensor transformations. These operations have regular structure and huge data volume, so they can be split into many similar small tasks and executed in parallel. As Dennard scaling and single-thread performance growth slowed, deep learning scaling increasingly relied on parallel hardware, and GPUs became the representative example of such parallel scaling.

## 2. Basic GPU architecture: SM, SP, and Tensor Core

You can think of a GPU as being made of many Streaming Multiprocessors (SMs). An SM is like a basic execution unit on the GPU: each SM has its own scheduling and control logic, registers, shared memory, and many execution units. Inside an SM there are many finer-grained processing units that can execute the same instruction on different data.

Modern NVIDIA GPUs also include Tensor Cores, hardware units specialized for matrix multiplication. Starting with V100, the gap between matrix multiplication throughput and ordinary floating-point throughput became huge: if most of your neural network time can be mapped to matrix multiplication, you can benefit from Tensor Cores; if you design many complex operations that are not matrix multiplications, the model may run slowly even if its theoretical FLOPs are not high.

This is why LLM architectures favor linear layers, QK^T and PV in attention, and large matrix multiplications in MLPs: these operations match GPU hardware very well.

## 3. SIMT, thread, warp, and block

The GPU execution model is usually called SIMT: Single Instruction, Multiple Threads. A group of threads executes the same instruction at the same time, but processes different data.

CUDA programming commonly uses three levels:

- thread: the smallest logical execution unit.
- warp: usually a group of 32 consecutive threads that execute the same instruction together.
- block: a group of threads, usually assigned to an SM for execution.

This model creates an important constraint: severe branch divergence inside the same warp should be avoided. For example, if half of the 32 threads take the if branch and the other half take the else branch, the GPU cannot truly execute both paths at the same time. It runs one subset while pausing the other, then switches. This reduces effective utilization. Therefore, high-performance GPU kernels usually prefer regular data access and regular control flow.

## 4. Memory hierarchy: the main battlefield of performance optimization

GPU compute units are very fast, but memory speeds differ greatly. From fastest to slowest:

- register: private to each thread, fastest, suitable for temporary scalars.
- shared memory / L1: inside an SM, very low latency, shared within a thread block.
- L2 cache: on-chip but not inside a single SM, slower.
- global memory / HBM: high-bandwidth memory outside the chip, large capacity but high latency.

Accessing shared memory may take only tens of cycles, while accessing global memory may take hundreds of cycles. If a kernel constantly reads and writes intermediate results from HBM, compute units wait for data and throughput is hard to improve.

A basic principle of good GPU algorithms is: access global memory as little as possible; once data is moved into an SM, do as much computation as possible in shared memory or registers; finally write only the necessary results back to HBM.

## 5. Roofline model: compute bottleneck or memory bottleneck

The Roofline model helps determine what limits program performance. The horizontal axis is often understood as arithmetic intensity: how many FLOPs are performed per byte read. The vertical axis is actual throughput.

When arithmetic intensity is low, the program is in the memory-bound region on the left: arithmetic units are not fully fed, and performance is mainly determined by memory bandwidth. Many elementwise operations are like this, such as ReLU, addition, and parts of LayerNorm: they read and write lots of data, but do little computation per element.

When arithmetic intensity is high enough, the program enters the compute-bound region: matrix multiplications are large enough, data reuse is sufficient, Tensor Cores are fully used, and throughput approaches the hardware peak.

In LLM training, large matrix multiplications are usually easier to make compute-bound, while small batches, small matrices, elementwise operations, reductions, and frequent writes of intermediate tensors tend to be memory-bound. The core of optimization is to push more work toward the upper-right of the roofline.

## 6. Why matrix multiplication needs tiling

In naive matrix multiplication C = A × B, each C[i,j] reads one row of A and one column of B. If every thread reads its needed elements directly from global memory, there will be many repeated reads: the same A element is reused by multiple outputs, and the same B element is also reused by multiple outputs. Fetching from HBM every time is obviously wasteful.

Tiling splits A, B, and C into small blocks. One block computes one tile of C. It first moves the corresponding tiles of A and B from global memory to shared memory, then repeatedly reuses these data in shared memory while accumulating partial sums. After finishing the current tile, it loads the next tile.

This has two benefits:

1. Global memory reads are reduced. When tile size is T, the ideal case can reduce some global memory accesses by about T times.
2. Access patterns become more regular. When loading a tile, consecutive threads can read consecutive addresses, enabling memory coalescing.

However, larger tiles are not always better. Tile size is jointly limited by shared memory capacity, register count, warp scheduling, Tensor Core shapes, and divisibility of matrix dimensions. If matrix dimensions are exact multiples of the tile size, warp size, or burst section, performance is usually better. If there is one extra element, an additional tile may be needed, making many SMs process sparse edge tiles and causing throughput to drop suddenly.

## 7. Memory coalescing, padding, and strange performance fluctuations

DRAM usually does not return one scalar at a time; it reads contiguous blocks. If threads in the same warp access neighboring addresses, the hardware can merge these accesses into fewer memory transactions. This is called memory coalescing. If threads access scattered addresses, multiple reads are triggered and bandwidth utilization drops.

This explains many phenomena that look mysterious: accessing a matrix by rows versus columns can have very different performance; whether vocab size, hidden size, or batch size is a multiple of 8, 16, 32, 64, or 128 can affect throughput. Karpathy once noted that padding nanoGPT’s vocab size to a multiple of 64 brought a clear speedup. The reason is that the matrix shapes become friendlier to GPU tiles, warps, and memory alignment.

Another phenomenon is wave quantization. Suppose an A100 has 108 SMs. If a matrix multiplication is split into 98 tiles, one wave can keep most SMs working. If dimensions increase slightly and the number of tiles becomes 120, the first 108 tiles run first, and the remaining 12 tiles run in a small second wave, where SM utilization is low. So a matrix can grow only slightly while performance drops sharply.

## 8. Lower precision, fusion, and recomputation

Three classes of techniques are commonly used to reduce memory pressure.

First is lower precision. FP16, BF16, FP8, or int8 reduce the number of bytes per element, so the same bandwidth can move more data and faster Tensor Cores can be used. Training usually uses mixed precision: inputs and weights use 16-bit values, while multiplication accumulates into an FP32 accumulator, balancing speed and numerical stability.

Second is operator fusion. If code computes sin(x), writes it back to HBM, reads it again to square it, then writes it back again, there are many memory round trips. A fused kernel completes multiple elementwise operations in one kernel, keeping intermediate values in registers or shared memory and writing only the final result back. torch.compile, Triton, and handwritten CUDA kernels are all commonly used for this optimization.

Third is recomputation. Backpropagation needs forward activations. The naive approach stores all activations in HBM and reads them during the backward pass. But if some activations are cheap to compute and expensive to read, they can be recomputed during backward, trading extra FLOPs for fewer memory reads and writes. This saves memory and can also speed up memory-bound workloads.

## 9. FlashAttention: combining these ideas

Standard attention contains QK^T, softmax, and multiplication by V. The problem is that the attention matrix has size n × n. If the full score matrix and softmax result are materialized in HBM, memory traffic becomes very expensive for long contexts.

The key point of FlashAttention is not reducing the mathematical computation of attention, but reducing HBM access. It uses tiling: Q, K, and V are moved into SRAM/shared memory in blocks, and QK^T plus subsequent accumulation are computed inside blocks. The difficulty is that softmax is a row-wise global operation, requiring the maximum value and normalization denominator of the whole row. FlashAttention uses online softmax: it maintains the running max and normalization sum for each row block by block, updating these statistics for each tile, so the full n × n matrix does not need to be written back to memory.

In the backward pass, FlashAttention also recomputes softmax-related quantities, avoiding storage of n × n intermediate activations. It therefore combines several core techniques from this lecture: tiling, shared-memory reuse, operator fusion, online softmax, and recomputation. The result is exact attention with much less HBM access, making long-sequence Transformer training and inference faster.

## 10. Summary: why LLM training depends on GPUs

LLM training depends on GPUs not only because GPU FLOPs are high, but also because the main computations in Transformers naturally fit GPUs: large matrix multiplications, regular tensor operations, and batchable parallel data flow. Tensor Cores make matrix multiplication an operation “blessed by hardware,” and mixed precision further amplifies throughput.

However, the real bottleneck of modern GPUs increasingly comes from memory movement rather than pure computation. High-performance implementations must ask: do warps access contiguous memory? Are unnecessary HBM reads and writes avoided? Is tiling used to improve data reuse? Are matrix dimensions aligned? Can operations be fused? Can recomputation trade compute for memory? Do tile counts match SM counts well?

Thus, understanding GPUs is not about memorizing the specifications of one model. It is about forming a way to judge performance bottlenecks: is computation dense enough? Is data repeatedly moved from HBM? Are warps diverging? Are accesses coalesced? Are tiles aligned? These details jointly determine whether LLM training can truly make full use of expensive GPUs.

## 11. Practical checklist: what to ask first when code is slow

When GPU utilization is low during LLM training or inference, check in the following order. First, see whether the CPU is holding things back: data loading, tokenization, logging, and frequent `.item()` calls can all make the GPU wait. Second, check whether kernels are too fragmented: if a Transformer block launches many short small kernels, many elementwise operations are not fused, and kernel launch plus HBM round trips will consume time. Third, check whether matrix shapes are suitable for Tensor Cores: hidden size, vocab size, and batch×sequence should be aligned to hardware-friendly multiples. Fourth, check whether memory usage forces the batch size to be too small: a small batch reduces matrix multiplication arithmetic intensity and insufficiently exposes parallelism. Fifth, use a profiler to confirm the bottleneck rather than relying only on `nvidia-smi` utilization, which provides only a coarse signal and cannot tell which kernel is slow.

## 12. A running example: why the “same model” can have very different speed

Suppose two implementations train the same Transformer with identical parameter count, batch size, and dtype. Implementation A directly composes attention, MLP, residual, and normalization from many small PyTorch operations. Implementation B uses fused layer norm, FlashAttention, a fused optimizer, and pads vocab size and hidden size to friendlier dimensions. Mathematically they are almost equivalent, but implementation B writes far fewer intermediate tensors, reduces CPU/GPU synchronization, and makes matrix multiplication land more consistently on efficient tiles. The result may not be a 5% improvement, but tens of percent or more.

This is the engineering intuition this lecture aims to build: a single formula in a model architecture paper becomes many kernels, many memory transactions, and many scheduling decisions on a GPU. LLM infrastructure requires understanding not only models, but also how formulas translate into hardware costs: which tensors are materialized, which intermediate values can be recomputed, which operations should be fused, and which dimensions trigger an extra wave of tiles. Once you master this intuition, later topics such as Triton, custom kernels, distributed training, and inference serving connect naturally.

A final practical rule: if an optimization reduces HBM reads and writes, improves Tensor Core utilization, lowers synchronization, or increases effective batch/sequence parallelism, it is usually worth trying. If it merely rewrites code at a lower level without changing the data movement path, the benefit is often limited.
