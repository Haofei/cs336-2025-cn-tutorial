# Stanford CS336 Lecture 8 Tutorial: Parallel Training (Part 2)

> Adaptation note: This English tutorial is adapted from the Chinese lecture notes. It preserves the original Markdown structure, technical terms, formulas, and code-style notation, while rewriting the explanation in natural tutorial English.

This lecture continues the systems-level discussion of large-model training: when a single GPU can no longer hold the model, optimizer state, or a sufficiently large batch, how do we distribute computation and data across multiple GPUs or even multiple nodes while preventing communication from becoming the bottleneck? The core principle is the same as in single-GPU optimization: keep expensive compute units as busy as possible, increase arithmetic intensity, and reduce unnecessary data movement.

## 1. A hardware view of multi-GPU training

A training cluster can be understood as a hierarchical storage and communication system:

- Inside a GPU: SMs execute computation; L1/shared memory is fast but small, while HBM is larger but slower.
- Multiple GPUs in one node: GPUs are connected through PCIe or NVLink/NVSwitch. NVLink is NVIDIA's dedicated high-bandwidth link for GPU-to-GPU communication and is usually far faster than PCIe.
- Multiple nodes: communication must also pass through NICs, switches, and other network equipment, with lower bandwidth and higher latency.

Therefore cross-GPU communication is more expensive than local HBM access, and cross-node communication is even more expensive than same-node communication. The goal of distributed training engineering is not to “avoid communication completely,” but to:

1. reduce communication volume;
2. place communication in suitable parts of the system;
3. overlap communication with computation as much as possible;
4. choose parallelization strategies according to hardware topology.

In the NVIDIA ecosystem, low-level communication is usually handled by NCCL. NCCL chooses communication paths according to the GPU topology and compiles high-level collective operations into low-level CUDA kernels and data transfers. PyTorch's `torch.distributed` provides convenient Python-level interfaces such as `all_reduce`, `all_gather`, and `reduce_scatter`.

## 2. Collective communication primitives

Distributed training frequently uses collective operations. Suppose there are `world_size` processes or devices, and each device is identified by a `rank`.

Common primitives include:

- `broadcast`: copy data from one rank to all ranks.
- `scatter`: distribute different slices from one rank to different ranks.
- `gather`: collect data from multiple ranks onto one rank.
- `reduce`: reduce data from multiple ranks by sum, average, max, etc., and place the result on one rank.
- `all_gather`: every rank obtains the concatenated data from all ranks.
- `reduce_scatter`: first reduce the inputs from all ranks, then distribute different slices of the reduced result to different ranks.
- `all_reduce`: every rank obtains the full reduced result from all ranks' data.

An important equivalence is:

```text
all_reduce = reduce_scatter + all_gather
```

For example, in data-parallel training, each GPU computes gradients on a different data shard, then uses `all_reduce` to average them so all GPUs keep consistent parameter updates. Many advanced strategies split `all_reduce` into `reduce_scatter` and `all_gather` to reduce peak memory or better overlap communication with computation.

When using these primitives, pay special attention to synchronization. Collective operations require all ranks in the same process group to call them in the same order. If one rank misses an `all_reduce`, other ranks may wait forever, and the program will appear to hang.

## 3. Benchmark: do not look only at theoretical bandwidth

H100 NVLink has very high theoretical bandwidth, but actual training bandwidth depends on tensor size, number of ranks, communication pattern, NCCL algorithm, node topology, and whether communication crosses nodes. In the course, benchmarking `all_reduce` with large tensors shows that practical bandwidth can be noticeably lower than the hardware specification; `reduce_scatter` may also differ from simple estimates.

Good engineering habits include:

- benchmark the target cluster directly rather than only reading product specifications;
- test same-node and cross-node communication separately;
- examine throughput and latency for different message sizes;
- use warmup, `torch.cuda.synchronize()`, barriers, and similar techniques to avoid timing errors;
- distinguish “bytes required by the algorithm” from “wall-clock time.”

Communication performance is hard to predict exactly from formulas alone, because NCCL uses implementation details such as ring algorithms, tree algorithms, hierarchical communication, and in-network reduction. Therefore practical tuning must rely on profiling and benchmarking.

## 4. Data-parallel DDP: split the batch, synchronize gradients

Data parallelism is the most intuitive parallel method: every GPU stores the full model but processes a different batch slice. Each rank independently runs forward and backward, obtains local gradients, and then averages all parameter gradients with `all_reduce`.

The training step can be summarized as:

1. split the global batch into local batches by rank;
2. each rank processes its local batch with the same model parameters;
3. backward produces local gradients;
4. run `all_reduce(mean)` on each parameter gradient;
5. each rank performs an optimizer step using the same gradients.

The loss on each rank may differ because the data differs, but after gradient synchronization the parameters remain consistent.

DDP is simple and has good compute scalability. Its drawback is that every GPU must store full parameters, gradients, and optimizer states. As models grow, memory becomes the first bottleneck. Another engineering point is that `all_reduce` is itself a synchronization point; if some ranks are slower, the others wait. This is the straggler problem.

## 5. ZeRO and FSDP: shard parameters, gradients, and optimizer states

To overcome DDP's memory limit, we can shard model state. ZeRO (Zero Redundancy Optimizer) divides training state into levels:

- ZeRO-1: shard optimizer states, such as Adam's first and second moments;
- ZeRO-2: additionally shard gradients;
- ZeRO-3: also shard parameters.

FSDP (Fully Sharded Data Parallel) can be viewed as PyTorch's ZeRO-3-like implementation: each rank permanently holds only part of the parameters. When computing a layer, it temporarily collects the full parameters for that layer using `all_gather`; after backpropagation, it reduces and reshards gradients with `reduce_scatter`.

FSDP's basic trade-off is:

- benefit: greatly reduces resident memory per GPU, enabling larger models to train;
- cost: frequent parameter `all_gather` and gradient `reduce_scatter` during forward/backward;
- engineering focus: choose wrapping granularity, prefetch, and bucket size carefully to avoid communication fragmentation.

If sharding is too fine, per-layer communication and scheduling overhead increase. If sharding is too coarse, peak memory rises. In practice, Transformer blocks are often used as FSDP units, combined with mixed precision, activation checkpointing, and CPU/offload strategies.

## 6. Activation checkpointing: trade recomputation for memory

Backpropagation needs activations saved during forward. Long sequences, large batches, and deep Transformers can make activation memory very large. Activation checkpointing saves only some intermediate results and recomputes missing activations during backward.

This is a classic “compute for storage” trade-off:

- without checkpointing: less computation, but many activations must be saved;
- checkpointing all or part of the blocks: lower memory, but backward requires extra forward recomputation.

In practice, do not blindly recompute everything; choose an appropriate granularity. Checkpointing is often done at the Transformer-block level. For simple pointwise operations immediately following matmul, it may not be necessary to save all intermediates because recomputation is cheap. Checkpointing is often paired with FSDP/ZeRO: after parameters, gradients, and optimizer states are sharded, activations may become the new memory bottleneck.

## 7. Tensor Parallel: split the hidden dimension, use frequent collectives

Tensor parallelism splits internal matrix dimensions of the model rather than the batch. For an MLP linear layer, the weight matrix can be split by columns or rows across multiple ranks. Each rank stores only part of the weights and computes part of the output.

The simplified course example is: each rank owns part of the hidden dimension for every layer. After local activations are computed, `all_gather` is needed to concatenate activations from all ranks into the full hidden vector before entering the next layer.

Characteristics of Tensor Parallel:

- advantage: extremely large matrices in a single layer can be distributed across multiple GPUs;
- disadvantage: every layer or every few operators may require collectives;
- suitable hardware: strongly depends on fast interconnects, usually preferred within same-node NVLink/NVSwitch.

In Transformers, more common splits include attention heads, MLP intermediate dimension, or vocabulary projection. Different splits correspond to different collectives: sometimes forward needs `all_reduce`, sometimes backward does; sometimes `reduce_scatter` + `all_gather` can optimize memory and communication.

## 8. Pipeline Parallel: split layers and handle pipeline bubbles

Pipeline parallelism splits the model along “depth”: rank 0 stores the first layers, rank 1 stores the later layers, and so on. During forward, the previous stage sends activations to the next stage; during backward, gradients are sent back.

The problem with a naive pipeline is the bubble: if one full batch is sent at a time, early stages are idle while later stages compute, and later stages are idle while waiting for input. The solution is to split the batch into multiple microbatches, so different microbatches flow through different stages simultaneously.

The intuitive rules for bubbles are:

- more pipeline stages mean larger fill and drain overhead;
- more microbatches reduce the bubble fraction;
- too many microbatches increase scheduling overhead and may affect batch norm, optimizer behavior, and related details.

Real systems must also design forward/backward schedules, such as GPipe's all-forward-then-all-backward schedule or 1F1B (one-forward-one-backward). To reduce waiting, use asynchronous `isend/irecv` so communication can overlap with computation for subsequent microbatches. Otherwise synchronous send/recv can often block the GPU.

## 9. Combining parallelism and engineering tuning

Large-model training usually combines multiple kinds of parallelism rather than using only one:

- Data parallelism: scale throughput across more nodes;
- FSDP/ZeRO: reduce model-state memory;
- Tensor Parallel: handle layers that are too wide, too large for one GPU, or too computationally heavy;
- Pipeline Parallel: handle very deep models through layer-wise splitting;
- Activation checkpointing: reduce activation memory;
- Sequence/context parallel: split the sequence dimension for long-context training.

A common rule is to put the most frequent and fine-grained tensor-parallel communication inside same-node high-speed interconnects; scale coarser data parallelism across nodes; and add pipeline parallelism when necessary. When choosing a strategy, consider memory, compute utilization, communication bandwidth, latency, and code complexity together.

Engineering tuning checklist:

1. Identify the bottleneck first: use profilers to determine whether the system is compute-bound, memory-bound, or communication-bound.
2. Adjust batch and microbatch: increasing microbatch can improve matmul utilization but increases activation memory.
3. Tune FSDP bucket/prefetch: make `all_gather` and `reduce_scatter` overlap with computation as much as possible.
4. Avoid too many small collectives: small messages are latency-dominated, and merging buckets is often faster.
5. Match the topology: use tensor parallelism within a node; avoid high-frequency cross-node communication.
6. Check synchronization points: barriers, loss logging, and checkpoint saving can all introduce waiting.
7. Maintain determinism and consistency: all ranks must enter collectives in the same order; random seeds, data sharding, and dropout must also be handled correctly.
8. Save full checkpoints regularly: under sharded training, checkpoints may be distributed across multiple ranks, so the save/restore format must be explicit.

## 10. A practical selection process

If choosing a parallelization plan from scratch, reason in the following order. First estimate single-GPU memory: how much is used by parameters, gradients, optimizer states, and activations? If the full model plus optimizer state does not fit, prioritize FSDP or ZeRO. If the main issue is activations from long sequences or large microbatches, enable activation checkpointing first. Second, check whether any single layer is too large: if attention heads, MLP intermediate layers, or vocabulary projection are tight in compute and memory on one GPU, tensor parallelism is needed, and the tensor-parallel group should be placed within one machine's fast interconnect whenever possible. Third, examine model depth: if there are many layers and FSDP plus tensor parallelism is still insufficient, add pipeline parallelism to split by layer. Only fourth should you scale data parallelism by placing multiple model replicas on more nodes to increase total throughput.

When tuning, do not look only at tokens per second. Also monitor GPU utilization, communication-time fraction, peak memory, and step-time variance. If GPU utilization is low and communication time is high, the parallel split is too fine or cross-node communication is too frequent. If memory is near the limit but communication is not high, add checkpointing or finer sharding. If pipeline stages are imbalanced, some ranks will wait for long periods, so repartition layers or adjust the number of microbatches. Real distributed training is an iterative process of measurement, bottleneck identification, and partitioning changes.

## 11. Common failure signals

The most common problem in distributed programs is not an error message, but a hang. When there is no output for a long time, first check whether all ranks entered the same collectives, whether tensor shapes match, and whether send/receive sources and destinations are paired correctly. If training runs but speed fluctuates sharply, check data loading, logging, checkpoint saving, and cross-node network congestion. If one rank uses much more memory than others, it usually indicates uneven model partitioning, imbalanced pipeline stages, or unreleased activations. When debugging, reproduce with a smaller model and fewer ranks first, then scale up gradually. After any change, measure again instead of relying on intuition about bottlenecks and optimization effects.

## 12. Summary

The core of this lecture is: distributed training is about partitioning data, parameters, gradients, optimizer states, and activations, while trading off computation, memory, and communication. DDP is simple but memory-redundant. ZeRO/FSDP reduces memory through sharding but increases communication. Activation checkpointing trades recomputation for memory. Tensor parallelism is good for large matrices but needs fast collectives. Pipeline parallelism can split deep models but must handle bubbles and scheduling.

Hardware will continue to improve, but model scale will also continue to approach hardware limits. Therefore system issues such as hierarchical memory, communication bottlenecks, recomputation, and sharding will not disappear. In large-model training, real performance comes from co-designing algorithms, model structure, parallelization strategy, and hardware topology.
