# Stanford CS336 Lecture 7 Tutorial: Parallelism 1

> Adaptation note: This English tutorial is adapted from the Chinese lecture notes. It keeps the original technical structure, terminology, formulas, and code, while rewriting explanations in natural tutorial-style English.

## Learning goals

After reading this lecture, you should be able to:

1. Explain why training large language models (LLMs) must scale from a single GPU to multiple GPUs, multiple nodes, and sometimes an entire data center.
2. Distinguish the core ideas of data parallelism, model parallelism, tensor parallelism, pipeline parallelism, and activation/sequence parallelism.
3. Understand the role of collective communication such as all-reduce, reduce-scatter, and all-gather in distributed training.
4. Reason about the trade-offs among compute, GPU memory, communication bandwidth, and batch size for different parallelization strategies.
5. Given a hardware topology, sketch a reasonable parallel training configuration for an LLM.

## Lecture map

This lecture is not about “how to make one kernel faster.” It is about “when the model is too large and training is too slow, how do we split the training job across many machines?” The main thread is:

- Start with motivation: a single GPU does not have enough FLOPs or memory, so we need multi-machine parallelism.
- Review communication basics: GPUs are not connected equally. NVLink/NVSwitch inside a node is fast, while cross-node InfiniBand or Ethernet is slower.
- Introduce data parallelism: replicate the model, split the data, and synchronize gradients; then use ZeRO/FSDP to reduce memory usage.
- Introduce model parallelism: instead of replicating the whole model, split the model itself, including pipeline parallelism and tensor parallelism.
- Finally discuss activation memory, sequence parallelism, communication/compute trade-offs, and practical rules for combining parallelism in large-model training.

## Core concepts

### 1. Why does LLM training need parallelism?

Large-model training has two hard constraints: compute and memory.

Compute means the total floating-point operations needed for training. The larger the model and the more tokens it sees, the more FLOPs training requires. Even though each new GPU generation is faster, one GPU alone cannot immediately satisfy frontier-model training requirements.

Memory means GPU memory. Parameters themselves are only part of the memory footprint. During training we also store:

- parameters: model parameters, usually BF16/FP16, about 2 bytes per parameter;
- gradients: about 2 bytes per parameter;
- optimizer states: Adam first and second moments plus master weights, often taking more space;
- activations: intermediate results from the forward pass saved for backpropagation.

A common estimate is that when training with Adam, each parameter may require about 16 bytes of training state. Therefore 7B, 70B, and larger models cannot simply fit onto one GPU.

### 2. Communication primitives: collective communication

Distributed training relies heavily on collective communication:

- all-reduce: every rank has a tensor; first reduce the tensors by summing/averaging, then send the result back to every rank. Data parallelism commonly uses it to synchronize gradients.
- reduce-scatter: reduce first, then scatter shards of the result to different ranks.
- all-gather: every rank has one shard; gather all shards so every rank obtains the full tensor.
- broadcast: copy data from one rank to all ranks.

Important equivalence:

```text
all-reduce ≈ reduce-scatter + all-gather
```

In bandwidth-bound settings, the communication volume is approximately equivalent. This explains why ZeRO can save memory without obviously increasing communication.

### 3. Hardware topology determines the parallel strategy

Parallel algorithms cannot be separated from hardware. In a typical NVIDIA training node, one machine may have 8 GPUs connected by high-speed NVLink/NVSwitch; different machines are connected by networks such as InfiniBand, with much worse bandwidth and latency.

Therefore the rule of thumb is:

- bandwidth-hungry tensor parallelism is usually placed within a node;
- data parallelism can cross slower networks, because each step only synchronizes one batch of gradients or parameter shards;
- pipeline parallelism communicates activations, mostly point-to-point, and can sometimes work across nodes or slower links.

## Step-by-step tutorial

When designing a parallelization plan, start by asking three questions. First, can one GPU hold the model parameters, gradients, optimizer states, and peak activations? If not, you must first shard the model or the training state. Second, where are the communication links fastest? Fast intra-node interconnects are suitable for frequent synchronization, while slow cross-node links should synchronize less often or mainly use point-to-point transfers. Third, how large can the effective batch size be for the current training run? If the batch size is already near the critical batch size, adding GPUs through data parallelism may only add communication without improving convergence speed.

A useful mental model is to view parallel strategies as splitting different dimensions. Data parallelism splits the sample dimension and mainly improves throughput. Tensor parallelism splits matrix width inside each layer and mainly solves layers that are too wide or too parameter-heavy. Pipeline parallelism splits network depth and mainly solves models with too many layers to fit as a whole. Sequence parallelism splits the sequence dimension and specifically handles activation components that tensor parallelism cannot reduce. Real systems usually do not choose only one; they stack them into so-called 3D/4D parallelism.

## Step 1: Data Parallelism: replicate the model, split the data

Data parallelism is the most natural parallelization method: every GPU stores the full model parameters but processes different data samples.

Let the global batch size be B and the number of GPUs be M. Each GPU processes B/M samples. Each GPU independently runs forward/backward, obtains local gradients, uses all-reduce to average gradients, and then every GPU performs the same optimizer step.

The SGD update can be written as:

```text
θ_{t+1} = θ_t - η · (1/B) · Σ_{i=1}^B ∇ℓ(x_i; θ_t)
```

Data parallelism simply splits this summation across GPUs.

Advantages:

- Good compute scalability: when the batch is large enough, more GPUs process more samples.
- Simple implementation and mostly insensitive to model architecture.

Disadvantages:

- Does not save memory: every GPU stores full parameters, gradients, and optimizer state.
- Communication volume depends on parameter count: every step must synchronize gradients.
- Limited by batch size: the number of GPUs cannot grow without bound beyond the effective batch size.

### Batch size is a resource

When batch size is small, data parallelism cannot keep scaling, because each GPU needs at least some useful samples. Even if batch size can be increased, optimization has a critical batch size: beyond a point, increasing batch gives diminishing returns, because the bottleneck changes from “gradient noise” to “number of gradient update steps.”

Thus batch size is not an unlimited free resource. It must be allocated among data parallelism, pipeline parallelism, and gradient accumulation.

## Step 2: ZeRO/FSDP: make data parallelism memory-efficient

Naive data parallelism wastes memory because every GPU replicates the full training state. ZeRO (Zero Redundancy Optimizer) progressively shards these states.

### ZeRO Stage 1: shard optimizer state

Every GPU still stores full parameters and gradients, but Adam optimizer states are split into shards. Each GPU is only responsible for updating the parameter shard it owns.

Flow:

1. Every GPU computes full gradients.
2. Use reduce-scatter to aggregate gradients to the GPU responsible for each parameter shard.
3. Each GPU uses its own optimizer state to update the corresponding parameter shard.
4. Use all-gather to collect the updated parameter shards back onto every GPU.

Communication-wise, reduce-scatter + all-gather is approximately equivalent to the original all-reduce, so Stage 1 is almost a “free” memory optimization.

### ZeRO Stage 2: also shard gradients

Stage 2 no longer keeps full gradients. During backpropagation, as soon as a layer's gradients are computed, they are reduced to the GPU responsible for that parameter shard, and the local temporary gradients are freed.

This avoids instantiating the full gradient vector in GPU memory. Total communication remains close to original data parallelism, but scheduling is more complex.

### ZeRO Stage 3 / FSDP: also shard parameters

FSDP (Fully Sharded Data Parallel) roughly corresponds to ZeRO Stage 3: parameters, gradients, and optimizer states are all sharded.

The core idea is “all-gather parameters on demand”:

1. Before a layer's forward pass, all-gather that layer's parameters.
2. After the layer computation, free the parameters.
3. During backward, all-gather parameters again on demand.
4. After gradients are computed, reduce-scatter them to the GPU responsible for the shard.

Communication grows from about 2×parameters to about 3×parameters, but with overlap communication and computation, prefetching, and related techniques, the practical overhead can be acceptable.

FSDP's advantage is generality: it does not require deep knowledge of the Transformer structure and is often the default memory optimization method for large-model training.

## Step 3: Model Parallelism: split the model itself

When the model or activations still do not fit, we need model parallelism. Unlike FSDP, the goal is not to “temporarily collect full parameters,” but to keep different parts of the model permanently on different GPUs. The main communicated objects become activations.

This lecture focuses on two kinds: pipeline parallelism and tensor parallelism.

## Step 4: Pipeline Parallelism: split the model by layers

Pipeline parallelism splits the model along the depth dimension: for example, GPU0 holds the first layers, GPU1 the middle layers, and GPU2 the final layers. During forward, activations flow from front to back; during backward, activation gradients flow from back to front.

A naive implementation creates severe bubbles: at a given moment only one GPU works while the others are idle. To reduce bubbles, the batch is usually split into multiple micro-batches, so different micro-batches occupy different pipeline stages like an assembly line.

A common approximation for bubble overhead is:

```text
bubble_ratio ≈ (pipeline_stages - 1) / micro_batches
```

Thus more micro-batches fill the pipeline better. But this also consumes batch-size resources.

Advantages:

- Parameters and some activations are distributed by layer, so memory scales well.
- Communication is mostly point-to-point activation transfer between adjacent stages.
- Can be suitable across slower network links.

Disadvantages:

- Scheduling is complex, especially advanced strategies such as 1F1B, interleaved pipeline, and zero-bubble pipeline.
- Bubbles reduce GPU utilization.
- Engineering is very hard and often requires deep integration with autograd and runtime scheduling.

One trick in zero-bubble pipeline is to split backward into two kinds of work:

- B: backpropagate activation gradients, with strict dependencies;
- W: compute weight gradients, with fewer dependencies, so it can be moved into bubbles.

This uses otherwise idle time to compute parameter gradients and improves utilization.

## Step 5: Tensor Parallelism: split the model by matrix width

Tensor parallelism splits matrix multiplications along the width dimension. Most Transformer computation comes from large matrix multiplications, so weight matrices can be split into submatrices; multiple GPUs compute partial results and combine them through collective communication.

For example, in an MLP:

```text
Y = GeLU(XA)
Z = YB
```

We can split A and B into A1/A2 and B1/B2. Each GPU handles part of the matrix multiplication and uses all-reduce to combine results when necessary.

Advantages:

- Does not consume batch size.
- No pipeline bubble.
- Natural for Transformer-style models dominated by matrix multiplication.

Disadvantages:

- Every layer may have synchronization barriers.
- Communication is activation communication, frequent and bandwidth-hungry.
- Usually only appropriate within fast intra-node interconnects, such as an 8-GPU NVLink/NVSwitch environment.

Rule of thumb: tensor parallel size is often set to the number of GPUs inside one node, such as 8. If tensor parallelism crosses nodes over slower links, throughput can drop sharply.

## Step 6: Activation and Sequence Parallelism

So far we have mainly handled parameters, gradients, and optimizer state, but activation memory can also become a major issue. Its peak usually occurs early in backpropagation: many forward activations have not yet been released, while gradients begin to accumulate. The longer the sequence and the larger the batch, the more obvious this becomes; long-context training is especially prone to activation bottlenecks.

Each Transformer layer's activations approximately contain two terms:

```text
activation_memory_per_layer ≈ S · B · H · 34 + 5 · A · S² · B
```

Here S is sequence length, B is batch size, H is hidden size, and A is the number of attention heads. The S² term on the right comes from quadratic components such as attention softmax, and can be greatly reduced by FlashAttention and recomputation.

Tensor parallelism can shard many matrix-multiplication-related activations, but pointwise operations such as layer norm, dropout, and residual-stream inputs may still keep full activations. Sequence parallelism shards these pointwise activations along the sequence dimension: different GPUs are responsible for different token positions.

This introduces extra all-gather / reduce-scatter, but can further reduce activation memory. Combined with activation recomputation, it often trades more compute for lower memory, enabling larger batches or larger models.

## Formula and pseudocode quick reference

### Data-parallel training pseudocode

```python
for batch in data:
    local_batch = shard(batch, rank)
    loss = model(local_batch)
    loss.backward()
    all_reduce(model.gradients)   # synchronize gradients across all ranks
    optimizer.step()
    optimizer.zero_grad()
```

### FSDP idea pseudocode

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

### Rules for combining parallel strategies

```text
First make sure the model fits in memory:
    1. Prefer tensor parallelism within a node
    2. If it still does not fit, use FSDP/ZeRO-3 or pipeline parallelism

After the model fits:
    3. Use the remaining GPUs for data parallelism to scale throughput
    4. If communication is too frequent, use gradient accumulation to increase effective batch
```

## Common misconceptions

1. “More GPUs are always faster.”
   Wrong. Communication, synchronization, pipeline bubbles, and batch-size limits all reduce scaling efficiency.

2. “Data parallelism solves memory problems.”
   Naive DDP does not. Only sharding techniques such as ZeRO/FSDP significantly reduce parameter-related memory.

3. “FSDP is model parallelism.”
   Not exactly. FSDP shards parameters, but gathers them on demand during computation; model parallelism emphasizes fixed parameter placement and mainly communicates activations.

4. “Tensor parallelism can freely scale across nodes.”
   Usually not. It communicates every layer and is extremely sensitive to bandwidth and latency, so it is best placed on fast intra-node interconnects.

5. “Pipeline parallelism is conceptually simple, so it is simple to implement.”
   The opposite is true. Efficient pipeline scheduling, micro-batches, 1F1B, zero-bubble scheduling, and autograd integration are all complex.

6. “Activation memory is not important.”
   For long-sequence, large-batch, large-model training, activations can become the main memory bottleneck.

## Exercises

1. Suppose a model has P parameters and is trained with Adam, with about 16 bytes of training state per parameter. Estimate the parameter-related training-state memory required for a 7B-parameter model.

2. Explain in your own words why:

```text
all-reduce ≈ reduce-scatter + all-gather
```

and describe how this equivalence helps ZeRO Stage 1.

3. In an 8-GPU node, NVLink inside the node is fast, while the cross-node network is slower. Where would you place tensor parallelism, pipeline parallelism, and data parallelism? Why?

4. If pipeline stages = 8 and micro-batches = 32, estimate bubble_ratio. What happens if the number of micro-batches is halved?

5. Explain the main difference between FSDP and tensor parallelism: what do they communicate, and what hardware properties do they depend on?

## Summary

The essence of LLM training parallelism is managing four scarce resources at the same time: memory, compute, communication bandwidth/latency, and batch size. Data parallelism is simple and good for scaling throughput, but is limited by batch size and memory replication. ZeRO/FSDP shard optimizer states, gradients, and parameters so data parallelism can also save memory. Pipeline parallelism splits the model by layers, with gentler communication but bubbles and complex scheduling. Tensor parallelism splits the model by matrix width and does not consume batch size, but requires fast interconnects. Sequence parallelism and activation recomputation further address activation memory.

In real training there is no single best plan. A common recipe is: use tensor parallelism within a node, combine it with sequence parallelism when needed; if the model still does not fit, add FSDP or pipeline parallelism; finally use data parallelism to consume the remaining GPUs, and use gradient accumulation to adjust communication frequency. Understanding what each strategy communicates and what hardware it requires is key to designing efficient LLM training systems.

From an engineering perspective, a good parallelization plan is not about chasing a term, but about keeping expensive GPUs from waiting. Communication can be prefetched and overlapped, memory can be traded against sharding and recomputation, and batch size can amortize synchronization costs, but every choice shifts pressure onto another resource. The core conclusion of Lecture 7 is that large-scale training is a systems-design problem: algorithms, hardware topology, optimizer state, activation lifetimes, and scheduling strategy must be considered together.
