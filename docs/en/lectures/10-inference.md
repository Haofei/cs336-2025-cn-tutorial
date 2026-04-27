# Stanford CS336 Lecture 10 Tutorial: LLM Inference

> Adaptation note: This English tutorial is adapted from the Chinese lecture notes. It preserves the original structure, technical terminology, formulas, and code-style notation, while presenting the material in clear tutorial English.

This lecture discusses **inference**, meaning generation at serving time: given a fixed, already-trained model, produce a response from a user's prompt. Training is usually a large one-time cost, while inference is called repeatedly in chat, code completion, batch processing, model evaluation, test-time compute, RL sampling, and many other settings. As a result, inference efficiency often directly determines product cost and user experience.

## 1. What should inference optimize?

LLM inference is usually measured by three main metrics:

- **TTFT (Time To First Token)**: the waiting time from when the user submits the prompt to when the first output token appears. It is mainly determined by prompt-processing time and is very important for interactive applications.
- **Latency**: the speed at which tokens arrive after generation begins. This is often understood as how long each token takes, or the streaming speed perceived by the user.
- **Throughput**: how many tokens the system can generate per unit time, usually measured in tokens/s. Batch jobs care more about throughput, while chat products must also maintain low latency.

Low latency and high throughput often conflict. To improve throughput, systems tend to combine more requests into a larger batch; but the larger the batch, the longer an individual user may have to wait for the current computation step to finish.

## 2. Why is inference different from training?

During Transformer training, the tokens in the whole sequence are known, so computation can be parallelized along the sequence dimension. Matrix multiplications are large enough to keep GPU/TPU compute units busy.

During inference, especially **autoregressive generation**, token `t` depends on the tokens already generated before it. Therefore the decode phase must proceed step by step. Each step usually generates only 1 token per sequence, which turns many operations into relatively “skinny” matrix multiplications or matrix-vector multiplications that are hard to run at full hardware utilization.

A key concept for understanding this is **arithmetic intensity**: how many FLOPs are performed per byte read from or written to memory. If arithmetic intensity is high, the computation is usually compute-bound; if it is low, it is usually memory-bound. GPUs such as H100 have enormous compute capability, but HBM memory bandwidth is still limited. If each step reads many parameters and a large KV cache but performs relatively little computation, the GPU ends up “waiting for memory.”

In training or prefill, `batch size × sequence length` is large enough to give high arithmetic intensity. In token-by-token decode, `T=1`, arithmetic intensity drops sharply, and inference can easily become bottlenecked by memory bandwidth.

## 3. Prefill and Decode: the two phases of inference

LLM inference can be divided into two phases.

### 3.1 Prefill

Given a prompt, the model processes all prompt tokens at once, computes the key/value tensors needed for attention in every layer, and obtains the logits for the next token. This phase resembles the training forward pass: the sequence dimension can be parallelized, and it is usually compute-bound and relatively fast. TTFT comes largely from prefill, especially when the prompt is long.

### 3.2 Decode / Generation

The model generates one new token from the current context, appends it to the context, and then continues to the next step. This phase is highly sequential, has `T=1`, is usually memory-bound, and is the hardest part of inference systems to optimize.

## 4. KV Cache: the core mechanism for avoiding repeated computation

The most naive generation method is to feed the full context back through the transformer after every generated token. This recomputes the key/value tensors for all historical tokens, giving very poor complexity.

The idea of the **KV cache (Key-Value cache)** is that in a causal transformer, the key/value tensors of historical tokens do not change when a new token arrives, so they can be cached. During prefill, the KV cache for the prompt is built. During decode, the model computes K/V only for the new token and appends them to the cache. Each step no longer recomputes the full prefix; it reads the historical KV cache to perform attention.

The size of the KV cache is roughly proportional to:

- batch size: how many sequences are served simultaneously;
- sequence length: how many tokens each sequence already contains;
- number of layers: every layer stores its own cache;
- number of KV heads: the number of key/value heads;
- head dimension: the dimension of each head;
- both K and V tensors, as well as numerical precision such as BF16.

Therefore the KV cache can easily become a major consumer of GPU memory. Long contexts, high concurrency, and large batches all quickly increase memory usage.

## 5. Why is memory bandwidth the bottleneck?

In MLP layers, different requests share the same model weights. The larger the batch, the more tokens can be served after reading the weights once, so batching improves arithmetic intensity.

Attention decode is harder: each sequence has its own KV cache. Even if the batch becomes larger, each request still needs to read its own historical K/V, which cannot be reused across the batch the way MLP weights can. As a result, the arithmetic intensity of attention decode stays close to constant and is often low enough to be memory-bound.

This explains a central principle of inference optimization: **reducing the amount of data that must be read from or written to HBM is often more important than simply reducing FLOPs.**

## 6. The tradeoff among batch size, latency, and throughput

Increasing batch size can improve throughput: one decode step generates 1 token for each of `B` requests simultaneously. The costs are:

1. each step must process more sequences, so per-step latency may increase;
2. every sequence must keep its KV cache, so memory usage grows with batch size;
3. throughput gains have diminishing returns and are ultimately limited by memory capacity.

If the system serves only one user, latency can be low but GPU utilization is poor. If it aggregates many users, throughput is better but users may wait longer. This is the basic tension in LLM serving.

A simple but effective scaling method is **replication**: place a separate copy of the model on each of multiple GPUs. Unlike training, no complex gradient synchronization is needed; latency remains roughly unchanged, and total throughput approximately increases with the number of replicas. If the model is too large, more complex strategies such as tensor parallelism, pipeline parallelism, or KV cache sharding become necessary.

## 7. Architectural techniques for reducing the KV cache

Because decode is mainly limited by KV cache reads and writes, many modern architecture changes can be understood as ways to “make the KV cache smaller.”

### 7.1 GQA: Grouped-Query Attention

In traditional multi-head attention, the numbers of query heads, key heads, and value heads are the same. **MQA (Multi-Query Attention)** takes the extreme approach of making all query heads share one set of K/V, but this may reduce expressiveness. **GQA (Grouped-Query Attention)** is a compromise: multiple query heads share a smaller number of KV heads.

This does not reduce query expressiveness too much, but it can greatly reduce the number of heads stored in the KV cache. Once the KV cache becomes smaller, memory usage decreases, memory transfer decreases, and both latency and throughput improve. It also allows a larger batch size.

### 7.2 MLA: Multi-head Latent Attention

**MLA (Multi-head Latent Attention)**, proposed in the DeepSeek family, does not necessarily reduce the number of KV heads. Instead, it projects K/V into a lower-dimensional latent space and caches that. In other words, it does not cache the full high-dimensional K/V; it caches compressed representations and reconstructs them or uses them in computation when needed. It reduces the KV cache along the dimension axis, with the same goal of lowering memory and bandwidth pressure.

### 7.3 CLA: Cross-Layer Attention

**CLA (Cross-Layer Attention)** shares K/V representations across layers. GQA shares across heads; CLA shares across layers. Since the KV cache normally stores a copy for every layer, sharing across layers can further reduce cache size, though it requires a tradeoff between model quality and efficiency.

### 7.4 Local / Sliding Window Attention

**Local attention** or **sliding window attention** attends only to the most recent `K` tokens. When generating very long sequences, K/V entries outside the window can be discarded, so the cache no longer grows linearly with total sequence length and instead stays roughly fixed by the window size.

The problem is that purely local attention damages the ability to model long-range dependencies. Therefore real models often use hybrid designs: most layers use local attention while a few layers retain full/global attention, or local attention is combined with KV sharing, GQA, and related techniques.

## 8. More radical directions: changing the Transformer

If the KV cache of full attention is the fundamental bottleneck, one direction is to reduce the number of full-attention layers or even replace parts of the transformer architecture.

- **State Space Models (SSM) / Mamba**: replace the sequence-growing KV cache with an RNN-like state representation, making inference state close to constant size. The challenge is preserving language-modeling ability, especially for tasks such as associative recall that require precise retrieval of distant information.
- **Linear Attention**: rewrite attention using kernels or feature maps so complexity becomes linear rather than quadratic, with an implementation form similar to recurrent state. Modern models often mix linear, local, and full attention.
- **Diffusion language models**: no longer generate strictly autoregressively token by token. Instead, generate a span of text in parallel and repeatedly refine it. This makes it easier to saturate hardware, but text quality and generality remain research questions.

These methods show that inference optimization is not only systems engineering; it also pushes model architecture design in return.

## 9. Quantization, pruning, and distillation

**Quantization** reduces memory usage and bandwidth by lowering numerical precision, for example from BF16 to FP8, INT8, or even INT4. Because inference is often memory-bound, reducing the number of bytes per parameter or KV element can directly improve speed and capacity. Low precision, however, introduces error, especially when large models contain outliers, meaning unusually large activations or weights. Common approaches include post-training quantization, keeping outliers in higher precision, and activation-aware quantization.

**Pruning** deletes unimportant layers, heads, or hidden dimensions so the model structure itself becomes smaller. A pruned model usually gets worse, so pruning is often combined with **distillation**: using the original large model as a teacher to transfer capabilities into a pruned or smaller student model.

These methods are usually lossy: they improve speed and cost, but the resulting quality must be verified.

## 10. Speculative Decoding: using a small model to speed up a large model

The key observation behind **speculative decoding / speculative sampling** is that verifying a given string of tokens is faster than generating those tokens one by one. Verification can be parallelized like prefill, while generation must be autoregressive.

The procedure is:

1. Use a cheap **draft model** to autoregressively generate `K` candidate tokens.
2. Use the expensive **target model** to compute the probabilities of these tokens in parallel.
3. Use an accept-reject rule to decide how many draft tokens to keep.
4. If a token is rejected, sample from the target model's corrected distribution and continue.

Mathematically, this method can guarantee that the output distribution is the same as directly sampling from the target model, i.e. “exact sampling from the target model,” as long as the accept-reject step is implemented correctly. The speedup depends on how close the draft model is to the target model: the more accurate the draft, the higher the acceptance rate and the faster the method. Medusa, EAGLE, and related methods focus on better drafts or parallel draft generation.

## 11. Serving systems: system issues under real traffic

During training, a batch is usually a tidy dense block of tokens. During serving, requests are dynamic: they arrive at different times, have different prompt lengths and generation lengths, may share prefixes, and may finish quickly or slowly. This requires the serving system to schedule dynamically.

### 11.1 Continuous Batching

**Continuous batching** does not wait for an entire batch to finish before admitting new requests. Instead, after every decode step, control returns to the scheduler: completed requests leave, and new requests join. This reduces GPU idle time and improves throughput.

### 11.2 Selective Batching

Requests have different lengths, so it is hard to batch the attention part perfectly. But the MLP part does not depend on interactions between sequences, so tokens of different lengths can be flattened into the batch dimension and computed together. This is the idea of **selective batching**.

### 11.3 PagedAttention and vLLM

The dynamic growth of the KV cache causes GPU memory fragmentation. The final generation length of a request is unknown, so pre-allocation wastes memory; after requests finish, they leave non-contiguous holes. **PagedAttention** borrows the idea of virtual memory from operating systems: it cuts the KV cache into fixed-size blocks/pages, so a request's logically contiguous context can be mapped onto physically non-contiguous GPU memory blocks. This reduces fragmentation and improves memory utilization.

If multiple requests share a prefix, **copy-on-write** can also be used: they share the same KV blocks, and copying happens only when later generation diverges, saving additional memory.

## 12. Sampling and decoding strategies

The previous sections mainly discussed “how to compute the logits for the next token faster,” but actually generating text also requires a **decoding strategy**: choosing a token from logits or a probability distribution.

The simplest method is **greedy decoding**, which chooses the highest-probability token at every step. It is stable, cheap, and reproducible, but it can produce templated text and may lock into a path too early for multi-step reasoning or creative writing. **Beam search** keeps multiple candidate sequences at the same time and is common in traditional machine translation. For open-ended LLM dialogue, however, it may produce repetitive, conservative answers and increases computation and memory pressure.

Random sampling is more common. **Temperature** rescales logits: low temperature makes the distribution sharper and the output more deterministic; high temperature makes the distribution flatter and the output more diverse. **Top-k sampling** samples only from the `k` highest-probability tokens and cuts off the low-probability tail. **Top-p / nucleus sampling** chooses the smallest token set whose cumulative probability reaches `p`, dynamically deciding the candidate set size. Real serving systems often add repetition penalty, frequency penalty, stop sequences, maximum output length, and other rules to control repetition, termination, and cost.

These strategies do not themselves change the transformer's main computational bottleneck, but they affect generation length, acceptance rate, and user-perceived quality. For example, the speed of speculative decoding depends on the probability that draft tokens are accepted by the target model. The higher the temperature and the more random the sampling, the harder it is for a small draft model to predict the large model, so the acceptance rate may fall. Therefore inference systems often need to tune the model, sampling parameters, and serving objectives together.

## 13. Summary

The central difficulty of LLM inference comes from autoregressive decode: the model generates only one token at a time, is hard to parallelize, and repeatedly reads model weights and each sequence's private KV cache. As a result, it is usually limited by memory bandwidth. Prefill is relatively easy to parallelize; decode is the main bottleneck.

Optimization paths can be grouped into several categories:

- Systems: continuous batching, selective batching, PagedAttention, model replication, and parallelism;
- Architecture: GQA, MLA, CLA, local attention, SSM, linear attention, and diffusion models;
- Model compression: quantization, pruning, and distillation;
- Decoding algorithms: speculative decoding, which trades small-model drafts plus large-model verification for lossless speedup.

The final goal is not merely “to run a fixed transformer faster.” It is to deliver the highest-quality model outputs possible under constraints on latency, throughput, memory, and cost. Inference efficiency has become a central driver of co-design across modern LLM architectures, algorithms, and systems.
