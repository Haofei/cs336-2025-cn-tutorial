# Stanford CS336 Lecture 4 Tutorial: Mixture of Experts (MoE)

> Adaptation note: This tutorial is translated and adapted from the Chinese tutorial version. It preserves the original structure, formulas, code snippets, and technical terminology while making the explanations natural for English-speaking learners.

## Learning objectives

After this lecture, you should be able to:

1. Explain the core advantage of Mixture of Experts (MoE) over a dense model: more total parameters with roughly the same amount of activated computation.
2. Understand key terms such as router / gating, expert, top-k routing, and load balancing.
3. Write the basic formula and pseudocode for an MoE forward pass.
4. Analyze MoE costs in training and inference: FLOPs, memory, communication, load imbalance, and token dropping.
5. Understand the engineering trade-offs common in modern MoE systems such as DeepSeek, Mixtral, Grok, and Llama 4.

## Prerequisites

This lecture assumes that you already understand the basic Transformer architecture, especially self-attention and FFN/MLP blocks; softmax, top-k, and residual connections; next-token prediction in language model training; and the basics of GPU/TPU parallel training.

## Lecture map

The main idea of MoE can be summarized in one sentence: replace the expensive FFN layers in a Transformer with multiple “expert” FFNs, and let each token activate only a small number of them.

This lecture discusses, in order:

- why MoE often outperforms a dense model at the same FLOPs;
- how the router chooses experts for tokens;
- how large experts should be, how many to use, and whether shared experts are useful;
- why load balancing is critical for training MoE;
- the system costs of MoE: communication, memory, parallelism, and token dropping;
- how the DeepSeek series combines these ideas into modern large-scale MoE architectures.

## Core concepts

### 1. The difference between a dense model and MoE

In a standard Transformer, each layer usually contains attention and an FFN. In a dense model, the FFN is one fixed large MLP: every token goes through the same FFN.

MoE replaces this FFN with multiple experts. Each expert is usually also an FFN, but each token does not go through all experts. Instead, a router selects K of them. If each token activates only 1 or 2 experts, the compute mainly depends on the activated parameters, not on the model’s total parameters.

Therefore, MoE has several advantages:

- More total parameters: the model has higher capacity and can memorize or express more patterns.
- Fewer activated parameters: each token uses only a small part of the parameters, so FLOPs do not grow linearly with the total number of experts.
- Natural fit for expert parallelism: different experts can be placed on different devices.

But MoE also introduces complexity: routing is a discrete choice and is harder to optimize; different experts can receive severely imbalanced loads; and sending tokens across devices creates communication overhead.

### 2. Experts are not necessarily “semantic experts”

The name “Mixture of Experts” can be misleading. It does not guarantee that one expert handles code, another handles math, and another handles Chinese. More precisely, an expert is a sparsely activated subnetwork. Experts may develop some specialization, but that specialization is usually not an interpretable human domain division.

A better mental model is this: MoE provides multiple optional nonlinear transformation paths in each layer. The router chooses a few paths based on the current hidden state. Since the hidden state already contains context, position, and the results of previous layers, the same surface token may be sent to different experts in different contexts. For example, “Python” may route differently in a programming context and an animal context, but that does not mean an expert can simply be named the “programming expert.”

### 3. Router / Gating

The router is a lightweight module. It takes a token’s hidden state as input and outputs an affinity score for each expert. A common implementation is a linear projection followed by softmax or sigmoid, then selecting the K experts with the highest scores.

The router output is also often called gating weights, which are used to combine the outputs of multiple experts with weights.

### 4. Top-K Routing

Modern large-scale MoE models have largely converged on token choice top-k routing: each token independently chooses the K experts with the highest scores.

Other approaches also exist:

- expert choice: each expert chooses the tokens it wants to process. This naturally balances load, but the chosen experts may not be the best choices for each token.
- global assignment: solve a global matching / optimal transport problem. This is elegant, but computationally expensive.
- hashing routing: use a hash function to assign tokens in a fixed way. Surprisingly, this can also bring gains, but it is less flexible than learned routing.
- RL routing: use reinforcement learning to handle the discrete choices. It is reasonable in principle, but in practice the cost and variance are too high.

## Step-by-step tutorial

### Step 1: Replace the FFN with a pool of experts

A standard FFN can be written as:

```text
h_out = h + FFN(h)
```

An MoE layer can be written as:

```text
h_out = h + sum_{i in TopK(router(h))} g_i(h) * Expert_i(h)
```

where:

- `h` is the current token hidden state;
- `Expert_i` is the i-th FFN;
- `router(h)` gives the token’s scores for all experts;
- `TopK` keeps only K experts;
- `g_i(h)` is the gating weight. It may or may not be fully normalized, depending on the implementation.

If K=1, the compute is close to one dense FFN. If K=2, it is roughly equivalent to activating two FFNs, so the FLOPs are approximately doubled. But the model’s total parameters can be far larger than those of a dense model.

### Step 2: Understand the number and size of experts

Early MoE models often copied the dense FFN into many same-sized experts. Later systems such as DeepSeek found that fine-grained experts are very effective: make each expert smaller, but use more of them.

For example, suppose the original FFN intermediate dimension is 4 times the hidden size. Fine-grained MoE can cut each expert’s intermediate dimension to 1/2, 1/4, or even smaller, then activate more experts. This increases the flexibility of routing combinations while controlling FLOPs.

Another design is the shared expert: regardless of what the router selects, each token always passes through one or more shared FFNs. The motivation is to preserve some general-purpose processing ability instead of making all computation depend on sparse routing. The DeepSeek series has used shared experts, but ablation studies in other models show that the gains are not always stable, so this remains an engineering choice.

### Step 3: Why load balancing is indispensable

Without constraints, the router can easily fall into a bad local optimum: all tokens are sent to only a few experts, while other experts are barely trained and become dead experts. This wastes memory and makes the MoE degrade into a much smaller model.

Therefore, training MoE usually adds an auxiliary load balancing loss. A common form from Switch Transformer is:

```text
L_balance = alpha * N * sum_i f_i * p_i
```

where:

- `N` is the number of experts;
- `f_i` is the fraction of tokens actually routed to expert i;
- `p_i` is the average router probability assigned to expert i;
- `alpha` is the loss weight.

The intuition is: if an expert already receives too many tokens, its corresponding routing probability is pushed down, encouraging tokens to spread to other experts.

Load balancing is not only a systems optimization, but also a modeling optimization. Even if we ignore GPU utilization, it is needed to avoid expert collapse.

### Step 4: Stability issues during training

MoE is hard to train mainly for three reasons:

1. top-k is a discrete choice, so unselected experts receive no gradient;
2. the router can become biased toward a few experts too early;
3. a softmax router can introduce numerical instability in low-precision training.

Common stabilization techniques include:

- compute the router in float32;
- add a z-loss to router logits to constrain the softmax normalizer;
- add load balancing loss;
- sometimes add noise or jitter to encourage exploration;
- use more data during fine-tuning to avoid overfitting caused by the huge number of total parameters.

DeepSeek V3 proposes auxiliary-loss-free balancing: maintain a bias `b_i` for each expert. If expert i has recently received too few tokens, increase `b_i`; if it has received too many, decrease `b_i`. This bias is used only for routing selection, not as the final gating weight. In practice, DeepSeek V3 still keeps a sequence-wise auxiliary loss to control load imbalance within a single sequence.

### Step 5: Training and inference costs

MoE FLOPs look attractive, but real system cost is not only FLOPs.

Training costs include:

- matrix multiplication cost for the activated experts;
- memory cost for storing all expert parameters;
- extra computation for the router and load balancing;
- communication cost for token dispatch and combine;
- idle time caused by load imbalance across devices.

Inference costs are similar. Although each token activates only a few experts, the weights of all experts must be stored somewhere. If experts are distributed across multiple GPUs, all-to-all communication is needed: first send tokens to the devices that host their selected experts, compute the expert outputs, then send the results back.

If one expert receives too many tokens, the system may trigger token dropping: tokens beyond capacity do not pass through that expert and are propagated only through the residual connection. This can make training or inference results depend on other requests in the same batch, introducing seemingly strange nondeterminism.

### Step 6: Expert Parallelism

Expert parallelism is an important systems advantage of MoE. Since experts are naturally partitioned, different experts can be placed on different devices. The process is roughly:

1. each device holds some token hidden states;
2. the router decides which experts each token should visit;
3. all-to-all sends tokens to the corresponding devices;
4. experts run their FFNs;
5. all-to-all sends outputs back to the original positions;
6. results are combined according to gating weights.

This adds another axis to large-model parallelism: in addition to data parallelism, tensor/model parallelism, and pipeline parallelism, we can use expert parallelism. But communication topology, batch size, number of experts, and capacity factor all affect efficiency.

An important rule of thumb is that expert computation must be “thick” enough to hide all-to-all communication cost. If each expert is too small and receives only a few tokens each time, the GPU will see many small matrix multiplications and fragmented communication, so real wall-clock time may be poor. Therefore, modern implementations use fused kernels, block-sparse matrix multiplication, libraries such as MegaBlocks, and similar techniques to organize expert computation into batches better suited to hardware.

### Step 7: Evolution of the DeepSeek series

DeepSeek MoE is the modern case repeatedly referenced in this lecture. Early DeepSeek MoE already used two key designs: fine-grained experts and shared experts, together with top-k routing and auxiliary load balancing. DeepSeek V2 kept the overall structure mostly unchanged, but scaled to tens of billions of activated parameters and added top-M device selection: first restrict the set of devices a token can access, then choose experts within those devices, reducing cross-device communication.

DeepSeek V3 kept the MoE backbone but changed the router: it uses gentler sigmoid-style scores and introduces an online load-based bias for each expert, implementing so-called auxiliary-loss-free balancing. It still keeps a sequence-wise auxiliary loss to prevent a single abnormal sequence from overloading a few experts during inference. This evolution shows that the basic MoE architecture is not complicated; real progress often comes from details of training stability, communication control, and load balancing.

## Formulas and pseudocode

### Top-K MoE forward pass

```python
# h: [tokens, d_model]
# W_router: [d_model, n_experts]
# experts: list of FFN modules
# k: number of active experts

scores = h @ W_router              # [tokens, n_experts]
probs = softmax(scores, dim=-1)    # or sigmoid + normalization
indices = topk(probs, k)           # selected experts per token

output = zeros_like(h)
for token t:
    for expert i in indices[t]:
        weight = probs[t, i]
        output[t] += weight * experts[i](h[t])

h_next = h + output
```

### Load balancing loss

```text
f_i = fraction of tokens actually routed to expert i
p_i = average router probability assigned to expert i
L_balance = alpha * N * sum_i f_i p_i
```

### DeepSeek V3-style bias update

```text
if load_i < target_load:
    b_i = b_i + gamma
else:
    b_i = b_i - gamma

routing_score_i = router_score_i + b_i
```

Note: `b_i` is used to decide top-k, but when the expert outputs are finally combined, the original gating score is usually still used.

## Common misconceptions

1. “Experts are human-interpretable domain experts.”
   Not necessarily. Experts are sparsely activated subnetworks. They may specialize, but usually not as clear code/math/Chinese experts.

2. “MoE has more parameters, so it must be more expensive.”
   It has more total parameters, but each token activates only a small number of them. FLOPs depend on activated parameters, while memory depends on total parameters. When training or deploying MoE, you must report total parameters, activated parameters, and number of active experts per token; otherwise model scale is easy to misread.

3. “More experts are always better.”
   More experts increase memory and communication complexity. If routing is imbalanced, many experts die; if experts are too fine-grained, communication fragmentation becomes worse. What matters is having more usable and sufficiently trained experts, not merely more experts written in a config file.

4. “After softmax, top-k can be removed.”
   It cannot be removed casually. If all experts participate in computation, MoE loses the advantage of sparse computation, and training and inference cost explode. Softmax mainly produces comparable, weightable scores; top-k enforces sparse activation. They solve different problems.

5. “Load balancing is only for GPU utilization.”
   Not only. It also prevents expert collapse and ensures that all parameters get trained. Without load balancing, the model’s loss may still decrease, but in reality only a few experts are being trained, wasting most of the capacity.

6. “MoE inference must be faster than dense inference.”
   Not necessarily. If the model is so large that weights are distributed across many machines and GPUs, communication and scheduling can offset the gains from sparse computation. MoE works best at sufficiently large scale with a mature inference system; for small single-GPU models, a dense model may be simpler and more stable.

## Practice exercises

1. Hand-write a toy MoE layer: input 2D vectors, use 4 experts, activate top-2 each time, and observe the routing of different tokens.
2. Turn off load balancing loss, record the number of tokens each expert receives, and see whether expert collapse appears.
3. Compare K=1 and K=2: how do training loss, computation, and expert utilization change?
4. Implement a capacity factor: each expert can receive at most a fixed number of tokens; tokens beyond that are dropped. Observe whether outputs are affected by batch composition.
5. Read the DeepSeek V3 technical report and identify two key designs outside MoE: MLA (Multi-head Latent Attention) and MTP (Multi-token Prediction).

## Summary

MoE is an important architecture in modern high-performance language models. Its core idea is simple: use a router to choose a small number of experts for each token, decoupling total parameters and activated parameters at different ratios. This can achieve lower loss at the same training FLOPs, or use a larger-capacity model at similar inference compute.

The real difficulties are engineering and optimization: discrete top-k routing is hard to train; experts can be imbalanced; cross-device communication can become a bottleneck; and token dropping during inference can introduce nondeterminism. Modern systems make MoE practical for large-scale training and deployment through top-k routing, fine-grained experts, shared experts, load balancing loss, expert parallelism, auxiliary-loss-free balancing, and related techniques.

## Connection to the next lecture

This lecture focused on MoE architecture and training. The next step is to understand system-level parallel strategies for large models more deeply: how data parallelism, tensor parallelism, pipeline parallelism, and expert parallelism combine, and how communication bandwidth, memory, and batch size jointly determine real training throughput. Only after understanding these system details can you judge whether MoE is “worth it” on a particular hardware cluster. If you continue into the systems part of the course, always separate “mathematical FLOPs” from “time on the machine”: MoE papers often show the former, while engineering success often depends on the latter.
