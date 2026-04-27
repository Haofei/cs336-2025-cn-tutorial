# Stanford CS336 2025 Language Modeling from Scratch — Complete English Tutorial


Source: Stanford CS336 2025 public course videos/transcripts. This is a tutorial-style adaptation for study, not an official Stanford document.


---


# CS336 2025 Lecture 1 Tutorial: Course Overview and Tokenization

> This is an English tutorial adaptation of the Chinese CS336 2025 study guide.

## Learning goals

After this lecture, you should be able to:

1. Explain that CS336, “Language Models from Scratch,” is not about merely calling APIs, but about understanding and implementing the language-model building pipeline.
2. Describe the central question of the course: given a compute budget and a data budget, how do we train the best possible model?
3. Recognize the full workflow: tokenizer, Transformer architecture, training, systems optimization, scaling laws, data, evaluation, and alignment.
4. Understand tokenization: converting Unicode strings into integer sequences that a model can process.
5. Explain the basic idea, training procedure, and encode/decode logic of BPE (Byte Pair Encoding).

## Prerequisites

Recommended background:

- Basic Python and PyTorch.
- Basic machine-learning concepts: loss, optimizer, batch size, overfitting.
- Some familiarity with Transformer and attention; the course rebuilds these from the bottom up.
- You do not need to be a GPU expert, but you should be willing to reason about performance from an engineering perspective.

## Lecture map

This lecture has two main parts:

1. Course overview: why build language models from scratch, what modules the course covers, and how engineering and research perspectives fit together.
2. Tokenization basics: why tokenizers are needed, what goes wrong with character-, byte-, and word-level schemes, and how BPE provides a practical compromise.

The course storyline is:

```text
raw data → cleaning/filtering → tokenizer → integer sequences → Transformer → training → evaluation → systems optimization → alignment/fine-tuning → usable model
```

## 1. Why build language models from scratch?

The lecture begins with a trend: researchers are increasingly far from the underlying technology. In earlier NLP work, researchers often implemented and trained models themselves. Later, many projects downloaded models such as BERT and fine-tuned them. Today, many applications only prompt a closed model through an API.

This abstraction is useful, but it leaks. A language-model API may look like “string in, string out,” but without understanding data, model architecture, systems, and training, it is hard to do foundational research. CS336’s motto is:

```text
To understand it, you have to build it.
```

The course is also realistic. Frontier models require enormous capital, GPU clusters, and unpublished engineering details. A class cannot have every student train a GPT-4-scale model. Instead, the course trains small models while emphasizing what small-scale experiments can and cannot teach us.

The course separates three kinds of knowledge:

- mechanics: how Transformer is implemented and how GPU parallelism works. This can be taught concretely.
- mindset: always think in terms of efficiency and scaling; make hardware do useful work.
- intuitions: which data and architecture choices work at scale. These can only be partially learned from small experiments.

## 2. Core perspective: efficiency, not blind scale

The instructor warns against misreading the bitter lesson as “only scale matters and algorithms do not.” A better summary is:

```text
Algorithms at scale matter.
```

Model quality is roughly the result of resources multiplied by efficiency. The more expensive the resources, the more important efficiency becomes. If one training run costs a large amount of money, you cannot waste attempts the way you might in a local toy experiment. Algorithmic efficiency, hardware utilization, data quality, and architecture all affect the final result.

The course repeatedly asks:

```text
Given a compute budget and a data budget, what is the best model we can train?
```

This is the engineering core: do not only ask “can the model be larger?” Ask whether each FLOP, GPU, and token is being used effectively.

## 3. Brief background of language models

Language models are not new. Shannon used language models to estimate the entropy of English. In traditional NLP, language models were components in machine translation, speech recognition, and other systems. Deep learning then accumulated several key ingredients:

- neural language models
- seq2seq models
- Adam optimizer
- attention mechanism
- Transformer
- model parallelism
- foundation models such as ELMo, BERT, and T5

After GPT-2 and GPT-3, scaling laws and engineered training became central. Model openness also split into layers:

- closed models: accessible only through APIs.
- open-weight models: weights are available, but data and training details may be incomplete.
- open-source models: weights, data, and implementation are made as open as possible, though papers still cannot replace building the system yourself.

## 4. The five course modules

### 4.1 Basics

The goal is to implement a minimal but complete language-model training pipeline:

- tokenizer: maps between strings and integer sequences.
- model architecture: mostly Transformer.
- training: loss, optimizer, learning-rate schedule, and training loop.

Assignments implement a BPE tokenizer, Transformer, cross-entropy loss, AdamW optimizer, and training loop. PyTorch is allowed, but prebuilt Transformer implementations are not the point.

### 4.2 Systems

Training is not only formulas; it is hardware. GPU compute is on-chip, while GPU memory is often off-chip, so data movement can dominate. The course discusses:

- kernels: for example, how matrix multiplication uses tiling and fusion to reduce data movement.
- Triton: a tool for writing high-performance GPU kernels.
- parallelism: data parallelism, tensor/model parallelism, and related strategies.
- inference: the process of generating tokens with a trained model.

Inference has two phases:

- prefill: process the prompt; all input tokens are known and can be parallelized, similar to training.
- decode: autoregressively generate one token at a time; this often underuses GPUs and becomes memory-bound.

The course also mentions speculative decoding: a cheaper small model proposes candidate tokens, then the large model verifies them in parallel to accelerate inference.

### 4.3 Scaling laws

Central question: given a FLOPs budget, how should we balance model parameters and training tokens? A larger model can see fewer tokens; a smaller model can see more. Where is the optimum?

The lecture discusses Chinchilla optimal ideas: fit scaling relationships on small experiments, then predict the best parameter count and loss for large runs. The value is that cheap experiments can guide expensive training decisions.

A caution: rules of thumb about the ratio between model parameters and training tokens have assumptions and usually ignore inference cost. They should not be applied mechanically.

### 4.4 Data and evaluation

Model capabilities are largely determined by data: multilingual data creates multilingual ability; code data creates coding ability. Common sources include Web/Common Crawl, Wikipedia, GitHub, StackExchange, books, and papers.

But “feed the internet to the model” is misleading. Raw web data contains HTML, PDFs, code repositories, spam, boilerplate, duplicates, and legal/safety issues. It needs:

- extraction: convert HTML/PDF and other formats into text.
- filtering: remove low-quality, harmful, or irrelevant content.
- deduplication: delete repeated data so training budget is not wasted.
- legal considerations: discuss what data can be used for training.

Evaluation includes:

- perplexity: how well the model predicts the next token.
- standardized benchmarks such as MMLU.
- instruction-following evaluation.
- evaluation of agentic systems that include language models.

### 4.5 Alignment

A pretrained base model mainly predicts the next token. It has raw capability, but may not follow instructions. Alignment makes a model more useful, safer, and better suited for interaction.

Alignment goals include:

- instruction following
- style control, such as length, bullet formatting, and tone
- safety, including refusing harmful requests

Common stages include:

- SFT, supervised fine-tuning: supervised learning on user/assistant prompt-response pairs.
- learning from feedback: improve the model using preference data or a verifier.
- PPO, DPO, GRPO: reinforcement-learning or preference-optimization methods. DPO is designed for preference data; GRPO is a simplified PPO-like method used by DeepSeek-style training.

## 5. Tokenization: why do we need a tokenizer?

Language models process numerical tensors, while raw text is a Unicode string. Tokenization converts a string into an integer sequence and, ideally, can decode it back into the original string.

A tokenizer has two directions:

```text
encode: string → list[int]
decode: list[int] → string
```

Vocabulary size is the number of possible token IDs. A larger vocabulary often lets one token represent longer text fragments, but it also makes the input/output layers larger. A smaller vocabulary creates longer sequences, increasing attention cost.

An important detail: modern tokenizers are usually reversible and encode spaces. For example, “hello” and “ hello” may be different tokens. This differs from traditional whitespace word splitting.

## 6. Simple tokenization schemes and their problems

### 6.1 Character-based tokenization

Map every Unicode character to its code point. English letters and emoji all have corresponding integers.

Problems:

- Unicode has a very large code-point range.
- Many characters are extremely rare but still occupy vocabulary space.
- Compression is poor.

### 6.2 Byte-based tokenization

Encode the string as UTF-8 bytes. Each byte ranges from 0 to 255, so the vocabulary is tiny and any text can be represented.

Advantages: simple, elegant, and no unknown-character problem.

Problem: sequences are too long. Each token represents only one byte, so the compression ratio is low. Since standard attention is quadratic in sequence length, byte-level sequences are inefficient for today’s Transformer architectures.

### 6.3 Word-based tokenization

Split text into words or fragments using spaces, regular expressions, or pre-tokenization rules, then assign each word an integer.

Advantage: common words can be represented by one token, making sequences shorter.

Problems: the vocabulary may become huge, and there will always be unseen words, spellings, names, code fragments, and other rare strings. Handling them with an UNK token loses information and complicates evaluation.

## 7. BPE: Byte Pair Encoding

BPE is an old compression algorithm later adopted for neural machine translation and then language models such as GPT-2. Its key idea is: do not manually decide what counts as a word; learn tokens from corpus statistics.

Intuition:

- Frequent consecutive fragments should be merged into one token to improve compression.
- Rare fragments can remain split; they need not consume vocabulary entries.
- Starting from bytes guarantees that every string can be represented.

BPE training:

```text
Input: training corpus, target vocabulary size or number of merges
Initialize: convert text into byte sequences; initial vocabulary is 0..255
Repeat:
  1. Count every adjacent token pair in the current sequences
  2. Find the most frequent pair, for example (116, 104)
  3. Assign a new token id to that pair, for example 256
  4. Replace all occurrences of that pair in the training sequences with the new token
Output: merge rules and vocabulary
```

Encoding new text:

```text
1. Convert the string into bytes
2. Apply learned merge rules in training order
3. Return the integer token sequence
```

Decoding:

```text
1. Map each token id back to its byte sequence
2. Concatenate the bytes
3. Decode the bytes as UTF-8
```

GPT-2-style tokenizers also use pre-tokenization: a regular expression splits text into pieces, and BPE runs inside each piece. This is a practical compromise for efficiency and behavior control.

## 8. Common misconceptions

1. “Training from scratch is the first step for every problem.”
   No. If prompting or fine-tuning solves the problem, use that first. Training from scratch is for learning foundations or for cases that truly need a new base model.

2. “Small-model conclusions always transfer to large models.”
   Not always. FLOP ratios between attention and MLP, emergent behavior, and stability can change with scale.

3. “Tokenization is just a minor preprocessing detail.”
   No. Tokenizers directly affect sequence length, training efficiency, vocabulary size, reversibility, and multilingual/code performance.

4. “Byte-level tokenization is clean, so it must be best.”
   It is elegant, but with current Transformer architectures it usually creates sequences that are too long.

5. “Internet data can be trained on directly.”
   No. Raw Common Crawl-style data contains spam, duplicates, HTML/PDF structure, and legal/safety issues. It must be carefully processed.

## 9. Practice exercises

1. Open a tokenizer visualization tool. Try English, Japanese or Chinese, numbers, code, and emoji. Observe token boundaries.
2. Implement a minimal byte tokenizer with `encode(str) -> list[int]` and `decode(list[int]) -> str`.
3. On a tiny corpus, manually perform three BPE merges. Record the highest-frequency pair and new token id each time.
4. Compare token counts for the same text under character-, byte-, and BPE-level tokenization. Think about how sequence length affects attention cost.
5. Randomly sample web text and judge what is high quality, what should be filtered, and what should be deduplicated.

## 10. Summary

This lecture establishes the CS336 frame: a language model is not an isolated Transformer, but an end-to-end engineering pipeline. The course builds the tokenizer, model, training loop, systems optimizations, data pipeline, evaluation, and alignment methods from the bottom up, always asking how to get the best model under limited compute and data.

Tokenization is the entrance to the pipeline. Character-, byte-, and word-level schemes each have serious drawbacks. BPE starts from bytes and repeatedly merges frequent adjacent tokens, giving a practical balance between universal representability and compression. Tokenizer-free architectures may become mature in the future, but BPE and its variants remain foundational in current frontier-model practice.

## Further reading and next lecture

- Andrej Karpathy’s videos on tokenization and building models from scratch.
- The original Transformer paper: “Attention Is All You Need.”
- GPT-2 tokenizer and byte-level BPE implementations.
- Papers on Chinchilla scaling laws.

Next lecture moves into PyTorch details and resource accounting: writing code that runs is not enough; we also need to track FLOPs, memory, and data movement to understand where compute resources go.


---


# Stanford CS336 2025 Lecture 2 Tutorial: PyTorch and Resource Accounting

> This is an English tutorial adaptation of the Chinese CS336 2025 study guide.

This lecture is not “another Transformer overview.” Its theme is a lower-level skill that becomes essential when training large models: using PyTorch to build models while continuously estimating memory, compute, time, and money. Research code cannot merely “run.” When parameter counts, token counts, and GPU counts grow, every matrix multiplication, optimizer state, and CPU/GPU transfer becomes real cost.

## 1. Why resource accounting matters

The lecture starts with two paper estimates.

First: with 1024 H100 GPUs, how long would it take to train a 70B-parameter dense Transformer on 15 trillion tokens? A rough formula is:

```text
training FLOPs ≈ 6 × number of parameters × number of tokens
usable FLOPs/day ≈ number of GPUs × peak FLOPs/s per GPU × MFU × 86400
training days ≈ total training FLOPs / usable FLOPs per day
```

If H100 effective utilization, or MFU, is 0.5, the answer is on the order of a hundred-plus days. The important point is not the exact number, but the habit: estimate total compute first, then divide by actual hardware throughput.

Second: with 8 × 80GB H100s, using AdamW and no sophisticated memory optimization, how large a model can fit? A common rough estimate is about 16 bytes per parameter: parameter, gradient, Adam first moment, Adam second moment, and related state.

```text
maximum parameters ≈ 8 × 80GB / 16 bytes ≈ 40B parameters
```

This ignores activations, batch size, sequence length, and other buffers, so it is only an upper-bound estimate. In real training, activations often become the bottleneck too.

## 2. PyTorch tensors: the atoms of everything

In PyTorch, parameters, gradients, optimizer state, data, and intermediate activations are all tensors. Understanding tensor storage is the first step in memory accounting.

A tensor’s memory is determined by the number of elements and the number of bytes per element.

```python
x = torch.zeros(4, 8)       # default float32
x.numel()                   # 32 elements
x.element_size()            # 4 bytes
memory = 32 * 4             # 128 bytes
```

Common numerical types:

| Type | Bytes/element | Notes |
|---|---:|---|
| FP32 / float32 | 4 | Traditional default; stable but slower and memory-heavy |
| FP16 / float16 | 2 | Saves memory and can be fast, but has limited dynamic range and can underflow/overflow |
| BF16 / bfloat16 | 2 | Similar exponent range to FP32 with less precision; common for deep learning |
| FP8 | 1 | Supported by newer hardware such as H100; strong speed/memory benefits but harder training stability |

FP16 and BF16 are both 16-bit, but distribute bits differently. FP16 gives more bits to the mantissa and has smaller dynamic range. BF16 keeps an FP32-like exponent range, so it can represent very small and very large values, which is useful for large-model training. In practice, master parameters and optimizer states may be stored in FP32, while forward/backward matrix multiplications use BF16 or FP8.

That is the core of mixed-precision training: the whole model does not use one dtype everywhere; each part trades stability against throughput.

## 3. Device placement and data movement

PyTorch creates tensors on CPU by default:

```python
x = torch.zeros(32, 32)     # CPU RAM
x = x.to("cuda")           # GPU HBM
```

You can also create directly on GPU:

```python
x = torch.zeros(32, 32, device="cuda")
```

During training, always know where each tensor lives. Moving data from CPU RAM to GPU HBM is not free. Frequent transfers make the GPU wait for data instead of doing computation. Production and research code often includes assertions or logs checking `x.device`, so a batch, mask, or loss target does not accidentally remain on CPU.

## 4. A tensor is a view of storage, not just an array

A PyTorch tensor points to underlying storage and carries metadata such as shape, stride, and offset. A contiguous 2D matrix may have stride `(4, 1)`: moving one row jumps four elements; moving one column jumps one.

This explains why many operations are nearly free. Slicing, transpose, and view often change metadata without copying data.

```python
x = torch.arange(6).view(2, 3)
y = x[0]       # view, shares storage
z = x.T        # transpose, usually shares storage
```

Shared storage has risks: in-place changes to `x` also affect `y`. Another common trap is contiguity. A transposed tensor is often not contiguous, so some `view` operations fail unless you first do:

```python
z = x.T.contiguous()
```

`contiguous()` may really copy data, so it is not free. High-performance code must distinguish “changed a view” from “allocated new memory.”

## 5. Naming dimensions reduces tensor bugs

Real model tensors are not just matrices; they have batch, sequence, head, hidden, and other dimensions. Code like `transpose(-2, -1)` or `view(b, s, h, d)` is common, but it becomes error-prone: does `-1` mean hidden or head_dim? Are comments still correct after a dimension changes?

The course recommends giving dimensions semantic names when possible. `einsum` expresses matrix multiplication with dimension meaning. Attention scores can be written as:

```python
scores = torch.einsum(
    "batch seq_q hidden, batch seq_k hidden -> batch seq_q seq_k",
    q, k,
)
```

The `hidden` dimension does not appear in the output, so it is summed over. `batch`, `seq_q`, and `seq_k` remain. The code directly says: take an inner product over hidden to get query-key similarity.

`einops.rearrange` is useful for reshape/transpose combinations, such as splitting the final dimension into heads:

```python
x = rearrange(x, "batch seq (heads dim) -> batch heads seq dim", heads=num_heads)
```

These tools do not necessarily reduce computation, but they dramatically reduce shape mistakes. In teaching and research code, readability is engineering efficiency: the easier tensor shapes are to see, the easier it is to account for resources and find performance problems.

## 6. Matrix multiplication is the main deep-learning cost

Most elementwise operations have FLOPs linear in the tensor size. In large models, the dominant computation is matrix multiplication.

For:

```text
[B, D] × [D, K] -> [B, K]
```

Each output element needs about D multiplications and D additions, so the rough FLOPs are:

```text
FLOPs ≈ 2 × B × D × K
```

This rule is crucial: matmul FLOPs are about 2 times the product of the three dimensions.

If `B` is understood as tokens or data points, and `D × K` as parameter count, a linear layer forward pass costs:

```text
forward FLOPs ≈ 2 × number of tokens × number of parameters
```

This roughly extends to Transformer as long as matrix multiplications dominate. Attention quadratic terms, sequence length, and non-matmul operations add corrections, but this is a useful paper estimate.

## 7. FLOPs versus FLOPs/s

“FLOPs” may mean floating-point operations, the total amount of work, while FLOPs/s means floating-point operations per second, the throughput. To avoid confusion:

```text
FLOPs      = total floating-point operations
FLOPs/s    = floating-point operations per second
```

Hardware vendors report peak FLOPs/s for A100, H100, and other GPUs under FP32, TF32, BF16, FP8, and sometimes structured sparsity assumptions such as 2:4 sparsity. If your model is dense, you cannot directly use the largest advertised sparse number.

Actual training depends on MFU:

```text
MFU = model effective FLOPs/s / hardware peak FLOPs/s
```

MFU measures how much of the hardware you are really using. Large matrix multiplications make high MFU easier. Small batches, fragmented kernels, communication, and data movement lower MFU. In practice, MFU above 0.5 is often good; a few percent usually indicates a serious code or parallelization bottleneck.

## 8. Autograd and the cost of backpropagation

PyTorch autograd saves us from manually writing gradients:

```python
pred = x @ w
loss = ((pred - y) ** 2).mean()
loss.backward()
w.grad        # filled by PyTorch
```

But autograd is not free. For a linear layer:

```text
X: [B, D]
W: [D, K]
H = XW: [B, K]
```

Forward cost is:

```text
2 × B × D × K
```

Backward computes at least:

```text
dL/dW = X^T × dL/dH
dL/dX = dL/dH × W^T
```

Each is also a matrix multiplication of roughly `2 × B × D × K`. Therefore:

```text
backward FLOPs ≈ 4 × B × D × K
```

For one complete training step over the main matmuls:

```text
forward + backward ≈ 6 × tokens × parameters
```

This is where the coefficient 6 in large-model training estimates comes from: about 2× for forward and 4× for backward.

## 9. Parameters, initialization, and nn.Module

Trainable PyTorch parameters are usually wrapped with `nn.Parameter` and placed inside an `nn.Module`. A simple deep linear network might be:

```python
class Cruncher(nn.Module):
    def __init__(self, d, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d, d, bias=False) for _ in range(num_layers)
        ])
        self.head = nn.Linear(d, 1, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)
```

Initialization should not blindly use a standard normal. If `W ~ N(0, 1)` and input dimension is large, output variance grows with fan-in and activations can explode. A common scaling is:

```python
w = torch.randn(d_in, d_out) / math.sqrt(d_in)
```

This matches the Xavier/Glorot idea: keep signal scale stable across layers. Truncated normal is sometimes used to avoid extreme values.

## 10. Optimizer state is a major memory cost

Training memory contains more than parameters. With Adam/AdamW, each parameter usually has:

1. the parameter itself
2. gradient
3. first moment `m`
4. second moment `v`
5. sometimes FP32 master weights or temporary buffers

If stored mostly in FP32, a dozen-plus bytes per parameter is common. This is why “number of parameters × dtype size” badly underestimates training memory.

Even simpler optimizers such as Adagrad store accumulated squared gradients. The optimizer `step()` reads `p.grad`, updates state, then updates parameters in place. State persists across steps, so it is long-term memory, not a temporary variable.

## 11. Activations: why forward intermediates are kept

Backpropagation needs intermediate activations from the forward pass. To compute the first layer’s weight gradient, for example, we need that layer’s input activation. Autograd therefore saves many intermediate results by default.

For a simple deep linear model with batch `B`, width `D`, and `L` layers, activation count is roughly:

```text
activations ≈ B × D × L
```

Total memory can be estimated by category:

```text
total memory ≈ bytes_per_elem × (parameters + gradients + optimizer state + activations)
```

For Transformer, activations also depend on sequence length, attention matrices, and MLP intermediate dimensions. If memory is insufficient, activation checkpointing trades extra compute for lower memory by not saving every activation and recomputing some during backward.

## 12. Data loading and the training loop

Language-model data is usually an integer sequence produced by a tokenizer. Real corpora may be terabytes, so they cannot all be loaded into RAM. A common approach is `numpy.memmap`, which maps arrays to disk files and reads slices on demand.

A typical training loop:

```python
model = Cruncher(d, num_layers).to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for step in range(num_steps):
    x, y = next_batch()
    x, y = x.to("cuda"), y.to("cuda")

    optimizer.zero_grad(set_to_none=True)
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
```

Engineering code must checkpoint periodically: model state, optimizer state, current step, random-number state, and more. Large training jobs will encounter interruption, preemption, OOM, or node failure. You cannot assume one run finishes uninterrupted. When resuming, also check the learning-rate scheduler, data position, and random seeds; otherwise, the “same” experiment silently becomes a different training curve. Reproducibility matters for research comparisons and for saving cluster time.

## 13. Compute, memory, and bandwidth must be considered together

Resource accounting cannot only count FLOPs. GPU training is constrained by compute units, memory capacity, and memory/interconnect bandwidth. Memory capacity determines whether the model and batch fit. FLOPs/s determines ideal matmul speed. Bandwidth determines how fast data moves through HBM, cache, CPU, GPU, and multi-GPU links.

An operation with many multiply-adds and little data movement is compute-bound; large matrix multiplication is the classic example. An operation such as elementwise add, mask, copy, or a reshape that triggers a contiguous copy may do few FLOPs but read/write lots of data, making it memory-bound. Even operations with few FLOPs can slow training because GPU cores wait for data.

Multi-GPU training adds communication bandwidth. Data parallelism synchronizes gradients; tensor parallelism exchanges intermediate results inside layers; pipeline parallelism sends activations between stages. Many systems optimizations do not reduce mathematical computation; they overlap compute and communication, remove unnecessary copies, and shape matrices to use Tensor Cores efficiently. Good training systems spend expensive FLOPs on large regular matmuls, not fragmented kernels and device transfers.

## 14. From research code to engineering cost awareness

The key idea of this lecture is: when writing a model, also write the cost ledger. For any model, ask:

- How many parameters does it have?
- How many bytes does each parameter really cost during training?
- How do activations grow with batch size and sequence length?
- What are the FLOPs of the main matrix multiplications?
- Does BF16/FP8 actually improve throughput?
- Is MFU 50% or 5%?
- Is performance limited by CPU/GPU transfers, non-contiguous copies, small kernels, or communication?

The course uses simple linear models so the formulas are transparent: forward is about `2 × tokens × params`, backward about `4 × tokens × params`, and training about `6 × tokens × params`; memory is split among parameters, gradients, optimizer state, and activations. Transformer accounting is more complex, but the method is the same.

Large-model engineering does not end when the network is mathematically correct. The code, numerical precision, hardware throughput, and training cost must all work together. PyTorch provides autograd and modular abstractions, but efficient training requires seeing through them: where tensors live, whether data is copied, what dtype is used, how large matmuls are, how much backward costs, and what the optimizer stores. With this resource-accounting mindset, research prototypes can become scalable, affordable, and reproducible training systems.


---


# CS336 2025 Lecture 3 Tutorial: Transformer Architectures and Hyperparameters

> This is an English tutorial adaptation of the Chinese CS336 2025 study guide.

This lecture asks: if you truly want to train a language model from scratch, what design choices matter beyond “what is a Transformer?” Modern large language models still use the Transformer core, but they are not identical to the 2017 version. A practical consensus recipe has emerged: pre-norm, RMSNorm, no bias, RoPE, SwiGLU, sensible width/depth ratios, and several stability techniques.

The sections below walk through these components and give rules of thumb for training new models.

## 1. From the original Transformer to the modern LLM Transformer

The original Transformer block roughly consists of:

1. token embedding and position encoding;
2. multi-head self-attention;
3. residual connections;
4. layer normalization;
5. feed-forward network, or MLP;
6. final output softmax.

Modern LLMs usually do not copy the original exactly. A block closer to LLaMA-style models and the course assignments is:

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

Common settings include:

- normalization before each sublayer, i.e. pre-norm;
- RMSNorm instead of traditional LayerNorm;
- linear layers usually without bias;
- RoPE for positional information;
- SwiGLU or other GLU variants in the MLP;
- in some newer models, an additional norm after sublayer outputs, creating a “double norm” structure.

The key is not to treat these as magic. They serve two goals: more stable training and better GPU efficiency.

## 2. Residual connections and normalization: the axis of stable training

### 2.1 Post-norm and pre-norm

The original Transformer used post-norm: run attention or MLP, add the residual, then apply LayerNorm:

```text
x = Norm(x + Attention(x))
x = Norm(x + MLP(x))
```

Modern LLMs mostly use pre-norm:

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

This looks like a small relocation, but the effect is large. The residual stream provides an almost identity path for gradients to flow from upper layers to lower layers. If normalization sits directly in the residual stream, it interferes with that path. In practice, post-norm is more prone to gradient explosions and loss spikes, and is more sensitive to warmup and learning rate. Pre-norm is usually more stable and easier for deep models.

A key modern rule: do not casually break the identity connection of the residual stream. Norms should mainly sit at the entrance or exit of non-residual branches, not repeatedly normalize the whole residual trunk.

### 2.2 LayerNorm and RMSNorm

LayerNorm normalizes each token’s hidden vector by subtracting the mean, dividing by standard deviation, then multiplying by a learned scale gamma and adding bias beta. RMSNorm is simpler: it does not subtract the mean and usually does not add beta; it scales by root mean square.

RMSNorm is popular because:

- quality is usually not worse than LayerNorm;
- it uses fewer operations;
- it has fewer parameters;
- more importantly, it reduces memory reads and writes.

In Transformer, most FLOPs come from matrix multiplication, but that does not make other operations irrelevant. Softmax and normalization have small FLOP counts yet can take significant wall-clock time because they are limited by memory movement. RMSNorm helps not only by doing slightly less arithmetic, but by moving less data.

### 2.3 Bias-free linear layers

Modern LLM linear layers often omit bias, including attention projections and MLP projections. Empirically this usually does not hurt quality, while reducing parameters and memory access. Some reports also suggest that removing bias improves optimization stability, especially at scale.

Summary: modern normalization choices target both stability and efficiency. Pre-norm keeps the residual stream clean, RMSNorm simplifies normalization, and no-bias linear layers reduce extra state and possible instability.

## 3. MLPs and activation functions: why SwiGLU became the default

Besides attention, the other major component in a Transformer block is the MLP. Early Transformers used ReLU, GPT-style models often used GELU, and many modern models use GLU variants, especially SwiGLU.

A standard MLP is:

```text
MLP(x) = W2 * activation(W1 * x)
```

GLU-style structures add a gate branch:

```text
MLP(x) = W2 * (activation(W1 * x) ⊙ (V * x))
```

Here `⊙` is elementwise multiplication. Intuitively, the model not only creates hidden features, but also learns a gate deciding which dimensions pass through and which are suppressed.

SwiGLU uses Swish as the nonlinearity:

```text
swish(x) = x * sigmoid(x)
```

Many models and ablations show that GLU variants often give a small but consistent improvement over ReLU/GELU MLPs. This does not mean a model cannot work without SwiGLU; GPT-3 did not use it and was still strong. But for a new model, SwiGLU is a safe default.

GLU adds the extra projection `V`. To keep parameter count similar to a standard MLP, the intermediate dimension is usually reduced to two thirds of the usual size. If a standard MLP uses `d_ff = 4 * d_model`, then SwiGLU often uses:

```text
d_ff ≈ 8/3 * d_model
```

That is why many LLaMA-like MLP hidden sizes are about 2.6 to 2.7 times `d_model`, not 4 times.

## 4. Attention and position encoding: RoPE’s modern role

Language models need token order. Earlier methods included sinusoidal position embeddings, learned absolute position embeddings, and relative position bias. Recent dense LLMs have largely converged on RoPE, rotary position embedding.

RoPE’s core idea is that attention often cares about relative distance, not absolute position. If a query and key are both shifted by the same amount while their relative distance stays the same, their inner-product relation should remain consistent.

RoPE implements this through rotation. It does not add a position vector at the input embedding. Instead, in every attention layer it applies a position-dependent rotation to query and key. Later positions rotate by larger angles; different dimension pairs use different frequencies, representing both short- and long-range information.

In 2D, if two vectors rotate by the same angle, their relative angle and inner product remain unchanged. RoPE splits high-dimensional vectors into many 2D pairs and rotates each pair by a fixed-frequency schedule. The query-key inner product therefore naturally encodes relative position.

RoPE is popular because:

- relative-position modeling is natural;
- it works well for both short and long contexts;
- many context-length extrapolation and extension tricks exist;
- it has been validated by many modern models.

Practical reminder: RoPE acts on Q and K, not by simply adding a vector to token embeddings. The rotation frequencies are usually a fixed schedule, not learned parameters.

## 5. Inference efficiency in attention: MHA, MQA, and GQA

In standard multi-head attention, every head has its own Q, K, and V. During training, full batches and sequences are processed at once, producing large matmuls and good GPU utilization. During inference, generation is autoregressive: one token at a time. To avoid recomputing old tokens’ K and V, systems maintain a KV cache.

The issue is that longer context means a larger KV cache. For every generated token, the model must read lots of historical K/V from memory. The bottleneck is often memory bandwidth, not compute.

MQA, multi-query attention, makes an aggressive simplification: keep multiple query heads, but let all heads share one set of K and V. This greatly reduces the KV cache and improves inference speed and long-context serving.

GQA, grouped-query attention, is a compromise: query heads are divided into groups, and each group shares one K/V set. It is more expressive than MQA and uses less KV cache than standard MHA. Many modern large models use GQA because it balances quality and inference cost.

Thus attention-head design is not only a training question; it is a deployment question. After release, much of a model’s cost is inference. GQA/MQA primarily help by reducing inference memory access and increasing throughput.

## 6. Rules of thumb for key hyperparameters

### 6.1 MLP intermediate dimension

For a standard ReLU/GELU MLP, a classic choice is:

```text
d_ff = 4 * d_model
```

For SwiGLU/GeGLU and other gated MLPs, to keep parameters similar:

```text
d_ff ≈ 8/3 * d_model
```

Ablations in Kaplan-style scaling-law work show that a fairly wide range of MLP ratios can work, but around 4× is a reasonable default. T5 once used an extreme 64× `d_ff`, proving the rule is not absolute; later T5 v1.1 returned to a more standard GLU ratio, showing conventional defaults remain competitive.

### 6.2 Attention head dimension

Common practice is:

```text
d_model = n_heads * d_head
```

Increasing the number of heads does not make total attention dimension grow without bound; `d_model` is split across heads. Most GPT-, PaLM-, and LLaMA-like models follow this approximately 1:1 setup. Very small head dimensions may theoretically create low-rank bottlenecks, but this default works well in practice.

### 6.3 Width/depth ratio

Model capacity can be increased by making the model wider or deeper. Width is controlled by `d_model`; depth by number of layers. Many models fall near:

```text
d_model / n_layers ≈ 100 to 128
```

This is not a law, but Kaplan-style experiments showed that the best width/depth region does not change dramatically over several scales.

System factors also matter. Deeper models suit pipeline parallelism, where layers are split across devices. Wider models suit tensor parallelism, where large matrices are split across GPUs. Hyperparameters are therefore influenced not only by loss, but also by cluster networks, parallel strategy, and memory limits.

### 6.4 Vocabulary size

Early English models often used 30k to 50k token vocabularies. Modern production models, especially multilingual ones, often use 100k to 250k or larger vocabularies.

A larger vocabulary helps because:

- multilingual text is split into fewer tokens;
- inference is cheaper for low-resource languages;
- emoji, code, and special symbols are covered better;
- large models can often use large vocabularies effectively.

If training a small English-only model, a smaller vocabulary is still fine. For general, multilingual, production-oriented models, larger vocabularies are the trend.

## 7. Dropout, weight decay, and training stability

Pretraining differs from traditional supervised learning: the dataset is huge and training usually does not run for many full epochs, so overfitting is not the main issue. This explains why dropout is less popular in modern LLM pretraining.

Weight decay, however, remains common. Here its role is not purely traditional regularization. Experiments show that weight decay interacts with the learning-rate schedule, especially cosine decay. During high-learning-rate phases it may appear to slow training, but as the learning rate decreases, models with weight decay can improve quickly and end with better training and validation loss.

In LLM pretraining, weight decay is better viewed as an optimization-dynamics tool, not only as regularization.

## 8. Large-model training stability: softmax is a risk area

As models grow and train longer, loss spikes and gradient-norm spikes matter more. A clear trend in modern architecture is stabilizing softmax operations. Transformer has two important softmaxes:

1. the output vocabulary softmax;
2. the attention softmax.

### 8.1 z-loss for output softmax

The output softmax computes:

```text
p(x) = exp(logit_x) / Z
```

where `Z` is the sum of exponentiated logits over the vocabulary. If `Z` becomes too large or unstable, softmax can cause numerical problems. z-loss adds an auxiliary term encouraging `log Z` to stay near 0, meaning the normalizer stays near 1.

PaLM used this technique, and later models adopted similar tricks. Its goal is not expressive power; it keeps output-softmax numerical ranges controlled.

### 8.2 QK norm for attention softmax

Attention softmax inputs come from QK inner products. If query/key norms become too large, logits become extreme, softmax can saturate, and gradients become unstable. QK norm normalizes Q and K before the inner product.

This directly controls the input scale to softmax. It was useful in vision Transformer and multimodal stability work, then entered text LLMs. A notable pattern is that normalization has expanded across modern models: block pre-norm, post-sublayer norm, and Q/K norm. Controlling activation scale is central to large-model training.

### 8.3 Logit soft capping

Another method is to softly cap attention logits, for example with `tanh`, so very large values are smoothly limited. Models such as Gemma 2 used similar ideas. It can control extremes, though it does not always improve quality; in some experiments QK norm is the safer choice.

## 9. Long-context attention: local windows and sparse structure

Full self-attention cost grows quadratically with sequence length. To support longer contexts, models may use structured attention:

- sliding window attention: each layer attends only to a nearby window;
- sparse attention: local and cross-block connections are designed manually;
- periodic full attention: only some layers perform global attention.

Recent models use hybrid patterns. For example, in every four blocks, one layer might use full attention without position encoding, while the other layers use RoPE with sliding window attention. This has two benefits:

1. Most layers process only local windows, keeping system cost manageable.
2. Long-distance information propagates through position-free full attention, reducing pressure on RoPE length extrapolation.

Long-context ability is therefore not just “stretch RoPE.” It requires joint design of attention pattern, position encoding, and system cost.

## 10. Practical default configuration

For a standard dense decoder-only LLM, a reasonable starting point is:

- block: pre-norm Transformer;
- norm: RMSNorm;
- linear: no bias by default;
- position: RoPE on Q/K;
- MLP: SwiGLU;
- MLP ratio: about `8/3 * d_model`;
- attention: MHA for small training experiments; prefer GQA for inference-oriented deployment;
- head dimension: satisfy `d_model = n_heads * d_head`;
- width/depth: use `d_model / n_layers ≈ 100-128` as a reference;
- dropout: usually none or very small for large-scale pretraining;
- weight decay: keep it and tune with the learning-rate schedule;
- stability: monitor gradient norm and loss spikes; consider z-loss, QK norm, extra norm, or logit soft cap.

## Summary

The main lesson is that modern LLM architecture is not the result of one single breakthrough. It is the gradual convergence of many empirical choices. Pre-norm and a clean residual stream make deep networks easier to train. RMSNorm, no-bias layers, and GQA show attention to memory movement and inference cost. SwiGLU, RoPE, and reasonable hyperparameter ratios provide stable and effective defaults. z-loss, QK norm, and related tricks address numerical instability that becomes more important at scale.

If you remember one sentence: training a Transformer is not simply stacking layers and parameters; it is a coordinated set of choices across architecture, hyperparameters, optimization dynamics, and hardware efficiency. Modern LLM “default recipes” matter because they have been validated by many expensive large-scale training runs and help you avoid costly mistakes.


---


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


---


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


---


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


---


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


---


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


---


# CS336 Lecture 9 Tutorial: Scaling Laws (Part 1)

> Adaptation note: This English tutorial is adapted from the Chinese lecture notes. It preserves the original structure, technical terminology, formulas, and code-style notation, while presenting the material in clear tutorial English.

This lecture discusses one of the most important engineering tools in large language model training: scaling laws. Their goal is not to claim that “models will become smarter forever,” but to use small-scale experiments to predict large-scale training outcomes. Before spending enormous compute, scaling laws help answer practical questions: how large should the model be? How much data should be used? How should architecture, optimizer, batch size, and learning rate change with scale? Given fixed FLOPs, how should we allocate model parameters and training tokens most efficiently?

## 1. Why scaling laws are needed

Suppose you have 100,000 H100s and can train a state-of-the-art open-source language model. The systems, data, and architecture are ready, but one expensive problem remains: you cannot tune hyperparameters by repeatedly training giant models. The traditional loop of “train a large model, observe results, tune again” is extremely expensive at frontier scale.

The scaling-laws approach is:

1. Train a set of small models spanning several orders of magnitude in compute, data, or parameter count.
2. Fit a simple functional relationship between model loss and resource investment, usually a power law.
3. Extrapolate that relationship to larger scale to predict large-model performance and choose a training plan.

Thus scaling laws are a scale-aware engineering method. They let us avoid blindly copying existing designs such as LLaMA or GPT, and instead systematically compare candidate architectures, optimizers, data mixtures, and training budgets.

## 2. Basic form: the relationship among loss, data, model size, and compute

Empirically, language-model cross-entropy loss often has a log-log linear relationship with data size, model parameter count, and training compute. That is, if the horizontal axis is the log of resource scale and the vertical axis is the log of excess loss (the part above irreducible loss), the curve is approximately a straight line. This is equivalent to a power law:

```text
L(x) = L_infinity + A * x^(-alpha)
```

Here `x` can be data size, non-embedding parameter count, or compute; `L_infinity` is irreducible loss; and `alpha` is the scaling exponent, indicating how fast loss decreases as resources increase.

This relationship usually has three regions:

- Random-guessing region: the model or data is too small, behavior is unstable, and extrapolation is difficult.
- Power-law region: loss decreases steadily with scale; this is where scaling laws are most useful.
- Saturation region: close to irreducible error; additional resources bring smaller returns.

When running scaling experiments, try to place data points in the power-law region. For example, when studying “data scaling,” the model should be large enough so model capacity is not the first bottleneck. When studying “model scaling,” training tokens must also not limit the model too early.

## 3. Data scaling: why power laws are natural

From a statistical learning perspective, more data reduces estimation error. The simplest example is estimating the mean of a Gaussian distribution: the mean squared error is about `sigma^2 / n`, which becomes a straight line with slope -1 after taking logs.

Real neural networks, however, are not estimating one mean; they learn complex functions in high-dimensional spaces. If the input space is divided into many small regions and we estimate a local average inside each region, higher dimensions require more data per region, and error decreases more slowly. A common result in nonparametric statistics is that the error exponent depends on the intrinsic dimensionality of the task. Therefore data scaling exponents in real tasks are often much smaller than 1: in early experiments, machine translation, speech, and language modeling exponents might be only about 0.1 to 0.3.

This shows that the scaling exponent is not just a fitting parameter; it also reflects the difficulty of task learnability. The smaller the exponent, the slower the gain from adding data.

Engineering uses of data scaling include:

- Comparing data-source quality: if different data mixtures mainly change the curve intercept rather than the slope, small models can be used to filter data.
- Optimizing data mixture: fit scaling curves for different data mixtures and predict which combination is better at large scale.
- Analyzing multi-epoch training: repeating the same tokens has diminishing returns and can often be modeled by correcting the scaling law with an “effective data size.”
- Trading off high-quality repeated data against lower-quality new data: when high-quality data is limited, decide whether to repeat Wikipedia/books or add more low-quality web data.

## 4. Model scaling: using small-scale experiments for architecture and hyperparameter choices

Scaling laws apply not only to data, but also to comparing models and training methods. The classic approach is to train several candidate methods and observe loss curves at multiple compute scales. If two curves have similar slopes and do not cross, their difference can be interpreted as a “constant-factor compute efficiency gap.” For example, Transformers had a clear advantage over LSTMs in Kaplan et al.'s experiments: to reach the same loss target, an LSTM might require many times more compute.

Similar methods can be used for:

- Architecture selection: compare whether Transformer, LSTM, state space model, GLU, MoE, and others still win after scaling up.
- Optimizer selection: Adam and SGD may show a stable compute-efficiency gap.
- Depth/width ratio: many hyperparameters do not have sharp optima but instead have broad “approximately optimal basins.”
- Parameter counting: embedding and non-embedding parameters scale differently; in MoE, total parameters and activated parameters must also be distinguished.

An important reminder: scaling laws are usually stable for next-token cross entropy / log loss, but not necessarily for downstream benchmarks. Perplexity decreases with scale, but this does not guarantee that question answering, in-context learning, reasoning, and other abilities improve according to the same law. Therefore engineering practice often uses loss as the main prediction target, while still requiring downstream evaluation for validation.

## 5. Batch size, learning rate, and scale

As training scale grows, batch size and learning rate cannot simply remain fixed.

Batch size has the concept of a “critical batch size.” At smaller batch sizes, increasing batch is almost equivalent to adding more effective gradient samples and improves parallel efficiency. Beyond a certain point, the marginal benefit drops quickly. This threshold depends on the target loss: the better the model is trained and the lower the target loss, the more precise gradients usually need to be, so larger batches may be tolerable or necessary. In practical large-model training, batch size is often increased gradually during training.

Learning rate also changes with model width. Under standard parameterization, wider models often have smaller optimal learning rates, so one must tune separately at different scales or fit a scaling relationship between optimal learning rate and model width. Another idea is μP (mu-parameterization): by rescaling initialization, learning rates, and forward outputs according to width, learning rates tuned on small models transfer more easily to large models. This reflects an important idea: not only hyperparameter tuning but also parameterization itself can be designed for cross-scale transfer.

## 6. Joint data-model scaling and Chinchilla optimality

So far we have discussed single-variable scaling: changing only data, model size, or compute. But in real training, fixed compute can be allocated to two things: a larger model or more training tokens. Both extremes waste compute: a small model trained on too much data saturates; a giant model that sees too few tokens also fails to learn well.

A joint scaling law tries to fit:

```text
L(N, D) = E + A / N^alpha + B / D^beta
```

Here `N` is the number of model parameters, `D` is the number of training tokens, and `E` is irreducible loss. Training compute is approximately proportional to `N * D` (more precisely often written as about `6ND` FLOPs). Given total compute, we can search along this constraint for the `N` and `D` that minimize loss.

The Chinchilla paper systematically studied this problem and reached the famous conclusion: under training-compute optimality, model parameters and training tokens should grow roughly in proportion; an empirical rule is about 20 training tokens per parameter. In other words, compared with models like GPT-3 that have many parameters but relatively few tokens, a Chinchilla-style choice uses a smaller model and more data, obtaining lower loss under the same training FLOPs.

Chinchilla used three methods:

1. Lower-envelope method: collect training curves for models of different sizes; for each compute value, find the checkpoint with the lowest loss, then fit the optimal parameter count and token count.
2. IsoFLOP analysis: fix several compute budgets and sweep model sizes under each budget; small models train on more tokens and large models on fewer tokens, then find the minimum of each curve.
3. Directly fit a two-dimensional loss surface: train different `N, D` combinations, fit the joint scaling law, and derive the optimal compute allocation.

IsoFLOP analysis is the most intuitive: at the same FLOPs, compare different model sizes horizontally and find the lowest-loss point; then observe how these optimal points change as FLOPs grow. Chinchilla's multiple methods gave similar conclusions. Later reproduction work found that the original curve fitting in the third method had a small issue; after correction, it became closer to the first two results.

## 7. Predicting training outcomes and designing experiments

When using scaling laws in practice, experiments can be designed as follows:

1. Define the target metric: prefer validation cross entropy over unstable benchmark scores.
2. Choose the scaling axis: data size, non-embedding parameter count, total FLOPs, or joint `N` and `D`.
3. Cover multiple orders of magnitude: small experiments must span a wide enough range, otherwise extrapolation is unreliable.
4. Control confounders: when studying data, the model must be large enough; when studying model size, data and training schedule must be reasonable; when comparing architectures, keep training budget, tokenizer, and data as consistent as possible.
5. Fit log-log curves: check whether points lie in the power-law region and whether there is curvature, saturation, or random-region abnormality.
6. Extrapolate and choose a plan: predict large-scale loss, optimal model size, token count, batch size, learning rate, or data mixture.
7. Run medium-scale validation: before the true large run, validate extrapolation with one or two points closer to the target scale.

## 8. Common traps and practical advice

First, do not treat all parameters as identical. Embedding parameters, dense-layer parameters, total MoE parameters, and activated MoE parameters contribute differently to training loss and inference cost. If they are mixed together directly in a fit, curves may bend or lead to wrong conclusions.

Second, do not extrapolate too far from models that are too small. The random-guessing region, poorly tuned learning rates, too-large batch sizes, or insufficient data can all make small-model points deviate from the true power law. Small experiments must first confirm stable training and comparable loss curves.

Third, do not look only at the final checkpoint. Schedules such as cosine learning-rate schedules require the full cooldown; an early truncated intermediate checkpoint is not equivalent to retraining a shorter-schedule model. Part of the difference between Kaplan and Chinchilla estimates comes from these training-curve handling details.

Fourth, distinguish “constant-factor advantage” from “slope advantage.” If a new architecture only shifts the curve downward, it may only be a fixed multiple more compute-efficient. If the slope is steeper, its advantage grows with scale; that is the signal truly worth betting on at frontier scale.

Fifth, scaling predictions should always include uncertainty. A scaling law is not a physical law; it is an empirical model under a specific dataset, codebase, optimizer, and training regime. The farther the extrapolation, the more conservative one should be. It is best to reserve a medium-scale validation point specifically to test whether the fitted curve still predicts real loss. If the validation point deviates clearly, recheck data quality, learning rate, warmup, weight decay, tokenizer, deduplication, and evaluation-set leakage.

## 9. A modern view: training optimality is not deployment optimality

Chinchilla solves the question: “Given training FLOPs, how do we obtain the lowest training/validation loss?” But today models are products, and inference cost is equally important. A larger model may be training-optimal with fewer training tokens, but its per-token inference cost is higher during deployment. Therefore many modern models use far more than 20 tokens per parameter: they prefer spending more one-time pretraining cost to make the model smaller and train it more densely, in exchange for lower long-term inference cost.

Thus conclusions from scaling laws must be understood together with the objective function. If the objective is training-FLOPs optimality, the Chinchilla ratio is an important baseline. If the objective is total cost (training plus massive inference), one may choose more tokens and a smaller model.

## Summary

Scaling laws are prediction tools for large-model engineering. They fit power-law relationships between loss and data, model parameters, and compute using small-scale training, helping predict large-run outcomes, compare architectures and optimizers, choose hyperparameters, and trade off model size against training tokens under a fixed budget. Chinchilla-style compute optimality shows that training-optimal models usually require parameter count and token count to grow in a coordinated way, rather than simply increasing parameters. Reliable scaling experiments must cover multiple orders of magnitude, control confounders, prioritize stable log loss, and confirm that the curve is truly in the power-law region before extrapolation.


---


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


---


# CS336 Lecture 11 Tutorial: Scaling Laws (Part 2)

> Adaptation note: This English tutorial is adapted from the Chinese lecture notes. It preserves the original structure, technical terminology, formulas, and code-style notation, while presenting the material in clear tutorial English.

The previous lecture introduced the basic idea of scaling laws: fit the relationship among loss, parameter count, data size, and compute using small-scale experiments, then extrapolate to large models. This lecture is closer to the real workflow of training large models: how do teams in public papers actually use scaling laws to choose learning rate, batch size, model size, training tokens, and architecture? When are these fits reliable, and when do they merely look like straight lines?

The core message can be summarized in one sentence: scaling laws are not a single formula, but an experimental methodology for reducing the risk of large-scale training. They usually involve small-model proxy experiments, hyperparameter transfer, IsoFLOP analysis, WSD learning-rate schedules, μP parameterization, and conservative validation of extrapolated results.

## 1. Scaling problems in real training

Before training a frontier language model, teams must answer several expensive questions:

1. Given training FLOPs, how large should the model be and how many tokens should it see?
2. How should batch size change with scale?
3. Does the learning rate need to decrease as the model gets larger?
4. Can hyperparameters tuned on small models transfer to large models?
5. If a new architecture looks good at small scale, will it still be worthwhile after scaling up?

These questions cannot be answered by repeatedly trying 70B or 400B models. A common strategy in public frontier experiments is to first train many proxy models at the 10M, 100M, or 1B scale, fit trends, and then perform only a few validation runs or a final training run at the target scale. Cerebras-GPT, MiniCPM, DeepSeek LLM, Llama 3, Hunyuan, and MiniMax-01 all demonstrate this workflow in different ways.

## 2. Cerebras-GPT: using μP to make hyperparameters more stable

Cerebras-GPT trained models from about 0.1B to 13B parameters and focused on validating **μP (mu-parameterization, maximal update parameterization)**. Under standard parameterization, as the model gets wider, the optimal learning rate often moves to smaller values. If the learning rate tuned on a small model is directly copied to a large model, training may become unstable or end at a higher loss. The goal of μP is to rescale initialization and per-layer learning rates so that the same “base learning rate” remains close to optimal across widths.

In practice, μP roughly does the following: initialize non-embedding weights with width-dependent scaling; when using Adam/AdamW, also scale learning rates for different layers according to fan-in or width rather than sharing a single global learning rate across all parameters. The benefit is that one can run dense grid searches on small models to find learning rate, initialization, width/depth ratio, and related settings; when the model is scaled up, these settings are more likely to remain valid.

Cerebras-GPT experiments show that μP curves better match the expected scaling law, while standard parameterization is more likely to oscillate or deviate across scales. This does not mean large models cannot be trained without μP. Rather, μP turns the problem of “retune the learning rate at every scale” into “tune it once at small scale and transfer it.”

## 3. MiniCPM: small models, long training, and WSD schedules

MiniCPM aims to train very strong small models, such as 1B to 2B models, by training them thoroughly on large amounts of data. Its scaling experiments contain three important tools.

First, it still uses μP to stabilize learning-rate transfer. MiniCPM searches hyperparameters on small proxy models with tens of millions of parameters, then scales to hundreds of millions or 1B parameters. In the experiments, the optimal learning rates for different model sizes largely fall in the same place, supporting the practical value of μP.

Second, it fits the **critical batch size**. The critical batch size can be understood as the point where continuing to increase batch size begins to give diminishing returns. Larger models and lower target losses can usually use larger batches. MiniCPM fits a log-log relationship between target loss and optimal batch size using training curves at different batch sizes, then extrapolates it to the target training scale.

Third, and very important for this lecture, it uses a **WSD (warmup-stable-decay)** learning-rate schedule to reduce the experimental cost of Chinchilla-style data/model scaling.

The problem with a standard cosine schedule is that if the total number of training tokens differs, the entire cosine curve differs. A checkpoint at 100B tokens inside a model trained to 1T tokens is not equivalent to “training from scratch for 100B tokens and then doing a full cooldown.” Therefore, one cannot simply treat intermediate checkpoints from a long run as experimental points for shorter data budgets.

WSD splits the learning rate into three phases:

```text
warmup -> stable plateau -> decay/cooldown
```

Its benefit is that the stable phase can be reused. To estimate the final loss for a shorter data budget, one can branch from an intermediate checkpoint and attach a separate decay phase. Thus, one long training run plus several short cooldowns can approximate multiple data-budget endpoints, greatly reducing repeated training cost.

MiniCPM uses WSD for Chinchilla-style analysis and estimates the optimal model/data ratio in two ways: one takes the lower envelope of training curves, and the other directly fits a two-dimensional loss surface:

```text
L(N, D) = E + A / N^alpha + B / D^beta
```

It obtains a high token/parameter ratio, about 192:1. This number should not necessarily be treated as a universal law. The more important lesson is that Chinchilla's 20:1 is not an unbreakable rule. Modern high-quality data, improved architectures, and stronger optimization may make the “more tokens, smaller model” strategy more attractive, especially when inference cost is included in the objective.

## 4. DeepSeek LLM: directly fitting learning rate, batch size, and IsoFLOP

The public DeepSeek LLM paper is valuable because it shows another, more direct scaling method: it does not rely on μP, but explicitly fits how batch size and learning rate change with scale.

DeepSeek first runs grid searches over batch size and learning rate on small models to find the optimum or approximately optimal region at each scale. It then puts these optimal batch sizes, learning rates, and training FLOPs on log-log plots, fits trends, and extrapolates them to 7B and 67B models.

There is an important practical judgment here: batch-size scaling is often relatively clean, while learning-rate scaling is noisier and more questionable. Sometimes the learning-rate curve can also be explained by a horizontal line. Therefore, fitting learning rate is mainly used to get the right order of magnitude rather than an exact formula. Large-model training often has a broad “usable basin”; as long as the learning rate is not wrong by an order of magnitude, training may still be viable.

DeepSeek also performs Chinchilla / IsoFLOP analysis. The IsoFLOP procedure is: fix multiple compute budgets; under each budget, train models of different sizes, where smaller models see more tokens and larger models see fewer tokens; find the lowest-loss point on each fixed-FLOPs curve; then fit how the optimal parameter count and optimal token count grow with FLOPs. Compared with learning-rate fitting, these IsoFLOP curves are usually more stable and more trustworthy.

DeepSeek also uses WSD-style cooldowns to reduce repeated training and ultimately extrapolates from about `10^20` FLOPs to about `10^24` FLOPs, accurately predicting the loss of 7B and 67B models. This shows that when the training regime, data, and architecture are kept consistent, loss-scaling extrapolation can indeed serve as a risk-control tool before large training runs.

These successful extrapolations also explain why many teams are willing to spend a large “small-experiment budget” before official training. These experiments may already be expensive, but the cost is worthwhile if they prevent one failed target-scale run. More realistically, a scaling law does not need to predict every final benchmark exactly. If it can catch a wrong learning-rate order of magnitude, an unsuitable batch size, an obviously bad token/parameter ratio, or an architecture that loses its advantage when scaled up, it has already saved a great deal of compute.

## 5. Trends in recent models: ratios change, methods are reused

The Llama 3, Hunyuan, and MiniMax-01 papers do not provide as many scaling details as MiniCPM or DeepSeek, but they still show several trends.

Llama 3 reruns IsoFLOP / Chinchilla analysis and obtains an optimal token/parameter ratio of about 40:1, higher than Chinchilla's 20:1. It also attempts to map training loss or negative log likelihood to downstream benchmark accuracy, for example by fitting a sigmoid relationship from loss to performance on MMLU-like tasks. The motivation is clear: teams ultimately care not about log loss itself, but about downstream capability. However, benchmark scores are noisier and saturate more easily, so in practice teams usually predict loss first and use it to help predict benchmarks.

Hunyuan's analysis obtains an even higher active-parameter token ratio, for example about 96:1. Here it is important to distinguish “total parameters” from “activated parameters” in MoE or sparse models; the ratios cannot be directly mixed with dense-model ratios.

MiniMax-01 uses scaling laws for architecture selection. It compares the loss-compute curves of softmax attention, linear attention, and hybrid attention, observing whether their lower envelopes and optimal model/token trends are close. If linear attention does not show a clear degradation in its scaling curve at the same compute, that supports using it in long-context models. This shows that scaling laws are not only for choosing size; they can also judge whether a new architecture is worth scaling.

## 6. The intuition of μP: controlling the scale of activations and updates

The mathematical intuition behind μP can be simplified into two conditions.

First, as the model gets wider, each coordinate of the activation should not explode or vanish. If one layer is a matrix multiplication `h_l = W_l h_{l-1}`, then to keep the output activation scale stable, a common initialization scales with the square root of fan-in, roughly:

```text
W_l ~ 1 / sqrt(fan_in)
```

This is consistent with the intuition behind Kaiming/Xavier initialization.

Second, after one gradient update, the change in activation should also not explode or vanish with width. This condition constrains how the learning rate should change with layer width. For SGD, one can derive a ratio similar to fan-out/fan-in. For Adam/AdamW, because adaptive normalization changes gradient scale, common μP rules scale learning rates by fan-in or width.

Therefore, μP is not only about initialization. It is about “initialization + per-layer learning rates + certain forward scalings” jointly ensuring that update scales remain stable. For Transformers, this may also involve scaling attention logits. Some μP implementations use `1/d` instead of the traditional `1/sqrt(d)` attention scaling in order to satisfy update-stability requirements.

Empirical studies show that μP is robust to many changes: switching among ReLU, SwiGLU, and Squared ReLU, or changing batch size within a certain range, still allows learning-rate transfer. But it is not universal. Learnable norm gains, strong weight decay, and sign-gradient-style optimizers such as Lion can break μP's transfer assumptions. In other words, μP is an engineering tool designed for particular optimizers and parameterizations; it is not a theorem that automatically holds for every training recipe.

## 7. A practical workflow for experimental fitting

A relatively complete scaling experiment can follow these steps:

1. Fix the data, tokenizer, architecture family, optimizer, and training code to reduce confounding variables.
2. Choose several small to medium model sizes covering at least a few orders of magnitude in parameters or FLOPs.
3. Run small-scale grid searches for key hyperparameters such as learning rate, batch size, warmup, and weight decay.
4. If using μP, tune on small models and verify that the learning rate transfers across widths; if not using μP, explicitly fit how learning rate and batch size change with scale.
5. Use WSD or an equivalent method to collect true cooldown losses at different token endpoints, avoiding misuse of intermediate cosine checkpoints.
6. Perform IsoFLOP analysis or two-dimensional `L(N,D)` fitting to estimate the optimal model size and training tokens for a given compute budget.
7. Validate on a medium scale that is smaller than the final scale but clearly larger than the fitting points, checking whether extrapolation is accurate.
8. During final training, keep monitoring: if loss deviates from prediction, investigate data, optimizer, batch, learning-rate schedule, and implementation bugs as early as possible.

## 8. Limitations and common misconceptions

First, a log-log straight line is not truth. A scaling law is an empirical fit that depends on the data distribution, model family, optimizer, training schedule, and evaluation set. The farther the extrapolation, the larger the uncertainty.

Second, training loss is the most stable signal, but downstream capabilities are not necessarily stable. Lower perplexity is usually a good sign, but mathematical reasoning, tool use, long-context ability, instruction following, and similar capabilities may have threshold effects or evaluation noise.

Third, there is no universal token/parameter ratio. 20:1, 40:1, 96:1, and 192:1 all appear in different papers. The differences come from data quality, architecture, whether the model is MoE, training objective, and deployment cost. Chinchilla gives a training-FLOPs-optimal baseline, not the product-total-cost-optimal answer.

Fourth, learning-rate scaling is more fragile than loss scaling. Batch size and IsoFLOP relationships are often easier to fit; learning-rate curves may be flat and noisy, so they should only be used as order-of-magnitude references.

Fifth, one must not treat an intermediate checkpoint from a long training run as the endpoint of a short training run. Without cooldown, checkpoint loss is often too high and systematically contaminates data-scaling fits. This is precisely the problem WSD solves.

## Summary

This lecture showed how scaling laws are used in real model training. Cerebras-GPT and MiniCPM use μP to stabilize hyperparameter transfer; MiniCPM and DeepSeek use WSD to reduce the cost of Chinchilla analysis; DeepSeek directly fits batch size, learning rate, and IsoFLOP curves and successfully predicts large-model loss; Llama 3, Hunyuan, and MiniMax-01 show that modern teams still reuse IsoFLOP analysis, though optimal token/parameter ratios and architecture questions change with the target.

Reliable scaling work is not “draw a line and believe it.” It is a systematic use of small-scale experiments to eliminate risk: learning rate, batch size, model size, data volume, and architecture choices should all have evidence before scaling up. At the same time, the limits of extrapolation must be acknowledged and predictions should be calibrated with medium-scale validation points. Its value is to turn large-model training from a gamble into a manageable engineering decision: first use cheap experiments to narrow the search space, then spend expensive training compute on the option with the strongest evidence.


---


# CS336 Lecture 12 Tutorial: LLM Evaluation

> Adaptation note: This English tutorial is adapted from the Chinese lecture notes. It preserves the original structure, technical terminology, formulas, and code-style notation, while presenting the material in clear tutorial English.

## 1. Why evaluation is not simple

On the surface, LLM evaluation looks like a script: choose a model, prepare prompts, call the model, collect outputs, compute metrics, and average the results. The real difficulty is deciding what question you want the number to answer.

Different roles care about very different evaluations:

- Users or companies: I need to choose among Claude, Gemini, GPT, and open-source models. Which is best for my business?
- Researchers: Does the model really have stronger general capability? Is AI progressing in a scientifically meaningful sense?
- Policymakers: What benefits and risks does the model bring? Is it safe enough?
- Model developers: Did a training, data, or alignment method actually improve the model?

Therefore there is no single “correct” evaluation. A leaderboard score is meaningful only after you understand its input distribution, calling protocol, scoring rules, and intended use. Evaluation also shapes model development: once a benchmark becomes a target, developers optimize for it. When a metric is over-optimized, it may lose its original meaning. This is Goodhart's law in LLM evaluation.

## 2. Evaluation framework: input, invocation, output, and interpretation

A reliable evaluation can be decomposed into four questions.

First, where do the inputs come from? Do the prompts cover real use cases? Do they include difficult examples, long-tail examples, and edge cases? If the task is multi-turn chat, later inputs depend on the model's earlier answers, so a static test set may not simulate real conversation. Red-team testing often also needs to adaptively generate attack prompts based on model behavior; otherwise rare failures are hard to find.

Second, how is the model called? Zero-shot, few-shot, chain-of-thought, tool use, RAG, and agent scaffolding can all significantly affect results. Early base models often needed few-shot examples to specify the format. Modern instruction-tuned models can usually follow zero-shot instructions such as “output only A/B/C/D.” Prompt order, formatting, and example choice all introduce variance.

Third, how is the output judged? Multiple-choice tasks can use accuracy, code tasks can use pass@1 or pass@k, and open-ended generation may need human preferences or LLM-as-a-judge. Cost must also be considered: a model with a higher score but much higher price, latency, or inference token count may not be the better system. Different errors also have different costs; medical, legal, and safety-critical settings cannot be evaluated only by average accuracy.

Fourth, how should the score be interpreted? Is 91% good or bad? Is it enough for deployment? Does it show that the model learned a capability, or merely that it has seen similar questions? Is the evaluated object a base model, a chat model, a complete agent system, or a training method? These questions must be made explicit in advance.

## 3. Perplexity: still an important foundational metric

Perplexity measures how much probability a model assigns to a dataset. A language model is fundamentally a probability distribution over token sequences; lower perplexity means the model can better predict the tokens in that dataset. Traditional language-modeling research trained and tested on fixed datasets such as Penn Treebank, WikiText, and One Billion Word, with the goal of reducing test perplexity.

After GPT-2, the paradigm changed: models are pretrained on large-scale web text and then directly transferred to many downstream tasks and perplexity benchmarks. Evaluation then becomes closer to out-of-distribution generalization: the model was not specifically trained on Penn Treebank, but may perform well because its training corpus is broad enough.

Advantages of perplexity:

- Smoothness: it uses the log probability of every token, giving more fine-grained information than binary right/wrong accuracy.
- Suitability for scaling laws: as model size, data, and compute change, loss curves are easier to fit.
- Comprehensive coverage: it cares about every token in the dataset, not only the final answer.
- Less vulnerable to answer-format gaming, as long as train/test separation is reliable.

But perplexity also has limitations. First, it is not always strongly correlated with downstream task performance; over short periods or on specific tasks, the relationship can be messy. Second, if a leaderboard requires models to provide probabilities, it must trust that the provider's logits or probability API is correctly normalized; implementation bugs can otherwise create falsely low perplexity. Finally, the perplexity-maximization view says that “matching the true distribution solves everything,” but this may not be the most efficient path, because many tokens are not important for practical tasks.

Some tasks are close to perplexity, such as LAMBADA's missing-word prediction and HellaSwag's multiple-choice continuation task: the model compares the likelihood of candidate continuations. But these tasks can also saturate and may suffer from approximate contamination from original web sources.

## 4. Multiple-choice benchmarks: MMLU, MMLU-Pro, GPQA, HLE

MMLU is one of the classic LLM knowledge benchmarks, containing multiple-choice questions across 57 subjects. It appeared after GPT-3, when making a base model answer questions across many subjects via few-shot prompting was still a novel setup. The name MMLU contains “language understanding,” but it is more like a knowledge exam: many questions test specific disciplinary facts rather than pure language understanding.

MMLU scores must be interpreted together with the training and evaluation setup. If a base model is not specifically optimized for MMLU but scores highly on multi-subject multiple-choice questions, this suggests strong general knowledge and transfer ability. But if developers collect similar questions, tune prompts, use chain-of-thought, and ensemble results specifically for MMLU, then a high score does not necessarily represent the same degree of general capability.

MMLU-Pro tries to address MMLU saturation: it removes noisy and easy questions, increases the number of options from 4 to 10, and more often uses chain-of-thought. This lowers frontier-model accuracy and gives the benchmark discriminative power again.

GPQA emphasizes expert-level difficult questions. The questions are written and verified by PhDs or domain experts, with the goal of being “Google-proof”: non-experts should have difficulty answering them even with search. Early GPT-4 did not perform very well, but newer models have improved substantially. This shows that “hard for humans to search” does not mean “hard for LLMs forever.” During evaluation, one must also confirm whether the model is allowed to use the internet, because black-box APIs may secretly call search tools.

Humanity's Last Exam (HLE) further collects extremely difficult, multimodal, multiple-choice or short-answer questions. It uses prizes and attribution to attract contributors and filters out questions that frontier models find too easy. Its advantage is difficulty; its disadvantage is clear distributional bias. People willing to write questions are often familiar with LLMs and may deliberately design “model-hard” questions, so HLE does not represent ordinary user needs.

## 5. Open-ended and instruction-following evaluation

The core ability of modern chat models is not merely taking exams, but completing open-ended tasks from natural-language instructions. Open-ended outputs have no unique ground truth, making evaluation harder.

Chatbot Arena asks users to submit real prompts, shows answers from two anonymous models, asks the user to choose the better answer, and computes Elo rankings from pairwise preferences. Its advantages are that it is dynamic, close to real usage, and can include new models. Its disadvantages are that the user distribution is uncontrolled, prompts may be for entertainment or testing, and the more important the leaderboard becomes, the easier it is to optimize for or manipulate. Recent debates around Arena also show that evaluation protocol, submission access, model versions, and data transparency all matter.

IFEval specifically evaluates “constraint following” in instruction following: for example, the output must be shorter than a certain number of words, must include or exclude certain words, or must use a specific format. Its advantage is that scripts can automatically verify the constraints. Its disadvantage is that it checks only formal constraints, not semantic quality. A 10-word story may satisfy the length requirement without being good.

AlpacaEval uses LLM-as-a-judge to compare a model answer against a reference model answer and compute win rate. It is automatic, fast, and reproducible, but the judge model has biases. For example, early small models could fool the GPT-4 judge by producing longer answers; length correction was added later. Datasets such as WildBench sample from real human-model conversations and ask judges to evaluate with a checklist, often also reporting correlation with Chatbot Arena.

## 6. Agent benchmarks: evaluating the model or the system?

Many tasks require tool use and multi-step iteration. In this case, the evaluation target is no longer just an LM, but a “model + agent scaffolding” system.

SWE-bench gives an agent a GitHub issue and codebase, asks it to modify the code and submit a patch, and finally checks whether unit tests pass. Cybench puts the agent in a CTF cybersecurity environment where it must execute commands, explore servers, and obtain flags. MLE-bench simulates Kaggle: the agent must read the task, write training code, tune hyperparameters, and submit results. These benchmarks are closer to real workflows, but their scores are strongly affected by tools, context management, retry strategy, time budget, and cost.

Therefore, when reporting agent scores, one must state: Is internet access allowed? How many steps can it run? Are human hints allowed? Are there hidden tests? How many dollars and how much time did it spend? If a system obtains a high score through massive sampling and expensive inference, it is not the same capability as a single low-cost answer.

## 7. Contamination: training-set contamination and evaluation validity

Modern models are trained on large-scale internet data, and developers usually do not release the full corpus, so train/test overlap is almost impossible to rule out completely. Contamination may be exact repetition, near duplication, paraphrase, translation, leaked solutions, or leaked answers. Simple n-gram deduplication can find some problems, but cannot detect cross-lingual or semantically equivalent versions.

There are three broad responses:

- Data decontamination: check document, paragraph, or n-gram overlap between the test set and training corpus, conservatively removing suspicious samples.
- Behavioral detection: infer whether a model has seen data by observing abnormal preferences for option order, question order, or rare text.
- Community norms: papers and model cards should report decontamination methods, whether test-set leakage was checked, confidence intervals, and standard errors.

Contamination affects not only multiple-choice questions but also HellaSwag, WikiHow-derived tasks, math problems, and code tasks. Benchmark data itself may also contain annotation errors. When model scores are very high, a substantial fraction of remaining errors may come from question noise rather than insufficient model capability.

## 8. Human evaluation, real use cases, and safety evaluation

Human evaluation is often used for open-ended tasks, but it must specify who the reviewers are, whether they are experts, what the scoring rubric is, whether evaluation is blind, and whether answer length and style are controlled. Preferences of ordinary internet users, judgments of domain experts, and product user satisfaction are not the same thing.

Real-use evaluation is harder than exams and also more important. Users may be “asking questions” because they do not know the answer and need help, or “testing the model” because they already know the answer. Standardized exams mostly belong to the latter category, while business value often comes from the former. Work from Anthropic and others clusters real conversations to analyze what people actually use models for, such as coding, writing, learning, and office work. In medicine, MedHELM asks clinicians to propose real tasks such as clinical-note summarization, treatment planning, and patient communication. But real data often involves privacy, creating tension between public reproducibility and realism.

Safety evaluation also cannot look only at “refusal rate.” HarmBench, AIRBench, and related benchmarks test whether a model complies with harmful requests or build risk categories from laws and company policies. But a model that refuses every question is of course “safe” and also useless. Therefore safety must be evaluated together with capability. It is also important to distinguish **capability** from **propensity**: whether the model knows how to do something dangerous is capability; whether it is willing to output it is propensity. Closed-source APIs focus more on propensity and jailbreak defenses; open-weight models also require attention to capability, because safety layers may be removed by fine-tuning.

## 9. Checklist for reliable evaluation practice

1. First state the purpose of evaluation: model selection, research, product monitoring, safety review, or training feedback.
2. Specify the evaluation target: base model, chat model, agent system, or training method.
3. Fix and disclose the calling protocol: prompt, few-shot examples, temperature, max tokens, tool permissions, and retry count.
4. Report quality, cost, latency, and variance together, not only average accuracy.
5. For multiple-choice tasks, check option-order bias; for open-ended tasks, check length bias and judge bias.
6. Perform contamination checks and report the method; interpret high-risk benchmarks conservatively.
7. Inspect sampled predictions; do not look only at leaderboard numbers.
8. For real deployment scenarios, build private, updated eval sets close to the user distribution.
9. Pair safety evaluation with usefulness evaluation to avoid artificially high “refuse everything” scores.
10. Remember that benchmarks are tools, not truth. Once a benchmark becomes a target, it will be optimized, saturated, and possibly distorted.

In summary, the core of LLM evaluation is not “running a score,” but translating a real-world question into an executable, interpretable, and reproducible measurement process. Good evaluation needs both the comparability of standardized benchmarks and the representativeness of real use cases. It must care about capability, but also about cost, safety, contamination, and data quality. Only by understanding the rules behind the number can we really know where a model is good, where it is weak, and whether it fits our goal.


---


# Stanford CS336 Lecture 13 Tutorial: Pretraining Data (Data 1)

> Adaptation note: This English tutorial is adapted from the Chinese lecture notes. It preserves the original structure, technical terminology, formulas, and code-style notation, while presenting the material in clear tutorial English.

This lecture starts from a central point: in modern language models, data often determines final capability more than model architecture does. Transformer architectures, optimizers, and parallel training techniques are now relatively public, while papers on top models usually describe training data only vaguely, for example as “from multiple data sources, up to a certain year.” This secrecy has commercial reasons, but also copyright and legal-risk reasons. For practitioners, the truly difficult question is not “how do we train once we have data,” but “which data is worth training on, and how do we turn the raw Internet into trainable corpora?”

## 1. Training stages and the role of data

Large-model training can roughly be divided into three stages:

- **Pretraining**: uses massive, relatively raw text, mainly from the Web, code, books, papers, encyclopedias, and similar sources. The goal is for the model to learn language, knowledge, and general patterns.
- **Mid-training**: after pretraining, uses smaller but higher-quality and more targeted data to strengthen capabilities such as mathematics, code, long context, and multilinguality.
- **Post-training**: includes instruction tuning, chat data, RLHF/RLAIF, and related methods, making the model more assistant-like: able to follow instructions, hold conversations, and satisfy safety requirements.

Terminologically, a **base model** usually refers to a model after pretraining and/or mid-training; an **instruct/chat model** is a model that has undergone post-training and is suitable for interaction. In practice, the boundaries among the three stages are blurry. For example, Stack Exchange question-answer data can be used during pretraining, but it also naturally resembles instruction data; modern data pipelines may also introduce model-filtered or model-rewritten data already during pretraining.

## 2. Why “Internet data” is not a simple concept

A common statement is that “large models are trained on Internet data,” but this is far too crude. Real data sources usually pass through three layers of transformation:

1. **Live service**: Wikipedia, GitHub, Reddit, Stack Overflow, news sites, and so on.
2. **Raw dump / crawl**: Common Crawl, Wikipedia dumps, GitHub Archive, and similar snapshots.
3. **Trainable dataset**: tokens after text extraction, language identification, cleaning, filtering, deduplication, sampling, and mixing.

Therefore, when someone says “we trained on GitHub / Common Crawl / Reddit,” you must ask: which snapshot was used? How was text extracted? How were licenses handled? Was deduplication performed? What filtering rules were used? Which fields and metadata were retained? These decisions significantly affect model capability.

## 3. Early data: Books and Wikipedia

The main data used by BERT was **BooksCorpus** and **Wikipedia**. BooksCorpus came from free self-published books on Smashwords and was later taken down because of Terms-of-Service issues. It illustrates the importance of book data: books have long-form structure, narrative coherence, and long-range dependencies, making them suitable for training models to understand longer contexts.

Wikipedia has long been regarded as a representative source of “high-quality text.” It has explicit editing norms: verifiability, cited sources, no original research, relatively few personal opinions, and topic selection through notability. But this also means Wikipedia does not cover all valuable content: personal experience, recipes, forum discussions, niche knowledge, and colloquial expression may all be missing.

Wikipedia also raises a safety issue: **data poisoning**. If an attacker can briefly insert malicious content before a data snapshot is produced, the content may enter the training set even if it is later reverted. More broadly, training data on the Internet is jointly shaped by many people with different motivations. Model behavior may be affected by that data, while the training organization may find it hard to fully audit.

## 4. WebText: using link signals to select Web pages

GPT-2’s WebText dataset demonstrated an important idea: do not crawl Web pages randomly; instead, use link and voting signals from human communities. OpenAI collected Web pages linked from Reddit posts whose karma exceeded a threshold, obtaining about 8 million pages and 40 GB of text. The intuition is that links shared by users and endorsed by the community are, on average, higher quality than ordinary Web pages.

WebText was not released, and the community later produced OpenWebText as a reproduction. The key idea in this family of methods is **link-based filtering**: use outgoing links from high-quality communities, encyclopedia citations, or manually curated pages as quality signals. Later LLaMA data used a similar idea: train a classifier to determine whether a Web page resembles pages cited by Wikipedia.

## 5. Common Crawl: the largest but very noisy public Web source

**Common Crawl** is the large-scale Web source most commonly used by academic and open-source communities. It has crawled the Web periodically since 2007, with each crawl containing billions of pages. The crawler starts from many seed URLs and maintains a frontier queue, roughly performing breadth-first search over the Web, while also handling engineering issues such as robots.txt, server load, duplicate URLs, and dynamic pages.

Common Crawl provides two important formats:

- **WARC**: raw HTTP responses, usually containing HTML but sometimes containing other resources.
- **WET**: plain text converted from HTML; this is a lossy conversion.

HTML-to-text conversion may look low-level, but it greatly affects training quality. Using Common Crawl’s own WET files, Trafilatura, jusText, or other tools produces different text, which in turn affects model evaluation. Modern data engineering often re-extracts article text from WARC instead of relying directly on WET.

Common Crawl is not “the whole Internet.” Its coverage is sparse, biased toward text, follows or at least considers robots.txt, and does not guarantee inclusion of all pages. At the same time, it contains large amounts of spam, advertising, templates, duplicates, low-quality text, and offensive content. Common Crawl is therefore more like raw material than a dataset that can be directly used for training.

## 6. Cleaning, filtering, and deduplication

Common steps from raw Web pages to training tokens include:

### Language Identification

Use fastText or another classifier to identify document language, keeping only target languages or sampling according to a multilingual mixture. Many early studies focused on English, but Common Crawl itself contains multilingual data.

### Rule-based Filtering

C4, Gopher/MassiveText, RefinedWeb, FineWeb, and others use many hand-written rules, such as keeping lines that end with punctuation, removing pages with too few sentences, filtering profanity, requiring a certain fraction of words to contain letters, removing boilerplate, and filtering likely code or templates. Rule-based methods are transparent, cheap, and interpretable, but they can leave behind well-structured junk text and may mistakenly remove dialects, minority-group text, or non-standard writing.

### Model-based Filtering

CCNet trained an n-gram model on Wikipedia and kept documents that “look like Wikipedia.” GPT-3 trained a quality classifier using WebText, Wikipedia, and books as positive examples, then searched for similar content in Common Crawl. DCLM went further: it used instruction-like data such as OpenHermes and ELI5 as positives, and used a fastText classifier to filter a pool of 240T tokens down to about 3.8T tokens.

Model-based filtering can significantly improve benchmarks, but the risk is that it narrows “quality” to the positive-example distribution. If the positives are biased toward encyclopedic text, English, or mainstream writing, the model may lose diversity. A recent trend is to accept, and even strengthen, model involvement in data selection again because the gains are so large.

### Deduplication

The Web contains enormous duplication: mirror sites, reposts, templates, dynamic URLs, code forks, and documentation copies all create duplicates. Deduplication is divided into exact deduplication and **fuzzy deduplication**. Deduplication reduces wasted training, lowers the probability that the model memorizes specific text, and prevents some sources from being overweighted.

### Harmful-content and privacy filtering

Many pipelines add toxicity classifiers, safe search, PII anonymization, and similar steps. But these filters are themselves imperfect: if too strong, they damage the real-world distribution; if too weak, they introduce safety, privacy, and legal problems.

## 7. A genealogy of typical pretraining datasets

- **C4 (Colossal Clean Crawled Corpus)**: Google/T5’s cleaned version of Common Crawl, relying mainly on rule-based filtering and keeping only English natural-language text.
- **The Pile**: a mixture of 22 high-quality data sources built by the EleutherAI community, including Common Crawl, OpenWebText, Stack Exchange, Wikipedia, arXiv, PubMed, GitHub, Books3, and others. It represents the “manually selected domains” approach.
- **MassiveText / Gopher**: DeepMind’s data mixture, including MassiveWeb, C4, books, news, GitHub, and Wikipedia, with rule-based and safety filtering.
- **LLaMA data**: Common Crawl + C4 + GitHub + Wikipedia + Project Gutenberg + Books3 + arXiv + Stack Exchange, totaling about 1.2T tokens. It was not released, but RedPajama produced a reproduction.
- **RefinedWeb / FineWeb**: argues that if Web filtering is good enough, strong data can be obtained from the Web alone. FineWeb is Hugging Face’s lightly filtered large-scale Common Crawl dataset and can serve as a base for further selection.
- **DCLM Baseline**: turns the full Common Crawl pool into a competition-style data benchmark and uses a strong quality classifier for aggressive filtering. It has become a common data source for recent open-source models.
- **Nemotron-CC**: NVIDIA’s extension of the DCLM idea. It uses large models to score “educational value,” distills this into faster models, combines multiple filters, and also tries to use LLMs to rewrite low-quality data or convert high-quality documents into task form.

These datasets reveal two tensions. The first is the trade-off between quality and scale: stronger filtering gives higher quality but fewer tokens. The second is the trade-off between quality and diversity: the more data resembles high-quality positives, the more likely it is to lose long-tail knowledge, colloquial language, and non-mainstream text.

## 8. The special value of code, question answering, books, and papers

Different sources provide different capabilities:

- **GitHub / The Stack**: mainly trains coding ability and may also improve structured reasoning. Processing requires license detection, duplicate removal, filtering generated files, distinguishing code from documentation, and deciding whether to use issues and commit history.
- **Stack Exchange / Stack Overflow**: naturally has a QA format, with questions, answers, comments, votes, and other metadata. It is suitable for selecting high-quality explanations and also blurs the boundary between pretraining and instruction training.
- **Project Gutenberg / PG19**: public-domain books with clear copyright status, suitable for long-context training, but with an older language style.
- **arXiv / PubMed / Semantic Scholar**: academic papers provide dense knowledge, mathematics, and technical expression, but format extraction, formulas, citations, and copyright all need handling.
- **Reddit / ELI5**: closer to user questions and plain-language explanations; can be used as positive examples for quality classifiers or as instruction-like corpora.

## 9. Copyright and data availability

Most original expression on the Internet is protected by copyright by default, even if a Web page does not display a copyright notice. There are roughly two routes for use: obtain a license, or claim **fair use**. Fair use considers factors such as whether the use is transformative, the nature of the work, the proportion used, and the effect on the original market. For large-model training, copying the training data itself implicates copyright; whether training is sufficiently transformative, whether the model memorizes and reproduces original text, and whether it substitutes for the original author’s market are all disputed questions.

In addition, even if content uses Creative Commons or might fall under fair use, a platform’s Terms of Service may prohibit automated downloading. Publicly viewable videos, for example, are not automatically free to crawl. Large companies can obtain data from Reddit, Stack Exchange, Shutterstock, and similar sources through commercial licensing; open-source and academic teams rely more on public dumps, clearly licensed data, and careful filtering.

## 10. Mid-training and post-training data

Mid-training and post-training focus more on specific capabilities. Long-context extension is often done late because training with very long sequences from the start is too expensive; data may include books, long papers, long code, synthetic long-dependency tasks, and so on.

For instruction data, early examples include Super-Natural Instructions and FLAN, which unified traditional NLP tasks into an instruction format. Later came synthetic-data methods such as Alpaca/self-instruct, Vicuna, OpenHermes, and Evol-Instruct, where strong models generate tasks, answers, or multi-turn conversations. Synthetic data is cheap and scalable, but it is constrained by the generating model’s license and may inherit the teacher model’s biases. Another route is to hire annotators to write high-quality instruction data, which is expensive but controllable; even then, one must prevent annotators from secretly using commercial models to generate answers.

## 11. Engineering practice summary

When building a pretraining data pipeline, you can reason through the following process:

1. Define target capabilities: general knowledge, code, mathematics, multilinguality, long context, dialogue style.
2. Collect raw sources: Web crawls, dumps, APIs, licensed data, public-domain data.
3. Extract text: convert HTML/PDF/code/email and other formats into plain text while retaining necessary metadata.
4. Perform basic cleaning: language identification, encoding repair, boilerplate removal, length filtering, format filtering.
5. Select for quality: rules, classifiers, LLM scoring, link signals, community voting signals.
6. Handle safety and compliance: copyright, license, robots.txt, ToS, PII, toxicity.
7. Deduplicate and sample: exact/fuzzy deduplication, avoiding repeated sources dominating training.
8. Mix data: set mixture weights according to capability and quality, and validate through small-model ablations.
9. Record versions: save snapshot time, processing code, filtering thresholds, and statistics to ensure traceability.

In practice, do not look only at the final token count. More useful monitoring includes retention rate by source, duplicate rate, average document length, language distribution, domain distribution, perplexity or quality-score distributions, examples of filtered samples, and gains on target evaluations after training. The data pipeline should be versioned like model code; otherwise it is hard to explain why a training run improved or worsened.

The core conclusion of this lecture is: data does not fall from the sky. Trainable corpora are the result of extensive engineering, heuristics, legal judgment, and experimental iteration. Modern models may not differ much in architecture; data sources, filtering strategies, deduplication quality, synthetic data, and licensed resources are often the important factors that determine model differences.


---


# Stanford CS336 Lecture 14: Data (Part 2) — From Raw Web Pages to Trainable Corpora

> Adaptation note: This English tutorial is adapted from the Chinese lecture notes. It preserves the original structure, technical terminology, formulas, and code-style notation, while presenting the material in clear tutorial English.

This lecture continues the discussion of “data engineering” in large-model pretraining. The previous lecture was more like a history of datasets: from early corpora to Common Crawl, C4, The Pile, LLaMA, Dolma, and so on. This lecture turns to actionable methods: when we have massive raw Web pages and a small amount of ideal data, how do we filter, mix, deduplicate, and turn these decisions into a training-data recipe?

The core problem can be summarized as follows: given a small, high-quality target set T and a huge but noisy raw set R, find the subset T' of R that “looks like T.” This is not only quality filtering; the same idea also applies to language identification, domain selection, toxicity filtering, mining math/code data, filtering synthetic data, and adjusting different domain mixtures during training.

## 1. The basic paradigm of data selection

Data selection is not simply “keep Web pages that look good.” A general pipeline usually contains three steps:

1. Define the target: what data do we want? It may be Wikipedia-style text, textbook-style code, mathematical proofs, English Web pages, low-toxicity discussions, or a task domain needed by a product.
2. Train or construct a scorer: use target data and raw data to estimate a score `score(x)`, indicating how much sample x resembles the target domain, how valuable it is, or how safe it is.
3. Select or resample: keep samples above a threshold, sample according to probabilities, or redistribute data using importance weights.

There are two practical constraints. First, the scorer must generalize: it is meaningless if it only retrieves T itself; we need it to discover new similar samples from R. Second, the scorer must be fast enough: Web-scale data is enormous, and if a giant model scores every item, filtering may cost nearly as much as, or even more than, pretraining itself.

## 2. Three common kinds of filters

### 2.1 n-gram language models: crude but cheap

The most traditional method is to train an n-gram language model, for example using KenLM with Kneser-Ney smoothing. It essentially counts occurrences of n-word sequences and estimates conditional probabilities. For example, given the context “the cat,” it estimates the probability that the next word is “in.” Because many n-grams have never appeared, smoothing backs off to shorter contexts.

The usage is straightforward: train an n-gram model on the target corpus, then compute perplexity for raw documents. Lower perplexity means the text is more similar to the target domain. CCNet used a similar method to sort paragraphs by perplexity and keep the better parts; later LLaMA data was also influenced by this kind of pipeline.

The advantages of this method are speed, simplicity, and scalability. Its disadvantages are also obvious: it mainly looks at local co-occurrence and cannot truly judge long-range logic or semantic quality. Shuffled paragraphs, template spam, and text with locally normal grammar but globally meaningless content may all fool it. Therefore n-gram filtering is better suited for removing obvious noise than for fine-grained quality assessment.

### 2.2 fastText / linear classifiers: a common industrial baseline

fastText is a lightweight text classifier. It hashes words or n-grams into fixed buckets and then performs linear classification through low-dimensional representations. Although the structure is simple, it is fast, parallelizable, and suitable for Web-scale filtering.

A typical training setup is to construct a binary classification task: positive examples come from high-quality or target-domain data, and negative examples come from raw data such as Common Crawl. The classifier outputs the probability that “this sample comes from the target domain.” GPT-3 used high-quality sources as positives and Common Crawl as negatives to train a quality classifier; LLaMA used Web pages cited by Wikipedia as positives; Dolma used fastText for language identification and toxicity filtering.

The key value of fastText is not that it is “smart enough,” but that it is “cheap enough to run over the whole Web.” If raw data must be compressed down to 1%, the filter processes 100 times the final training volume; in that setting, scoring cost per sample must be extremely low.

### 2.3 Importance resampling: from “classification” to “distribution matching”

A classifier answers “does this look like the target domain,” but training data also needs to preserve distributional diversity. Importance resampling provides a more principled view: suppose the target distribution is P and the raw distribution is Q. We can only sample from Q, but we want the final samples to look as if they came from P. Therefore each sample receives a weight:

w(x) = P(x) / Q(x)

Then we resample in proportion to w(x). The intuition is: if some kind of text is common in the target domain but rare in the raw data, raise its sampling probability; if the opposite is true, lower it.

In real settings, P is hard to estimate accurately because target data is small. In practice, people use hashed n-grams to estimate a rough distribution, then compute approximate weights. This does not always bring huge gains, but compared with pure binary classification, it emphasizes distribution matching in the domain mixture rather than simply “passing a threshold.”

## 3. Quality assessment, domain mixture, and language selection

There is no single definition of “good data.” Quality may mean grammatical fluency, high information density, high educational value, low toxicity, little templating, fit to a target task, or provenance from trustworthy sources. Therefore data filtering is often decomposed into multiple independent dimensions.

Language identification is the most basic example. If the goal is an English model, mixing in many other languages consumes the token budget and reduces English training intensity. But if the model is large enough, multilingual data may also bring positive transfer. Bloom is about 30% English and emphasizes multilingual capability; frontier models usually cover hundreds of languages. Whether to filter by language is fundamentally a training-data decision: target users, model capacity, compute budget, and evaluation metrics jointly determine the mixture.

Domain selection is equally important. OpenWebMath treats “mathematics” as a special language: first use rules to find candidates, then use KenLM and fastText classifiers trained on mathematical-proof data such as Proof-Pile to filter them, ultimately obtaining about 15 billion math tokens. The results show that high-density data targeted at the math domain can outperform much larger but unfocused data. This illustrates that domain mixture is not “the bigger the better”; it must match the target capability.

Quality assessment can also be assisted by strong models. Phi-1’s idea was to train a small model, but give it high-value “textbook-style” code data. Researchers first asked GPT-4 to judge whether Python code snippets had educational value for beginners, obtaining about 100,000 labeled examples, then used a cheaper classifier to scale up to large data. This is a common pattern: use an expensive model to create a small high-quality T, then distill it into a cheap filter that can process R.

## 4. Synthetic data: target data can be “generated”

When no ready-made target corpus exists, a strong language model can synthesize or select target data. For example, it can be asked to generate textbook-style code, mathematical reasoning, chemistry QA, or to label Web pages with an “educational value” score. In this case, T is no longer just some existing source; it is defined by the requirement and the prompt.

But synthetic data has risks: the distribution may be too narrow, style may be uniform, errors may be amplified, and it may be highly similar to existing data. Therefore synthetic data usually should not be added without limit. It should go through quality classification, deduplication, human spot checks, and downstream evaluation. A more robust approach is to use synthetic data to improve a specific capability while preserving the diversity of real data; use a strong model to label a small batch, then train a cheap filter to scale it up.

## 5. Deduplication: reducing waste and memorization

Filtering decides “which data is worth training on”; deduplication decides “how many times the same information is trained on.” The Web naturally contains massive duplication: mirror sites, license text, product templates, copy-pasted articles, and template pages with only a few words changed. C4 once contained an ordinary English sentence that appeared tens of thousands of times. It is not bad text, but training on it 60,000 times is meaningless.

Exact deduplication is simple: hash sentences, paragraphs, or documents, group samples with identical hashes, and keep only one. It has high precision and is easy to parallelize, but it cannot find near duplicates. A Bloom filter uses a bit array and multiple hash functions to save memory; it has no false negatives, but may have false positives, making it suitable for approximate set queries at huge scale.

Near-duplicate deduplication is usually based on Jaccard similarity. Split documents into shingles or n-gram sets; if the intersection-over-union of two sets exceeds a threshold, treat them as near duplicates. Direct pairwise comparison is O(N²), which is infeasible. The key property of MinHash is that the collision probability of two sets under MinHash equals their Jaccard similarity. Combined with LSH (locality sensitive hashing), multiple hashes are divided into several bands, so highly similar documents collide with high probability and dissimilar documents collide with low probability. This finds duplicate candidates in linear or near-linear time.

Deduplication requires care. Removing junk Web duplicates during pretraining is usually beneficial. But in mid-training or continued training, repeating high-quality data for multiple epochs may be exactly what we want. A more reasonable strategy may be not to keep only one copy, but to downweight duplicate counts, for example sampling according to log or square root counts, so that “important and common” content receives higher weight without being amplified linearly by its raw repetition count.

## 6. Curriculum, annealing, and training-data recipes

Data decisions do not only happen before training. Many modern training runs change the mixture over time: early on, they use large-scale, diverse, loosely filtered data to learn general language and world knowledge; later, they gradually anneal toward high-quality, target-domain, instruction-style, or reasoning data to improve final evaluations. This is similar to curriculum learning: start broad, then increase density and difficulty.

Common strategies include:

- Expand coverage early: mix Web pages, books, code, multilingual data, forums, and so on, avoiding early overfitting to a narrow domain.
- Raise target-domain proportion in the middle: if code, math, or a particular language matters, gradually increase that domain’s tokens.
- Quality annealing late: reduce low-quality Web data and increase textbooks, QA, reasoning, human-written data, or data selected by strong models.
- Use synthetic data in limited amounts: avoid style collapse while using it to fill gaps in scarce capabilities.
- Adjust through evaluation loops: every mixture change should be checked against downstream benchmarks, perplexity, human samples, and safety metrics.

Therefore domain mixture is an optimization problem, not a static table. The best proportions usually cannot be written down by intuition in one attempt. They require training small models, doing ablations, inspecting data samples, and iterating. A practical principle is to record the data recipe as part of the model: token counts from each source, filtering thresholds, deduplication granularity, repetition sampling multipliers, and the time periods during which each source enters training should all be traceable. Otherwise, when a model changes on some capability or safety metric, it is hard to know whether the cause was model scale, optimization hyperparameters, or the data recipe.

## 7. Practical checklist

When building a pretraining corpus, you can audit it with the following questions:

1. What is the target capability? General chat, code, mathematics, multilinguality, or a specialized domain?
2. Where does the target data T come from? Human sources, trusted sites, strong-model labels, synthetic generation, or rule-based preselection?
3. Is the filter cheap enough? It processes the massive raw R, not the final small corpus.
4. How is the quality threshold chosen? Too loose keeps noise; too strict loses diversity and low-resource groups.
5. Does the mixture match the token budget? Increasing one domain’s proportion means reducing training opportunities for other domains.
6. Has exact and near-duplicate deduplication been done? Has training-set leakage into evaluation sets been avoided?
7. Is repeated high-quality data needed? If so, is repetition linear or downweighted?
8. Are data decisions validated through small-scale training rather than only by filter scores?

## Summary

The main thread of this lecture is: data does not naturally fall into the training set. It is produced through a series of computable and scalable decisions that are nevertheless full of trade-offs. n-gram models, fastText, and importance resampling provide basic tools for finding target data within raw Web pages; language identification, quality filtering, toxicity filtering, and domain mining are all instances of the same framework; synthetic data and strong-model labeling mean that “target data” itself can be designed; deduplication reduces meaningless repetition, lowers memorization risk, and saves compute.

Real data capability comes from a loop: inspect data, write filters, train models, evaluate results, adjust the mixture, and repeat. For large-model training, data selection is often as important as model architecture and compute scale, and in specific capabilities it may determine final performance even more strongly.


---


# CS336 Lecture 15 Tutorial: Alignment, SFT, and RLHF

> Adaptation note: This English tutorial is adapted from the Chinese lecture notes. It preserves the original structure, technical terminology, formulas, and code-style notation, while presenting the material in clear tutorial English.

## 1. From pretraining to alignment: why post-training is still needed

Pretraining compresses many capabilities into model parameters: language, knowledge, code, reasoning patterns, common sense, and many styles. But a base model trained only with next-token prediction usually does not naturally behave like a useful chat assistant. GPT-3 was already strong, but it did not follow instructions reliably. The key change in ChatGPT was post-training: making the model better at understanding user intent, more willing to complete tasks according to instructions, and able to refuse or redirect in dangerous situations.

In this lecture, alignment is not an abstract slogan but an engineering pipeline: first use supervised data to teach the model “how it should answer,” then use preference data plus reinforcement learning or alternative algorithms to push the model toward behavior humans prefer. The goals include helpfulness, truthfulness, and harmlessness. The difficulty is that these three often conflict. For example, to be helpful the model may fabricate an answer; to be safe it may over-refuse; to cater to preferences it may output answers that are longer but not more correct.

## 2. SFT: teaching the model to enter assistant mode through demonstrations

Supervised Fine-Tuning (SFT) is the first step in the InstructGPT pipeline. The data format is simple: given a prompt or dialogue context, provide an ideal response, then perform maximum-likelihood training on the response tokens. Intuitively, the model learns by imitating expert demonstrations.

Common SFT data sources fall into three categories:

- Task-aggregation data, such as FLAN: existing NLP datasets are rewritten into instruction format, including summarization, classification, question answering, multiple choice, and so on. The advantages are large scale and low cost; the disadvantages are that the format often does not look like real chat, many answers are short, and the task origin is obvious.
- Human-written data, such as OpenAssistant: volunteers or annotators write complex prompts and detailed answers. The advantages are naturalness and high quality; the disadvantages are cost, slowness, and difficulty in consistently controlling style and factual quality.
- Model-generated data, such as Alpaca: start from a small number of human seed prompts, expand them into more instructions, then use a strong model to generate answers. The advantages are low cost and many long answers; the disadvantages are inheriting teacher-model biases and possibly asking the student model to imitate capabilities it does not yet have.

A core lesson of SFT is that a small amount of high-leverage data can significantly change model behavior. A strong base model may need only relatively little instruction data to change from “complete the text” into “answer the user.” But “high-quality data” does not mean “the longer, more knowledge-dense, and more citation-heavy, the better.” If SFT examples require the model to answer facts it does not know, the training loss rewards it for generating tokens that look like correct answers, including strings that look like citations. The model may then learn not “retrieve facts and cite them,” but “when facing a complex question, fabricate a citation at the end.”

This shows that SFT more easily teaches the type signature and style of outputs: whether to use bullet points, whether to give long explanations, whether to cite, whether to apologize, and whether to refuse. It can also teach new knowledge, but small-scale SFT is often less stable for this than pretraining or large-scale mid-training. If demonstration data clearly exceeds the model’s existing capabilities, the model may learn hallucination-like shortcuts. Therefore good SFT data should match the model’s capabilities and include reasonable abstention behaviors such as “I don’t know,” “more information is needed,” and “please verify this.”

## 3. Safety SFT: balancing refusal and over-refusal

Safety alignment can also be injected through SFT. Mixing a small batch of safety examples into instruction tuning can teach the model to refuse or provide safe alternatives in scenarios involving scams, malware, violence, self-harm, and similar risks. In research, even a few hundred carefully constructed safety examples can produce visible effects.

But safety is not simply increasing the refusal rate. The real difficulty is distinguishing dangerous requests from requests that look dangerous on the surface but are legitimate. For example, “how can I kill a Python process?” refers, in a computing context, to terminating a process, not harming a living thing. If the data only teaches the model to refuse whenever it sees sensitive words, it will produce over-refusal and reduce usefulness. Safety data needs to cover boundary cases, dual-use questions, contextual ambiguity, and safe versions of allowed answers.

## 4. SFT training is becoming closer to pretraining

In academic settings, SFT is often understood as: take a base model and run a few epochs of gradient descent on instruction data. But frontier-model post-training has become more like a full training stage. Many modern pipelines mix high-quality data, code SFT, question answering, chat, multilingual books, and safety data near the end of pretraining during the learning-rate decay phase. This is often called mid-training or decay-stage data mixing.

The benefit is that data scale can be larger, the model is less likely to catastrophically forget due to a brief fine-tuning phase, and instruction behavior can be integrated more deeply into the model. The cost is that the boundary between “base model” and “chat model” becomes blurry. Today, many so-called base models may already have seen large amounts of instruction-style data late in training. Therefore, when comparing different models, remember that base does not necessarily mean completely unaligned.

## 5. Why preference data is needed

SFT requires humans or strong models to directly write ideal answers, but producing high-quality long answers is expensive and tiring, and the answers humans write themselves are not necessarily the answers they most prefer. Verification is often easier than generation: asking an annotator to compare A/B and choose which is better is usually cheaper than asking them to write a perfect answer from scratch. This is the motivation for preference data.

The basic form of preference data is: for the same prompt, two or more model answers are shown, and the annotator chooses the better one. InstructGPT-style annotation guidelines usually revolve around three ideas: helpful, truthful, and harmless. Real guidelines are more detailed: whether the answer addresses the user’s true intent, follows the required format, hallucinates, is toxic, contains inappropriate content, or should ask for clarification.

But preference annotation is also difficult. Annotators often work under time pressure and may not have enough time to check facts, verify calculations, or detect subtle hallucinations. Longer answers are more likely to be judged “detailed and helpful,” even when they contain errors. Different annotators focus on different things: experts emphasize factuality, while ordinary crowdworkers may emphasize formatting, fluency, and politeness. Annotators’ cultural, national, religious, and political backgrounds also affect value judgments, and alignment sits at the end of the pipeline where it has strong influence on final model behavior.

Therefore preference data is not only a technical resource; it is also a social choice. It requires clear rubrics, fair pay, quality audits, diverse annotator populations, and transparent records of bias sources.

## 6. Reward Model: turning pairwise preferences into optimizable rewards

The classic second step of RLHF is to train a reward model. Suppose each answer y under prompt x has a latent scalar reward R(x, y), but we cannot observe it directly; we can only observe human comparisons: whether answer A is better than answer B.

A common formulation is the Bradley-Terry preference model: the probability that A beats B depends on the difference R(x, A) - R(x, B), usually passed through a sigmoid. When training the reward model, the chosen answer is encouraged to score higher than the rejected answer. Once trained, the reward model can assign a scalar score to any new answer and serve as the reward signal for RL.

Note that the reward model is only an approximation to human preferences, not truth itself. It learns biases in the annotation data, such as preference for longer answers, preference for lists, or preference for a certain tone, and it can also be exploited by the policy model. If the RL stage over-optimizes the reward model, the model may receive high reward while real humans dislike it. This is reward hacking, or Goodharting.

## 7. RLHF: using PPO to optimize between reward and constraints

The classic third step of the InstructGPT pipeline is to optimize the policy model with PPO. The goal is not to continue imitating a reference distribution, but to find a policy π(y|x) whose expected reward from the reward model is higher.

The practical objective usually contains constraints:

- Reward term: make the model generate answers the reward model likes.
- KL penalty: restrict the post-RL policy from moving too far away from the SFT model, avoiding collapse in language quality, mode collapse, or reward hacking.
- Sometimes pretraining loss is mixed in to reduce catastrophic forgetting.

PPO can be understood as a stable engineering version of policy gradient. The model samples answers, the reward model scores them, and the advantage increases good outputs and weakens bad outputs; at the same time, importance ratios and clipping limit the size of each update. It is effective, but complex to implement, difficult to tune, unstable to train, and unfriendly to academic and open-source practice.

We also need to distinguish on-policy from off-policy data. On-policy data comes from the model currently being optimized, so it can improve the model’s current mistakes. Off-policy data comes from other models or older models; it is cheaper and reusable, but may not cover the regions the current model most needs to fix. Modern pipelines often mix the two.

## 8. DPO and alternatives: turning the RL problem into a supervised loss

Because PPO is troublesome, researchers have tried many alternatives: do SFT only on preferred responses; add good/bad tokens to chosen/rejected responses and condition generation on them; use a reward model to sample multiple answers and then train on the best one. These methods sometimes work, but are usually less stable than classic RLHF.

Direct Preference Optimization (DPO) became popular because it removes the explicit reward model and PPO rollouts, rewriting preference optimization as a direct supervised-style loss. The key idea is: in the optimal-policy problem with KL regularization, a policy implicitly defines a reward; substituting this implicit reward into the Bradley-Terry preference model lets us directly maximize the probability that the chosen response is preferred over the rejected response.

The training intuition of DPO is simple: relative to a reference model, increase the log probability of the chosen response, decrease the log probability of the rejected response, and use a coefficient to control the distance from the reference. Unlike PPO, it does not require online sampling or complex RL state, so it is easier to implement, reproduce, and scale. Its weakness is that it usually relies on existing preference pairs and uses less on-policy exploration; if the preference data is low quality, distributionally biased, or entirely old-model outputs, DPO is also limited.

Related alternatives include RLAIF (AI feedback), Constitutional AI, rejection-sampling-style training, and various DPO variants. RLAIF uses strong models instead of humans to make preference judgments, with lower cost and larger scale. GPT-4-like judges correlate with human preferences on many open-ended tasks. But AI judges have self-preference, length bias, position bias, and value bias, and should not be treated as unbiased annotators.

When using AI feedback, also watch for “closed-loop amplification”: if the same model family generates candidates, judges them, and is used to distill a student model, the system may become increasingly biased toward the expression style that family likes, rather than moving closer to real user needs. Better practice is to mix multiple judges, sample for human review, keep hard negative examples, and report win rates after length normalization separately. For fact-dense or mathematical tasks, it is better to use verifiable signals, tool checks, or expert audits rather than relying only on open-ended preferences.

## 9. Practical checklist for an alignment pipeline

A typical alignment pipeline can be organized as follows:

1. Start from a strong base model, and prepare a chat template and basic instruction data.
2. Perform SFT so the model reliably enters assistant mode, covering general capabilities, format following, multi-turn dialogue, and initial safety refusals.
3. Collect prompts and let one or more models generate candidate answers.
4. Use humans or AI to annotate pairwise preferences and build chosen/rejected data; also audit length, style, factuality, and annotator agreement.
5. Train a reward model, or directly use preference optimization methods such as DPO/IPO/KTO.
6. If using PPO/RLHF, add KL constraints and safety monitoring to avoid reward hacking.
7. Validate with multidimensional evaluations: open-ended preference, factuality, math and code benchmarks, refusal rate, over-refusal rate, red-team testing, real user tasks, cost, and latency.
8. Iterate on data based on failure cases instead of chasing a single leaderboard.

## 10. Risks and evaluation: do not treat preference as truth

Alignment is easily misunderstood as “making the model more likable.” This is dangerous. Humans and AI judges both prefer long, structured, polite, seemingly confident answers, but these features are not the same as correctness. A model may win preference evaluations because it is better at using bullet points, citing, and catering to the user, while hallucinating more.

Therefore evaluation must be diverse:

- Open-ended human eval or chatbot arenas measure user preference, but length and judge biases must be controlled.
- Standard benchmarks measure knowledge, reasoning, code, and mathematics, preventing post-training from optimizing only style.
- Safety evaluation should measure both harmful compliance and over-refusal; “refuse everything” must not masquerade as safety.
- Factuality evaluation needs to verify claims, not merely judge fluency.
- Real deployment must also monitor distribution shift, jailbreak attacks, abuse, user satisfaction, cost, and latency.

In summary, the main thread of Lecture 15 is: pretraining gives the model capabilities; SFT teaches the model how to behave; preference data tells the model which behaviors are more liked; a reward model or DPO turns preferences into an optimizable objective; and RLHF/preference optimization pushes the model toward more helpful, truthful, and safe regions. The real challenge is not only the algorithm, but also data quality, annotation incentives, value bias, evaluation bias, and the balance between safety and usefulness.


---


# CS336 Lecture 16 Tutorial: Alignment with Reinforcement Learning (Part 1)

> Adaptation note: This English tutorial is adapted from the Chinese lecture notes. It preserves the original structure, technical terminology, formulas, and code-style notation, while presenting the material in clear tutorial English.

This lecture is the second lecture in the post-training section. The topic moves from traditional RLHF to “reinforcement learning from verifiable rewards.” The central questions are: why does language-model alignment need RL? What are algorithms such as PPO and GRPO actually optimizing? Why is training unstable, and what engineering details matter most?

## 1. From RLHF to verifiable rewards

RLHF (Reinforcement Learning from Human Feedback) usually starts from human preference data: given two responses to the same prompt, humans label which one is better. The goal is to train a language-model policy so that it is more likely to produce responses humans prefer.

Here, policy means the model’s probability distribution over output sequences given a context. Unlike pretraining or SFT (supervised fine-tuning), RLHF is not simply “fitting a data distribution”: what the model generates changes the reward it receives, so the objective includes sampling from the current model. This makes optimization harder than ordinary maximum likelihood.

DPO (Direct Preference Optimization), discussed in the previous lecture, turns preference optimization into an objective similar to supervised learning. It does not explicitly train a reward model or run a full RL loop; instead, it directly adjusts the policy using preference pairs. The intuition behind DPO is simple: increase the probability of the chosen response and decrease the probability of the rejected response. The more wrong the model’s implicit reward judgment is, the larger the update. Because DPO is easy to implement, it became a mainstream post-training method for open-source models for a period of time.

However, DPO also has limitations. It naturally fits pairwise preferences, but it is less suitable for tasks that have only scalar rewards, such as “right or wrong” answers to math problems. It is also usually offline: first collect a batch of preference pairs, then train on them. For reasoning models, researchers would prefer to optimize online, directly from verifiable outcomes while the model continuously generates new solutions.

## 2. Two risks of RLHF: overoptimization and worse calibration

One of the most important empirical phenomena in RLHF is overoptimization. A reward model is only a proxy for human preference, and it contains noise and errors. Early in training, optimizing the proxy reward often improves true human preference. But after further optimization, the model may begin to “exploit holes in the reward model”: the proxy reward keeps rising, while the true win rate stagnates or even falls. This is similar to the train-test gap in supervised learning: the reward model on the training set is not the same as a real preference oracle.

Another phenomenon is worse calibration. A pretrained model can be viewed as a probabilistic generative model, while a model after RLHF is more like a policy adjusted for a particular reward. If the reward does not encourage “expressing uncertainty,” the model may become more confident, more sycophantic, and less willing to say “I don’t know.” Therefore, do not interpret the output probabilities of an RLHF model directly as reliable estimates of true probabilities.

These issues show that human preference is valuable, but difficult to optimize at large scale with low noise and high stability. A natural direction is therefore to look for tasks with clearer rewards, such as mathematics, code, formal proofs, executable tests, and similar domains. In these areas, whether an answer is correct can be automatically verified. The reward is closer to the true goal and less vulnerable to reward hacking.

## 3. RL basics: policy, reward, value, advantage

In language-model RL, one sample usually contains a prompt and a response generated by the model. Generating the full response can be viewed as one rollout. Common terms are:

- Policy: the current language model `πθ`, the probability distribution over token sequences after a prompt.
- Reward: the score assigned to the generated result. In RLHF it may come from a reward model; in math or code it may come from answer matching, unit tests, format checks, and so on.
- Value function: a function that estimates how much future reward a state or partial generation will obtain. It is often used to reduce the variance of policy gradients.
- Advantage: how much better an action or output is than a baseline. Intuitively, if the advantage is positive, increase the probability of that output; if it is negative, decrease it.

The most basic policy-gradient idea is: increase the log probability of high-reward outputs, and decrease the log probability of low-reward outputs. Many RL algorithms can essentially be understood as “upweight good stuff, downweight bad stuff.” They differ in how they define good and bad, how they reduce variance, and how they prevent the policy from moving too far in one step.

Language-model RL has another feature: many tasks are closer to contextual bandits. The model sees a prompt, generates a complete answer, and then receives a terminal reward. There are no complex state transitions like in traditional game environments. However, during training, regularization terms such as KL penalties are still often distributed at the token level, while task rewards such as “right or wrong” are placed at the final token or sequence level.

## 4. PPO: powerful but engineering-heavy

PPO (Proximal Policy Optimization) was one of the most important early algorithms for RLHF. Starting from policy gradient, it introduces two key mechanisms.

First, importance sampling and the old policy. Pure on-policy methods require each update to use newly generated samples from the current policy, which is expensive because rollout is slow. PPO allows us to first sample a batch of data with an old policy, then perform multiple gradient updates on the same batch of rollouts.

Second, clipping. PPO does not want the new policy to change too much relative to the old policy. It therefore uses a probability ratio and clips it between `1-ε` and `1+ε`, for example from 0.8 to 1.2. In this way, even if a sample has very high reward, the model cannot push its probability up without limit, which improves training stability.

PPO also usually needs a value model to estimate the advantage, for example with GAE (Generalized Advantage Estimation). This reduces gradient variance, but the cost is engineering complexity: one must maintain a policy model, reward model, and value model, and sometimes deal with different tokenizers, KL shaping, value loss, policy loss, clip norm, synchronization between rollout and training workers, and more. Real PPO has many implementation details; small differences can affect the result.

For large language models, the value model is especially expensive: it is often as large as the policy, so memory and compute costs almost double. Therefore people want a method that preserves PPO’s stability while removing the value model.

## 5. GRPO: replacing the value model with an in-group baseline

GRPO (Group Relative Policy Optimization) can be viewed as a simplified variant of PPO and is also a key algorithm in the DeepSeek Math / R1 series. It preserves ideas such as policy gradient, KL regularization, and ratio clipping, but removes the value function and the complicated GAE machinery.

The core idea of GRPO is: for the same question `q`, sample `G` responses at once to form a group. Each response has a reward. Then construct the advantage using the mean and standard deviation of rewards inside the group:

A_i = (r_i - mean(r_1, ..., r_G)) / std(r_1, ..., r_G)

In other words, we no longer ask “how high is the absolute reward of this response?” Instead, we ask “how much better is it than the other responses to the same question?” This is natural: different questions have different difficulties. Easy questions have high average rewards, and hard questions have low average rewards. The in-group mean can serve as a baseline for question difficulty. This removes the need to train an additional value model.

If we do only one online update per batch of rollouts, GRPO can even become very close to ordinary policy gradient: responses above the group average are pushed up, and responses below the average are pushed down. Implementation only requires generating multiple responses, computing rewards, normalizing within groups, adding a KL penalty, and performing a gradient update.

But GRPO also has subtle issues. Standard-deviation normalization is not an ordinary baseline that is strictly allowed by the policy-gradient derivation. It amplifies groups with very small reward variance, such as questions where all responses are wrong or all are correct. This may shift training focus toward questions that are “too hard” or “too easy,” rather than moderately difficult questions with the most useful learning signal.

Another issue is length normalization. If the sequence reward is divided by output length, then when the answer is wrong the model may generate longer content to dilute the negative reward; when the answer is correct, it tends to be shorter. This can induce the model to output very long chain-of-thought when uncertain, which looks like “thinking longer” but may simply be a bias of the objective. Later analyses such as Dr. GRPO argue that removing some forms of length normalization can preserve reward while reducing unbounded length growth.

## 6. Why verifiable rewards drive reasoning models

Take DeepSeek R1 as an example. Its training process demonstrates a very simple but effective paradigm: on verifiable tasks such as math and code, perform RL with outcome reward, meaning whether the final answer is correct. R1-Zero performs RL almost directly from a base model. The reward mainly includes accuracy reward and format reward. The format reward requires the model to place reasoning inside specific think tags. Although this looks like only a formatting constraint, in practice it is important for stable training.

An important conclusion from R1 is that complex MCTS search or PRM (Process Reward Model) is not necessarily required. A PRM can score intermediate reasoning steps and in theory provides richer feedback, but it is difficult to build a reliable process supervisor. R1 found that simple outcome-based reward plus GRPO can already produce strong reasoning ability.

Models released for real use usually do not do only RL. A more common pipeline is: first perform a small amount of long chain-of-thought SFT so that the model learns a readable reasoning format; then perform verifiable-reward RL to improve math/code correctness; finally perform general instruction tuning and RLHF to restore broad abilities such as chat, writing, safety, and general assistance. This shows that SFT and RL are complementary: SFT provides the initial behavior pattern, while RL continues optimizing the model toward the real objective.

Kimi K1.5 and Qwen3 reflect similar ideas. Kimi emphasizes data filtering and length control: using best-of-N to filter out overly easy questions, constructing curriculum learning, and adding a length reward late in training to prevent reasoning chains from becoming too long and making inference cost uncontrollable. Qwen3 adds thinking mode fusion: the same model supports both think and no-think modes, and test-time thinking length can be controlled through a token budget, enabling inference-time scaling.

## 7. Training stability and engineering notes

The difficulty of LLM RL is not only algorithmic, but also systemic. Rollout requires autoregressive generation, which is much slower than ordinary teacher-forcing training. After training workers update weights, the weights must also be synchronized to inference workers. Long chain-of-thought creates uneven batch lengths and reduces GPU utilization. Many systems separate training and inference into different workers and use inference engines such as vLLM to generate samples.

Stable training usually depends on the following techniques:

1. KL regularization: constrain the new policy so it does not drift too far from the reference policy, avoiding collapse in language quality.
2. Clipping or explicit regularization: control the size of a single policy update.
3. Reasonable baseline: reduce policy-gradient variance, such as PPO’s value function or GRPO’s in-group mean.
4. Reward shaping: add auxiliary objectives such as format, language consistency, and length as weighted rewards, but the weights require empirical tuning.
5. Data difficulty control: examples that are too easy provide no learning signal, and examples that are too hard are all wrong and also provide no signal; best-of-N filtering and curriculum can improve training efficiency.
6. Length control: longer reasoning may improve performance, but it may also be induced by the objective; one must trade off correctness against inference cost.

## 8. Division of labor among SFT, RL, and RLHF

Putting this lecture back into the full alignment pipeline, we can see that three kinds of training play different roles. SFT gives the model demonstrations of “how it should answer,” such as following instructions, writing long reasoning chains, using a fixed format, and avoiding obviously harmful outputs. Its advantages are stability, low cost, and ease of debugging; its disadvantage is that it can only imitate behaviors already present in the data and cannot directly encourage the model to explore better solutions than the demonstrations.

RL moves the model from “able to imitate” toward “able to optimize an objective.” In math and code, the model can try many different solutions. As long as the final answer is correct or the tests pass, it receives positive reward. This exploration can discover behavior patterns not covered by SFT data and can continually push up the probability of correct answers. RLHF extends the optimization target from verifiable tasks to human preferences, such as helpfulness, politeness, safety, and stylistic consistency. But because preference rewards are noisier, RLHF needs KL, early stopping, evaluation sets, and human inspection even more to prevent overoptimization.

Therefore, a practical order is usually: first use SFT to establish a controllable initial policy; then perform RL on high-quality, verifiable, appropriately difficult data to improve reasoning and problem-solving; finally use preference optimization or RLHF to correct the general chat experience. If the order is reversed and one starts RL directly from a weak or format-chaotic model, the reward may be too sparse and training unstable. If one does only SFT and no RL, the model may remain at the stage of “looking like it can reason,” rather than achieving the highest accuracy on truly verifiable targets.

Evaluation must also distinguish among different goals. Improvement on math leaderboards does not mean the general assistant is better, and improvement in general preference does not mean reasoning is stronger. A reliable training pipeline must monitor task accuracy, response length, KL distance, format violation rate, refusal rate, human preference, and safety metrics at the same time. Only when these curves are jointly reasonable can we say that RL is truly improving the model, rather than merely exploiting a benchmark or reward loophole.

## 9. Summary

The main thread of this lecture is: RLHF showed that RL can be used for language-model alignment, but human-preference reward is noisy and easy to overoptimize; verifiable rewards provide clearer and more scalable training signals. PPO is a classic and powerful RLHF algorithm, but the value model and many implementation details make it costly and difficult to tune. GRPO replaces the value model with relative in-group rewards from multiple responses to the same question, greatly simplifying training, and therefore has become an important post-training tool for reasoning models.

From the experience of R1, Kimi K1.5, and Qwen3, successful recipes often include a small amount of high-quality long-CoT SFT, RL on verifiable tasks, stabilizing constraints such as KL/length/format, and then general RLHF or instruction tuning. The final goal is not to make the model “think indefinitely,” but to push the policy toward higher accuracy, better alignment, and more stable behavior under controllable cost.


---


# CS336 Lecture 17 Tutorial: Alignment with Reinforcement Learning (Part 2)

> Adaptation note: This English tutorial is adapted from the Chinese lecture notes. It preserves the original structure, technical terminology, formulas, and code-style notation, while presenting the material in clear tutorial English.

This lecture continues the previous lecture’s themes of RLHF and RL for Verifiable Rewards (RLVR). The focus is not on introducing entirely new concepts, but on carefully unpacking policy gradients for language models, PPO/GRPO-style algorithms, reward design, and key details in engineering implementation. The central question is: once a model already has some capability, how can we continue optimizing it with “scoreable outcomes,” rather than merely imitating human-labeled data?

## 1. The reinforcement-learning setting for language models

In classical reinforcement learning, we need to define states, actions, rewards, and transition dynamics. For language models, these concepts have very concrete correspondences:

- State: the prompt plus the response prefix generated so far.
- Action: generating the next token.
- Trajectory / episode / rollout: starting from a prompt, the model continuously generates a complete answer.
- Reward: how good the answer is overall.

This lecture mainly discusses outcome reward, meaning a reward that is assigned only after the full answer has been generated. For example, in a math problem, the model first writes its reasoning process and finally outputs “the answer is 3 miles.” The reward function extracts the final answer and compares it with the ground-truth answer. If it matches, the reward is 1; otherwise, it is 0.

This differs from general RL in an important way: the transition dynamics of a language model are very simple. The new token is simply appended to the existing context. Therefore, language models naturally support test-time compute: sample multiple candidate answers, search, rerank, and verify. In robotic control, it is difficult to have world dynamics that can be simulated so exactly.

But the difficulty also shifts. Robots are often hard because they must “reach a physical state.” Language models can write almost any token sequence; the hard part becomes whether those tokens truly correspond to correct reasoning, correct answers, and reliable behavior.

## 2. From SFT to policy gradient

The RL objective for language models is to maximize expected reward:

\[
J(\pi)=\mathbb{E}_{s\sim p(s), a\sim \pi(\cdot|s)}[R(s,a)]
\]

Here \(s\) is the prompt, and \(a\) can temporarily be viewed as the full response. The basic policy-gradient trick is:

\[
\nabla J(\pi)=\mathbb{E}[R(s,a)\nabla \log \pi(a|s)]
\]

Intuitively, this looks very similar to SFT. In SFT, a human provides a good answer and the model maximizes the probability of that answer. In policy gradient, the model samples an answer itself, then weights that answer according to the reward. If the reward is 1, increase the probability of that response; if the reward is 0, the naive form produces almost no update.

This explains why RLVR requires the initial model to have some capability. If the task is too hard and the model almost never samples a correct answer, all rewards are 0, the gradient is also close to 0, and training cannot get started. Practical systems usually need:

- a sufficiently strong base/SFT model;
- enough sampling to increase the probability of encountering positive reward;
- smoother rewards or partial rewards;
- baselines, advantages, normalization, and other variance-reduction methods.

## 3. Baselines, advantages, and variance reduction

The main problem with policy gradient is high variance. A high absolute reward does not necessarily mean that the action is a good choice under the current prompt. For example, a wrong answer on an easy problem might receive 9 points, while a relatively good answer on a hard problem might receive only 2 points. If we update directly according to reward magnitude, the model may incorrectly favor “suboptimal actions under easy prompts.”

The solution is to introduce a baseline:

\[
\mathbb{E}[(R(s,a)-B(s))\nabla\log\pi(a|s)]
\]

As long as \(B(s)\) does not depend on the action \(a\), it does not change the direction of the expected gradient, because the subtracted term is a policy-independent constant. But it can greatly reduce variance.

A common choice is to make the baseline approximate the expected reward under the current state:

\[
B(s)\approx \mathbb{E}_{a\sim\pi}[R(s,a)]
\]

Then \(R(s,a)-B(s)\) is the advantage: how much better this response is than the average response under the same prompt. Responses better than average are strengthened, and responses worse than average are suppressed. This also answers the question “why not set the reward for wrong answers to -1?” After centering, samples below the group average naturally receive negative advantage.

## 4. GRPO: using multiple responses to the same prompt as a group

PPO usually uses a value function / critic to estimate the baseline. GRPO (Group Relative Policy Optimization) uses an idea that is more natural for language models: for the same prompt, sample multiple responses at once, form a group, and use the mean reward inside the group as the baseline.

The rough procedure is:

1. Start from a batch of prompts.
2. Sample multiple candidate responses for each prompt.
3. Compute the reward for each response.
4. Compute the mean and standard deviation inside the group for the same prompt.
5. Use \((r_i-\bar r)/\sigma\) as the update signal.
6. Update the current policy so that the probabilities of responses above the group average go up, and the probabilities of responses below the group average go down.

This is what “relative” means: we do not ask whether a response has a high absolute score; we ask whether it is better than the other responses under the same prompt. Language models naturally fit this structure, because multiple candidates can be generated in parallel for the same prompt. In traditional robotic RL, trajectories often have very different states, so this group structure is not as natural.

Standardization has another advantage: changes in reward scale become less sensitive. If all rewards are multiplied by 100, the normalized advantages are basically unchanged. However, if all responses in a group have exactly the same reward, all centered deltas are 0, and the model will not update from that group. This matches intuition: there is no relative quality signal inside the group.

## 5. Ratio, clipping, and KL in PPO/GRPO

A common important quantity in policy optimization is:

\[
\rho=\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}
\]

It represents how much the current policy has changed the probability of the same response relative to the old policy used during sampling. PPO/GRPO multiply it by the advantage and apply clipping:

\[
\text{clip}(\rho, 1-\epsilon, 1+\epsilon)
\]

The purpose of clipping is to limit the size of a single update and prevent the model from suddenly drifting too far because of a few high-reward samples. In implementation, one must be careful: \(\pi_{old}\) should be treated as a constant, and gradients must not pass through the old policy. Engineering implementations usually use `no_grad`, or directly save the old policy’s log probability for these responses during rollout.

Besides the old policy, training may also have a reference model for KL regularization:

\[
\text{reward objective} - \beta \mathrm{KL}(\pi_\theta || \pi_{ref})
\]

The reference model is usually the initial SFT model or a more slowly updated model. The KL penalty prevents the current model from drifting too far from the original language ability in pursuit of reward, such as producing broken formats, excessive opportunism, or losing generality. In practice, three kinds of models or quantities may coexist: the current training policy, the old policy used for the ratio, and the reference policy used for KL.

## 6. Reward design, reward hacking, and verifiable rewards

The appeal of RLVR is that rewards can be computed automatically, without asking humans for every sample. Examples include matching final answers in math problems, passing code unit tests, producing correct sorting results, and satisfying theorem provers. These rewards are deterministic, scalable, and low-cost.

But reward design is very dangerous. The sorting example from the lecture illustrates this point: if the reward only checks whether “output tokens come from the input” and whether “adjacent tokens are ordered,” the model may find loopholes, such as repeatedly outputting certain tokens or exploiting local sortedness to obtain a high score without actually completing the sorting task. This is reward hacking: the model optimizes the metric, but not the true objective we care about.

The denser the reward, the stronger the training signal, but the more likely it is to introduce wrong shortcuts. The sparser the reward, the closer it may be to the real objective, but the harder optimization becomes. Common engineering compromises include:

- use an exact final reward to guarantee the target is correct;
- use partial rewards to help exploration, while continuously checking whether they can be exploited;
- strictly parse and validate the output format;
- use hidden test sets or diverse environments to prevent overfitting to the reward;
- manually inspect high-reward samples to look for opportunistic patterns.

## 7. Relationship between reasoning models and RLVR

The key to reasoning models is not simply “learning to write longer chain-of-thought.” Rather, on verifiable tasks, massive sampling and reinforcement learning make the model more frequently produce reasoning trajectories that lead to correct answers. As long as the reward can reliably judge the final outcome, the model has a chance to discover strategies more effective than human demonstrations.

This is also the potential of RL compared with SFT: SFT can only imitate existing answers, while RL can exceed demonstrations on measurable objectives. But the prerequisite is that the “measurement” itself is trustworthy enough. Mathematics, code, formal proofs, games, and tool-use environments are better suited to RLVR. Open-ended writing, value judgments, and real user satisfaction are closer to RLHF and require reward models, human preferences, or LLM-as-judge methods, but they are also more vulnerable to bias and reward attacks.

## 8. Evaluation and engineering risks

At the end of the lecture, the instructor emphasizes that RL training is much more complex to engineer than pretraining or SFT. Pretraining is mainly next-token loss on a fixed dataset; RL requires repeatedly generating new data, scoring it, updating the model, and generating new data again. The loss itself is also no longer directly interpretable in the way supervised-learning loss is, because the training distribution changes with the policy. What must really be monitored are reward, pass rate, format error rate, KL, sample diversity, and external evaluation performance.

During evaluation, it is especially important to distinguish “the training reward increased” from “the real capability improved.” If the reward function has a loophole, the training curve will look good, but the model may simply have learned to output a certain template, exploit a parser bug, repeat high-scoring tokens, or trigger unexpected behavior in the test environment. Therefore, at least three types of evaluation should be prepared. First, an in-distribution validation set matching the training distribution, used to quickly detect overfitting and training collapse. Second, hidden or harder out-of-distribution evaluations, used to check whether the model has learned a general strategy. Third, human review of samples, used to find errors that automatic metrics have difficulty capturing, such as fabricated reasoning, format opportunism, or inconsistency between explanation and answer.

Also note that a “good sample” in RL is not necessarily always good. A high-reward answer sampled early in training might be only accidentally correct. If too many gradient steps are taken on it, the policy can quickly collapse into a narrow mode and exploration decreases. Sampling temperature, the number of candidates per prompt, how many steps each batch of data is reused for, clip range, KL coefficient, and reference-model update frequency all affect the balance between exploration and stability. In practice, one often needs to inspect average reward, best-sample reward, response length, repetition rate, entropy, refusal rate, and KL curves together, rather than relying on a single metric.

Engineering must also handle:

- inference cost: each prompt requires multiple sampled responses;
- reward computation cost: tests may need to be run, environments called, or agents executed;
- multi-model management: current policy, old policy, reference model, critic/reward model;
- distributed synchronization: rollout workers and trainer must exchange models, samples, and logprobs;
- memory overhead: a reference model may double GPU memory usage;
- stale policy: samples are generated by old parameters but trained with new parameters, so the drift must be controlled.

A typical RLVR system separates rollout, reward, training, and evaluation into multiple services. Rollout workers use the current or slightly stale model to generate candidates; reward workers parse answers, run tests, or call environments; the trainer updates the model using saved logprobs, rewards, and advantages; the evaluator periodically evaluates on fixed benchmarks. An error in any link can contaminate training: parser bugs create incorrect rewards, environment nondeterminism increases reward noise, workers using overly stale models distort the ratio, and incomplete distributed logs make issues hard to reproduce.

Therefore, RLVR is not only an algorithmic problem, but also a systems problem. A workable training system must ensure that sampling, rewards, optimization, monitoring, and safety evaluation are all reliable. The stronger the optimizer, the more it amplifies flaws in the reward definition. Before scaling up training, one should repeatedly validate the reward function with small models, small data, and interpretable examples.

## 9. Summary

The main thread of this lecture can be summarized as follows: language-model RL treats complete answers as action sequences, evaluates outcomes with verifiable or learned rewards, and then uses policy gradients to increase the probability of high-reward answers. Naive policy gradient is like “SFT weighted by reward,” but high variance, sparse rewards, and credit assignment make training difficult. Baselines and advantages reduce variance through relative comparison; GRPO naturally constructs an in-group baseline by sampling multiple responses to the same prompt; clipping and KL regularization control update size and prevent policy collapse.

RLVR is an important path for training reasoning models because it lets the model self-improve on verifiable tasks. But its success depends on whether the reward function is truthful, hard to hack, and generalizable, and whether the entire training system is stable. The final principle is: if you can measure something reliably, you can optimize it; if the measurement has loopholes, optimization will amplify those loopholes. In other words, improvements in reasoning-model capability come from the closed loop of “generate—verify—update,” not merely from writing longer reasoning text. What really matters is whether the verifier can distinguish effective reasoning from plausible-looking nonsense.


---
