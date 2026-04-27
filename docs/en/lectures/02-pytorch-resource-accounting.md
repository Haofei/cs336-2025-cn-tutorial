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
