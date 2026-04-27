# Stanford CS336 2025 第 2 讲教程：PyTorch 与资源核算

本讲的主题不是“再讲一遍 Transformer”，而是训练大模型时更底层、也更容易被忽略的能力：会用 PyTorch 搭模型，并且能随时估算它消耗多少显存、多少计算、多少时间和多少钱。研究代码不能只追求“能跑”，当参数量、token 数和 GPU 数量变大时，每一次矩阵乘法、每一份 optimizer state、每一次 CPU/GPU 数据搬运都会变成真实成本。

## 1. 为什么要做资源核算

课程一开始用两个纸上估算问题建立直觉。

第一个问题：用 1024 张 H100，在 15 万亿 token 上训练一个 700 亿参数 dense Transformer，大约要多久？粗略公式是：

```text
训练 FLOPs ≈ 6 × 参数量 × token 数
可用 FLOPs/天 ≈ GPU 数 × 单卡峰值 FLOPs/s × MFU × 86400
训练天数 ≈ 总训练 FLOPs / 每天可用 FLOPs
```

如果假设 H100 的有效利用率 MFU 为 0.5，结果大约是一百多天量级。这里最重要的不是精确数字，而是思维方式：先估总计算量，再除以硬件实际吞吐。

第二个问题：8 张 80GB H100，如果使用 AdamW 且不做复杂优化，最多能放多大的模型？一个常用粗估是每个参数需要约 16 字节：参数、梯度、Adam 的一阶矩、二阶矩等状态共同占用显存。于是：

```text
最大参数量 ≈ 8 × 80GB / 16 bytes ≈ 400 亿参数
```

这还没有严肃计入 activation、batch size、sequence length 等因素，所以只是上界级别的估算。真正训练时，activation 往往也会成为瓶颈。

## 2. PyTorch 张量：所有东西的原子

在 PyTorch 中，参数、梯度、optimizer state、数据和中间 activation 都是 tensor。理解 tensor 的存储方式，是做内存核算的第一步。

一个 tensor 的显存占用由两个量决定：元素个数和每个元素的字节数。

```python
x = torch.zeros(4, 8)       # 默认 float32
x.numel()                   # 32 个元素
x.element_size()            # 4 bytes
内存 = 32 × 4 = 128 bytes
```

常见数值类型如下：

| 类型 | 字节/元素 | 特点 |
|---|---:|---|
| FP32 / float32 | 4 | 传统默认，稳定但慢、占显存 |
| FP16 / float16 | 2 | 省显存、快，但动态范围小，容易 underflow/overflow |
| BF16 / bfloat16 | 2 | 和 FP32 类似的指数范围，精度较粗，深度学习常用 |
| FP8 | 1 | H100 等新硬件支持，速度和显存优势大，但训练稳定性更难 |

FP16 和 BF16 都是 16 bit，但分配方式不同。FP16 给尾数更多 bit，动态范围较小；BF16 保留类似 FP32 的指数位，因此能表示很小或很大的数，更适合大模型训练。实际训练中，常见策略是：参数主副本和 optimizer state 用 FP32 保存，前向/反向中的矩阵乘法尽量用 BF16 或 FP8。

这就是混合精度训练的核心：不是全模型统一用一种 dtype，而是在稳定性和吞吐之间做局部取舍。

## 3. 设备位置与数据搬运

PyTorch 默认在 CPU 上创建 tensor：

```python
x = torch.zeros(32, 32)     # 在 CPU RAM
x = x.to("cuda")           # 搬到 GPU HBM
```

也可以直接在 GPU 上创建：

```python
x = torch.zeros(32, 32, device="cuda")
```

训练时要始终知道每个 tensor 在哪里。CPU RAM 到 GPU HBM 的搬运不是免费的，频繁搬运会让 GPU 等数据而不是做计算。工程代码里常常会加入断言或日志，例如检查 `x.device`，避免某个 batch、mask 或 loss target 意外留在 CPU 上。

## 4. Tensor 不是“数组本身”，而是对存储的视图

PyTorch tensor 本质上是指向底层 storage 的对象，并带有 shape、stride、offset 等元数据。一个二维连续矩阵的 stride 可能是 `(4, 1)`：行方向每走一步跳 4 个元素，列方向每走一步跳 1 个元素。

这解释了为什么很多操作几乎免费：切片、转置、view 等操作通常只改变元数据，不复制底层数据。

```python
x = torch.arange(6).view(2, 3)
y = x[0]       # view，共享 storage
z = x.T        # transpose，也通常共享 storage
```

但共享 storage 也有风险：如果你原地修改 `x`，`y` 看到的内容也会变。另一个常见坑是 contiguous。转置后的 tensor 往往不是连续存储的，某些 `view` 操作会失败，需要先：

```python
z = x.T.contiguous()
```

`contiguous()` 可能会真的复制数据，因此它不是免费的。写高性能代码时，要区分“只改视图”和“分配新内存”。

## 5. 维度命名：让张量代码更不容易错

真实模型里的 tensor 往往不只是二维矩阵，而是带有 batch、sequence、head、hidden 等多个维度。直接写 `transpose(-2, -1)`、`view(b, s, h, d)` 很常见，但长期维护时容易出错：`-1` 到底是 hidden 还是 head_dim？某个维度改了以后注释是否还正确？

课程推荐的思路是尽量给维度起名字。`einsum` 可以把矩阵乘法写成带维度语义的表达式。例如注意力分数可以理解为：

```python
scores = torch.einsum(
    "batch seq_q hidden, batch seq_k hidden -> batch seq_q seq_k",
    q, k,
)
```

没有出现在输出端的 `hidden` 会被求和，出现在输出端的 `batch、seq_q、seq_k` 会被保留。这样代码直接表达了“对 hidden 做内积，得到每个 query 和 key 的相似度”。

`einops.rearrange` 则适合做 reshape/transpose 的组合。例如把最后一维拆成多头：

```python
x = rearrange(x, "batch seq (heads dim) -> batch heads seq dim", heads=num_heads)
```

这类写法不一定改变底层成本，但能显著减少维度错误。对于教学和研究代码，可读性本身就是一种工程效率：越容易看出张量形状，越容易做资源核算，也越容易定位性能问题。

## 6. 矩阵乘法是深度学习的主成本

多数 elementwise 操作的 FLOPs 与 tensor 元素数线性相关，但大模型中真正主导计算的是矩阵乘法。

若做：

```text
[B, D] × [D, K] -> [B, K]
```

每个输出元素需要 D 次乘法和约 D 次加法，因此粗略 FLOPs 为：

```text
FLOPs ≈ 2 × B × D × K
```

这条规则非常重要：矩阵乘法的 FLOPs 约等于 2 乘以三个维度的乘积。

如果把 `B` 理解为 token 或数据点数，`D × K` 理解为参数量，那么线性层一次前向的计算量就是：

```text
forward FLOPs ≈ 2 × token 数 × 参数量
```

这个结论可以粗略推广到 Transformer：只要矩阵乘法主导，前向计算量就接近 `2 × tokens × parameters`。注意力的二次项、sequence length 和其他操作会带来修正，但纸上估算时这个近似很有用。

## 7. FLOPs 与 FLOPs/s：别混淆计算量和速度

“FLOPs”有时指 floating point operations，即总操作数；有时又被用作 floating point operations per second，即每秒吞吐。为避免混淆，可以写成：

```text
FLOPs      = 总浮点操作数
FLOPs/s    = 每秒浮点操作数
```

硬件厂商会给出峰值 FLOPs/s，例如 A100、H100 在 FP32、TF32、BF16、FP8 下的峰值都不同。低精度通常吞吐更高，FP8 高于 BF16，BF16 高于 FP32。但规格表常带有“结构化稀疏”假设，例如 2:4 sparsity；如果你的模型是 dense，就不能直接拿最高宣传数字。

实际训练时还要看 MFU：

```text
MFU = 模型有效 FLOPs/s / 硬件峰值 FLOPs/s
```

MFU 衡量模型把硬件“榨干”了多少。大矩阵乘法占比越高，MFU 越容易高；小 batch、碎操作、多通信、多数据搬运都会降低 MFU。实践中，MFU 超过 0.5 往往已经不错，只有几个百分点通常说明代码或并行方式存在严重瓶颈。

## 8. 自动求导与反向传播的成本

PyTorch 的 autograd 让我们不用手写梯度：

```python
pred = x @ w
loss = ((pred - y) ** 2).mean()
loss.backward()
w.grad        # PyTorch 自动填充
```

但自动求导不代表计算免费。仍以线性层为例：

```text
X: [B, D]
W: [D, K]
H = XW: [B, K]
```

前向计算 `H = XW` 的成本是：

```text
2 × B × D × K
```

反向时至少要算两类梯度：

```text
dL/dW = X^T × dL/dH
dL/dX = dL/dH × W^T
```

每个也都是一个矩阵乘法，各自约 `2 × B × D × K`。所以反向总成本约为：

```text
backward FLOPs ≈ 4 × B × D × K
```

因此一次完整训练 step 中，仅对主要矩阵乘法而言：

```text
forward + backward ≈ 6 × token 数 × 参数量
```

这就是开头大模型训练估算里那个系数 6 的来源：前向约 2 倍，反向约 4 倍。

## 9. 参数、初始化与 nn.Module

PyTorch 中可训练参数通常用 `nn.Parameter` 包装，并放进 `nn.Module`。例如一个简单深层线性网络可以写成：

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

初始化不能随便用标准正态。若 `W ~ N(0, 1)`，输入维度很大时，输出方差会随 fan-in 增长，激活值可能爆炸。常见做法是按输入维度缩放：

```python
w = torch.randn(d_in, d_out) / math.sqrt(d_in)
```

这与 Xavier/Glorot 初始化思想一致：让信号在层间传播时尺度尽量稳定。有时还会使用截断正态，避免极端大值。

## 10. Optimizer state 也是显存大头

训练时显存不只装参数。以 Adam/AdamW 为例，每个参数通常对应：

1. 参数本身
2. 梯度
3. 一阶动量 `m`
4. 二阶动量 `v`
5. 有时还有 FP32 master weights 或其他临时 buffer

如果这些主要以 FP32 保存，每个参数十几字节很常见。这就是为什么“模型参数量 × dtype 大小”远远低估训练显存。

对于更简单的 Adagrad，也需要为每个参数保存累计平方梯度。optimizer 的 `step()` 大致逻辑是：读取 `p.grad`，更新状态，再原地更新参数。状态会跨 step 保留，所以它是长期显存占用，而不是临时变量。

## 11. Activation：为什么前向中间结果要保留

反向传播需要用到前向产生的中间 activation。例如计算第一层权重梯度时，需要该层输入 activation。因此 autograd 默认会保存许多中间结果。

简单深层线性模型中，如果 batch 为 `B`，宽度为 `D`，层数为 `L`，activation 数量粗略为：

```text
activations ≈ B × D × L
```

总显存可以按类别拆开估：

```text
总内存 ≈ bytes_per_elem × (参数 + 梯度 + optimizer state + activation)
```

对于 Transformer，activation 还会受 sequence length、attention 矩阵、MLP 中间维度等影响。若显存不够，一个重要技巧是 activation checkpointing：不保存所有 activation，反向时重新计算一部分，用更多计算换更少内存。

## 12. 数据读取与训练循环

语言模型数据通常是 tokenizer 输出的整数序列。真实语料可能达到 TB 级，不能全部读入内存。常用方式是 `numpy.memmap`：让数组映射到磁盘文件，按需读取片段。

典型训练循环如下：

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

工程上还必须周期性 checkpoint：保存 model state、optimizer state、当前 step、随机数状态等。训练大模型一定会遇到中断、抢占、OOM 或节点故障，不能假设一次运行到结束。恢复训练时还要确认学习率调度器、数据迭代位置和随机种子是否一致，否则同一个实验可能悄悄变成另一条训练曲线。对研究来说，可复现性是比较方法的前提；对工程来说，它也是节省集群时间和排查线上故障的前提。

## 13. 计算、内存与带宽要一起看

做资源核算时，不能只数 FLOPs。GPU 训练通常同时受三类资源限制：计算单元、显存容量和内存/互联带宽。显存容量决定模型和 batch 能不能放下；FLOPs/s 决定理想情况下矩阵乘法能多快；带宽决定数据在 HBM、cache、CPU、GPU 以及多卡之间移动得多快。

一个操作如果做了很多乘加、但读写数据相对少，通常是 compute-bound，典型例子是大矩阵乘法。相反，如果一个操作只是逐元素加法、mask、copy、reshape 后触发 contiguous 复制，计算量很小但要读写大量数据，就可能是 memory-bound。此时即使 FLOPs 很少，也会拖慢训练，因为 GPU 核心在等数据。

多卡训练还会引入通信带宽问题。例如数据并行需要同步梯度，张量并行需要在层内交换中间结果，流水并行需要传递 activation。模型越大，通信越不能被忽略。很多系统优化的目标不是减少数学计算，而是让计算和通信重叠、减少不必要的数据复制、让矩阵形状更适合 Tensor Core。换句话说，好的训练系统要让“昂贵的 FLOPs”尽量发生在大而规整的矩阵乘法里，而不是浪费在碎片化 kernel、跨设备搬运和临时拷贝上。

## 14. 从研究代码到工程成本意识

这节课最重要的观念是：写模型时要同时写“成本账本”。看到一个模型，不仅要问 loss 能不能降，还要问：

- 参数量是多少？
- 每个参数训练时实际占多少字节？
- activation 随 batch size 和 sequence length 怎么增长？
- 主要矩阵乘法 FLOPs 是多少？
- 用 BF16/FP8 后吞吐是否真的提高？
- MFU 是 50% 还是 5%？
- 是否被 CPU/GPU 搬运、非 contiguous copy、小 kernel 或通信拖慢？

课程用简单线性模型推导，是为了让公式透明：前向约 `2 × tokens × params`，反向约 `4 × tokens × params`，训练约 `6 × tokens × params`；显存则由参数、梯度、optimizer state、activation 四部分组成。到了 Transformer，这些账更复杂，但基本方法不变。

真正的大模型工程不是“写出数学上正确的网络”就结束了，而是要让代码、数值精度、硬件吞吐和训练成本一起成立。PyTorch 给了我们自动求导和模块化抽象，但高效训练要求我们看穿这些抽象：知道 tensor 在哪里、是否复制、dtype 是什么、矩阵乘法有多大、反向要多花多少、optimizer 会额外保存什么。只有具备这种资源核算意识，研究原型才可能走向可扩展、可负担、可复现的训练系统。
