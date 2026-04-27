# Stanford CS336 Language Modeling from Scratch 2025 中文完整教程


来源：Stanford CS336 2025 YouTube playlist transcripts。本文是教程化整理版，不是逐字稿。


---


# CS336 2025 第 1 讲教程：课程概览与 Tokenization（分词/标记化）

> 本文根据 Stanford CS336 2025 Lecture 1 transcript 整理而成。写法是中文教程/讲义，不是逐字稿；英文术语会尽量保留，并给出中文解释。

## 学习目标

学完本讲，你应该能够：

1. 说明 CS336「Language Models from Scratch（从零构建语言模型）」关注的不是只会调用 API，而是理解并实现语言模型构建管线。
2. 理解课程的核心问题：在给定 compute（计算资源）和 data（数据）预算下，如何训练出最好的模型。
3. 认识语言模型从 tokenizer（分词器）、Transformer 架构、训练、系统优化、scaling laws（缩放定律）、数据、评估到 alignment（对齐）的完整流程。
4. 理解 tokenization（分词/标记化）的作用：把 Unicode 字符串转换成整数序列，再让模型处理。
5. 掌握 BPE（Byte Pair Encoding，字节对编码）的基本思想、训练过程和编码/解码逻辑。

## 先修知识

建议具备：

- Python 与 PyTorch 基础。
- 基本机器学习概念：loss（损失）、optimizer（优化器）、batch size（批大小）、overfitting（过拟合）。
- 对 Transformer、attention（注意力机制）有初步印象即可；本课会从底层实现逐步展开。
- 对 GPU、并行计算不要求精通，但需要愿意从工程视角理解性能瓶颈。

## 本讲地图

本讲分两大部分：

1. 课程总览：为什么要从零构建语言模型、课程会覆盖哪些模块、工程与研究视角如何结合。
2. Tokenization 入门：为什么需要分词器，字符级、字节级、词级方案的问题，以及 BPE 如何折中解决。

课程的主线可以概括为：

```text
原始数据 → 数据清洗与过滤 → tokenizer → 整数序列 → Transformer → 训练 → 评估 → 系统优化 → 对齐/微调 → 可用模型
```

## 一、为什么要从零构建语言模型？

课程开头强调了一个现象：研究者与底层技术越来越远。早些年，AI/NLP 研究者通常会自己实现和训练模型；后来大家下载 BERT 这类模型做 fine-tuning（微调）；现在很多工作甚至只需要 prompting（提示词调用）一个闭源模型。

这当然带来了便利，但抽象层也会“漏水”：语言模型 API 看似只是“字符串输入、字符串输出”，但如果不知道底层的数据、模型、系统和训练机制，就很难做真正基础性的研究。CS336 的理念是：

```text
To understand it, you have to build it.
要理解它，就必须亲手构建它。
```

不过，课程也很现实：frontier models（前沿大模型）通常需要巨额资本、庞大 GPU 集群和未公开的工程细节。课堂无法让每个人训练一个 GPT-4 级别模型。因此课程会训练小模型，但强调要清楚小规模实验能教会我们什么、不能教会我们什么。

课程区分了三类知识：

- mechanics（机制）：Transformer 如何实现、GPU 并行如何工作。这部分可以扎实教授。
- mindset（思维方式）：始终从效率和规模化角度思考，尽量榨干硬件性能。
- intuitions（直觉）：哪些数据与架构选择在大规模上有效。这部分只能部分学习，因为小规模规律未必代表大规模。

## 二、核心视角：效率，而不是盲目堆规模

讲师提醒，不要把 bitter lesson（苦涩教训）误解成“只有规模重要，算法不重要”。更准确的说法是：

```text
Algorithms at scale matter.
在规模化条件下有效的算法才重要。
```

模型效果可以粗略理解为资源投入与效率共同作用的结果。资源越昂贵，效率越关键；如果一次训练要花巨大成本，就不能像本地小实验那样反复浪费。算法效率、硬件利用率、数据质量、模型结构都会影响最终结果。

因此，本课反复追问的问题是：

```text
给定计算预算和数据预算，能训练出的最佳模型是什么？
```

这也是工程视角的核心：不是只问“模型能不能更大”，而是问“每一单位 FLOP、每一块 GPU、每一个 token 是否被有效使用”。

## 三、语言模型发展的简要背景

语言模型并不是近几年才出现。早期 Shannon 就用语言模型估计英文熵；传统 NLP 中，语言模型常作为机器翻译、语音识别等系统的组件。后来深度学习时代逐渐积累了关键技术：

- neural language model（神经语言模型）
- seq2seq（序列到序列模型）
- Adam optimizer（Adam 优化器）
- attention mechanism（注意力机制）
- Transformer
- model parallelism（模型并行）
- foundation models（基础模型）如 ELMo、BERT、T5

GPT-2、GPT-3 之后，scaling laws（缩放定律）和工程化训练成为核心趋势。与此同时，模型开放程度也分层：

- closed models（闭源模型）：只能通过 API 使用。
- open-weight models（开放权重模型）：权重可用，但数据和训练细节可能不完整。
- open-source models（开源模型）：尽量开放权重、数据和实现，但论文也无法替代亲手构建。

## 四、课程五大模块

### 1. Basics（基础管线）

目标是实现一个最小但完整的语言模型训练流程：

- tokenizer（分词器）：字符串与整数序列之间转换。
- model architecture（模型架构）：主要是 Transformer。
- training（训练）：loss、optimizer、learning rate schedule、训练循环。

作业中会实现 BPE tokenizer、Transformer、cross-entropy loss（交叉熵损失）、AdamW optimizer、training loop。课程允许使用 PyTorch，但不会直接使用现成 Transformer 实现。

### 2. Systems（系统优化）

训练模型不只是数学公式，也关乎硬件。GPU 的算力在芯片上，显存常在芯片外，数据搬运可能成为瓶颈。课程会讨论：

- kernels（算子内核）：如矩阵乘法如何用 tiling（分块）、fusion（融合）减少数据移动。
- Triton：用于编写高性能 GPU kernel 的工具。
- parallelism（并行）：data parallelism（数据并行）、tensor/model parallelism（张量/模型并行）等。
- inference（推理）：模型实际生成 token 的过程。

推理分为：

- prefill：处理 prompt，所有输入 token 已知，可并行，类似训练。
- decode：自回归逐个生成 token，常更难充分利用 GPU，容易 memory-bound（受内存带宽限制）。

课程还提到 speculative decoding（推测解码）：用较便宜的小模型先生成候选，再让大模型并行验证，从而加速推理。

### 3. Scaling Laws（缩放定律）

核心问题：给定 FLOPs 预算，模型参数量和训练 token 数该如何平衡？大模型能看更少数据，小模型能看更多数据，最优点在哪里？

课程提到 Chinchilla optimal（Chinchilla 最优）相关思想：通过小规模实验拟合规律，再预测大规模设置下的最佳参数与 loss。它的价值在于用较少计算指导更昂贵的训练决策。

补充说明：讲中提到一个经验规则，即模型参数量与训练 token 数之间存在常用比例关系；这类规则有适用前提，且不包含推理成本，因此不能机械套用。

### 4. Data（数据与评估）

模型能力很大程度由数据决定：训练多语言数据会带来多语言能力，训练代码数据会带来代码能力。常见数据来源包括 Web/Common Crawl、Wikipedia、GitHub、StackExchange、书籍、论文等。

但“把互联网喂给模型”是误导性说法。原始网页常常是 HTML、PDF、代码仓库或垃圾内容，必须经过处理：

- extraction（抽取）：从 HTML/PDF 等转成文本。
- filtering（过滤）：去除低质量、有害或无关内容。
- deduplication（去重）：删除重复数据，避免浪费训练预算。
- legal considerations（法律问题）：哪些数据可以训练也需要讨论。

评估方面会涉及：

- perplexity（困惑度）：衡量语言模型预测下一个 token 的能力。
- standardized benchmarks（标准化测试），如 MMLU。
- instruction following evaluation（指令跟随评估）。
- agentic system evaluation（评估包含语言模型的完整系统）。

### 5. Alignment（对齐）

预训练得到的 base model（基础模型）主要会预测下一个 token，具有“原始潜力”，但不一定会听指令。alignment（对齐）是让模型更有用、更安全、更符合交互需求的过程。

课程中概括了对齐的几个目标：

- instruction following（遵循指令）
- style control（控制回答风格，如长短、项目符号、语气）
- safety（安全拒答有害请求）

常见阶段包括：

- SFT, supervised fine-tuning（监督微调）：用 user/assistant 的 prompt-response 对进行监督学习。
- learning from feedback（从反馈学习）：用偏好数据或 verifier（验证器）改进模型。
- PPO、DPO、GRPO：不同的强化学习或偏好优化算法，其中 DPO 适合偏好数据，GRPO 是 DeepSeek 使用过的一类简化 PPO 的方法。

## 五、Tokenization：为什么需要分词器？

语言模型本质上处理数字张量，而原始文本是 Unicode 字符串。tokenization（分词/标记化）就是把字符串转换成整数序列，并且最好能 decode（解码）回原字符串。

一个 tokenizer 需要两个方向：

```text
encode: string → list[int]
decode: list[int] → string
```

vocabulary size（词表大小）指 token 可以取多少种整数值。词表越大，单个 token 往往能表示更长片段，但模型输入/输出层也会更大；词表越小，序列会更长，attention 成本会增加。

一个重要细节：现代 tokenizer 通常是可逆的，空格也会被编码进 token。例如 “hello” 和 “ hello” 可能是两个不同 token。这和传统 NLP 中简单按空格切词不同。

## 六、几种朴素 tokenization 方法及问题

### 1. Character-based tokenization（字符级分词）

把每个 Unicode 字符映射到 code point（码点）。例如英文字符和 emoji 都有对应整数。

问题：

- Unicode 码点范围很大，词表利用不均衡。
- 很多字符极少出现，却占用词表空间。
- 压缩效率不理想。

### 2. Byte-based tokenization（字节级分词）

先用 UTF-8 把字符串转成 bytes（字节），每个字节取值 0 到 255，因此词表很小，且任意文本都能表示。

优点：简单、优雅、没有未知字符问题。

问题：序列太长。每个 token 只表示一个字节，compression ratio（压缩率，这里指每 token 表示多少字节）很低。由于 attention 朴素实现对序列长度是二次复杂度，字节级序列会非常低效。

### 3. Word-based tokenization（词级分词）

按空格、正则表达式或预分词规则切成词/片段，再给每个词分配整数。

优点：常见词可以用一个 token 表示，序列更短。

问题：词表可能非常大，而且总会遇到未见过的新词、拼写、专有名词、代码片段等，只能用 UNK（unknown token，未知词）处理，导致信息丢失和评估麻烦。

## 七、BPE：Byte Pair Encoding（字节对编码）

BPE 是一种很老的数据压缩算法，后来被引入神经机器翻译，再被 GPT-2 等语言模型采用。它的核心思想是：不要手工决定什么是“词”，而是从语料统计中学习 token。

直觉如下：

- 高频连续片段应该合并成一个 token，以提高压缩率。
- 低频片段可以拆成多个 token，不必浪费词表。
- 从 bytes 开始，可以保证任何字符串都能表示。

BPE 训练过程：

```text
输入：训练语料，目标词表大小或合并次数
初始化：把文本转成字节序列，初始词表为 0..255
重复：
  1. 统计当前序列中所有相邻 token pair 的出现次数
  2. 找到出现次数最多的 pair，例如 (116, 104)
  3. 为这个 pair 分配一个新的 token id，例如 256
  4. 在训练序列中把所有该 pair 替换成新 token
输出：merge rules（合并规则）与 vocabulary（词表）
```

编码新文本时：

```text
1. 将字符串转成 bytes
2. 按训练时学到的 merge rules 的顺序依次应用合并
3. 得到整数 token 序列
```

解码时：

```text
1. 将每个 token id 映射回对应 byte 序列
2. 拼接 bytes
3. 用 UTF-8 解码回字符串
```

GPT-2 风格的 tokenizer 还会先做 pre-tokenization（预分词）：用正则表达式把文本切成若干片段，再在片段内部运行 BPE。这样做主要是工程效率和行为控制上的折中。

## 八、常见误区

1. “从零训练模型是所有问题的第一步。”
   不是。讲师明确说，如果 prompting 或 fine-tuning 能解决问题，就应该优先使用它们。从零训练适合学习底层机制或确实需要新基础模型的场景。

2. “小模型实验结论一定能推广到大模型。”
   不一定。attention 与 MLP 的 FLOPs 占比、emergent behavior（涌现行为）等都会随规模改变。

3. “tokenization 只是文本预处理小细节。”
   不是。tokenizer 直接影响序列长度、训练效率、词表大小、可逆性和多语言/代码处理效果。

4. “字节级 tokenizer 最干净，所以一定最好。”
   字节级确实优雅，但在当前主流 Transformer 架构下通常序列过长、计算低效。

5. “互联网数据可以直接训练。”
   不行。原始 Common Crawl 等数据包含大量垃圾、重复、HTML/PDF 结构和法律/安全问题，需要认真处理。

## 九、实践练习

1. 打开任意 tokenizer 可视化工具，输入英文句子、中文句子、数字、代码和 emoji，观察 token 边界是否符合直觉。
2. 手写一个最小 byte tokenizer：实现 `encode(str)->list[int]` 与 `decode(list[int])->str`。
3. 用一个很小语料手动执行 3 次 BPE merge，记录每次最高频 pair 和新 token id。
4. 比较同一段文本在字符级、字节级、BPE 下的 token 数，思考序列长度如何影响 attention 成本。
5. 随机抽样一些网页文本，尝试判断哪些是高质量内容、哪些需要过滤或去重。

## 十、总结

本讲建立了 CS336 的总体框架：语言模型不是一个孤立的 Transformer，而是一条端到端工程管线。课程强调从底层构建 tokenizer、模型、训练循环、系统优化、数据流程、评估和对齐方法，并始终围绕效率展开：在有限计算和数据下，如何得到最好的模型。

Tokenization 是这条管线的入口。字符级、字节级、词级方案各有明显缺陷；BPE 通过从字节出发、不断合并高频相邻 token，在可表示性与压缩效率之间取得实用折中。虽然未来可能出现成熟的 tokenizer-free（无分词器）架构，但在当前主流前沿模型实践中，BPE 及其变体仍是重要基础。

## 延伸阅读与下一讲衔接

- Andrej Karpathy 关于 tokenization 和从零构建模型的视频。
- Transformer 原论文：Attention Is All You Need。
- GPT-2 tokenizer 与 byte-level BPE 实现。
- Chinchilla scaling laws 相关论文。

下一讲将进入 PyTorch 细节和 resource accounting（资源核算）：不仅要写出能运行的程序，还要追踪 FLOPs、内存和数据移动，理解计算资源究竟花在了哪里。


---


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


---


# CS336 2025 第 3 讲中文教程：Transformer 架构与超参数

本讲的主题是：如果你真的要从零训练一个语言模型，除了“Transformer 是什么”之外，还必须知道一大堆看似琐碎、但会直接影响训练稳定性、吞吐和最终效果的工程选择。现代大语言模型的核心架构并没有完全脱离 Transformer，但和 2017 年原始论文里的版本相比，已经形成了一套更实用的“共识配方”：pre-norm、RMSNorm、无 bias、RoPE、SwiGLU、合适的宽深比、以及若干稳定性技巧。

下面按组件梳理这些设计选择，并给出训练新模型时可直接参考的经验法则。

## 1. 从原始 Transformer 到现代 LLM Transformer

原始 Transformer block 大致由以下部分组成：

1. token embedding 与位置编码；
2. multi-head self-attention；
3. residual connection；
4. layer normalization；
5. feed-forward network，也就是 MLP；
6. 最后接输出 softmax。

但现代 LLM 通常不是完全照搬原始版本。一个更接近 LLaMA 系列和课程作业实现的 block 通常是：

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

并且常见配置包括：

- normalization 放在子层之前，也就是 pre-norm；
- normalization 多用 RMSNorm，而非传统 LayerNorm；
- 线性层通常不使用 bias；
- 位置编码多用 RoPE；
- MLP 多用 SwiGLU 或其他 GLU 变体；
- 有些新模型还会在子层输出后再加一次 norm，形成“双 norm”结构。

理解这些变化的关键，不是把它们当成魔法，而是把它们放在两个目标下看：第一，训练更稳定；第二，在 GPU 上更高效。

## 2. Residual 与 normalization：稳定训练的主轴

### 2.1 Post-norm 与 pre-norm

原始 Transformer 使用 post-norm：先做 attention 或 MLP，再加 residual，然后做 LayerNorm。形式类似：

```text
x = Norm(x + Attention(x))
x = Norm(x + MLP(x))
```

现代 LLM 基本转向 pre-norm：

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

这看起来只是移动了一个 normalization 的位置，但影响很大。Residual stream 的价值在于提供一条接近 identity 的路径，让梯度能从高层顺畅传播到底层。如果把 norm 放在 residual stream 中间，就会干扰这条直接路径。实践中，post-norm 更容易出现梯度爆炸、loss spike，对 warmup 和学习率更敏感；pre-norm 通常更稳定，也更容易训练深层模型。

因此，一个很重要的现代经验是：不要随意破坏 residual stream 的 identity 连接。norm 应该主要放在非 residual 分支的入口或出口，而不是把整条 residual 主干反复归一化。

### 2.2 LayerNorm 与 RMSNorm

传统 LayerNorm 会对每个 token 的 hidden vector 做：减均值、除标准差，再乘可学习缩放参数 gamma，并加 bias beta。RMSNorm 则更简单：不减均值，也通常不加 beta，只按 root mean square 缩放。

为什么 RMSNorm 流行？主要原因是：

- 效果通常不差于 LayerNorm；
- 操作更少；
- 参数更少；
- 更重要的是，减少了内存读取和写入。

在 Transformer 中，大部分 FLOPs 来自矩阵乘法，但这不代表其他操作不重要。softmax、normalization 这类操作 FLOPs 占比很小，却可能占据相当可观的运行时间，因为它们受内存移动限制。RMSNorm 的优势不只是少算一点，而是少搬一点数据。

### 2.3 无 bias 线性层

现代 LLM 的线性层常常去掉 bias，包括 attention projection 和 MLP projection。经验上，这通常不会损害效果，还能减少参数和内存访问。一些报告还指出，去掉 bias 有助于优化稳定性，尤其在大模型训练中更明显。

总结这一节：现代 Transformer 的 normalization 设计服务于稳定性与效率。pre-norm 保持 residual 主干畅通，RMSNorm 简化归一化，无 bias 降低额外状态和潜在不稳定因素。

## 3. MLP 与激活函数：为什么 SwiGLU 成为默认选择

Transformer block 中除了 attention，另一个大头就是 MLP。早期 Transformer 使用 ReLU，GPT 系列广泛使用 GELU，而许多现代模型使用 GLU 变体，尤其是 SwiGLU。

普通 MLP 可以写成：

```text
MLP(x) = W2 * activation(W1 * x)
```

GLU 类结构多引入一条 gate 分支：

```text
MLP(x) = W2 * (activation(W1 * x) ⊙ (V * x))
```

其中 `⊙` 是逐元素乘法。直观上，模型不仅生成一组 hidden features，还学习一个门控向量，决定哪些维度应该通过、哪些应该抑制。

SwiGLU 使用 Swish 作为非线性：

```text
swish(x) = x * sigmoid(x)
```

大量模型和消融实验表明，GLU 变体往往比 ReLU/GELU MLP 有稳定的小幅收益。这个收益不一定是“没有就训练不好”，例如 GPT-3 并未使用 SwiGLU，仍然是很强的模型；但如果你在设计新模型，SwiGLU 已经是一个很稳妥的默认选择。

需要注意的是，GLU 多了一组投影参数 V。为了让参数量和普通 MLP 大致匹配，通常会把中间维度缩小到原来的 2/3。如果普通 MLP 设 `d_ff = 4 * d_model`，那么 SwiGLU 常用：

```text
d_ff ≈ 8/3 * d_model
```

这就是许多 LLaMA-like 模型中 MLP hidden size 看起来不是 4 倍而是约 2.6 到 2.7 倍的原因。

## 4. Attention 与位置编码：RoPE 的现代地位

语言模型需要知道 token 的顺序。早期方法包括 sinusoidal position embedding、learned absolute position embedding、relative position bias 等。近年的 dense LLM 几乎都收敛到 RoPE，也就是 rotary position embedding。

RoPE 的核心思想是：attention 关心的往往不是绝对位置，而是相对距离。对于 query 和 key，如果位置整体平移，但两者相对距离不变，那么它们的内积关系也应尽量保持一致。

RoPE 用旋转实现这个目标。它不是在输入 embedding 底部加一个位置向量，而是在每一层 attention 中，对 query 和 key 做位置相关的旋转。位置越靠后，旋转角度越大；不同维度对使用不同频率，从而同时表达近距离和远距离信息。

在二维中可以直观理解：两个向量如果都旋转同样角度，它们的相对夹角不变，内积也不变。RoPE 将高维向量拆成多个二维对子，对每一对维度按固定频率旋转。这样 query-key 内积天然编码相对位置。

RoPE 流行的原因包括：

- 相对位置建模更自然；
- 在小上下文和大上下文中效果都好；
- 有许多上下文长度外推和扩展技巧；
- 已被大量现代模型验证。

实践中要记住：RoPE 作用在 Q 和 K 上，而不是简单加到 token embedding 上；rotation 的频率通常是固定 schedule，不是训练出来的参数。

## 5. Attention 的推理效率：MHA、MQA 与 GQA

标准 multi-head attention 中，每个 head 都有自己的 Q、K、V。训练时我们通常一次处理完整 batch 和完整序列，有大矩阵乘法，GPU 利用率较好。但推理时是自回归生成：一次生成一个 token。为了避免重复计算过去 token 的 K 和 V，系统会维护 KV cache。

问题是：上下文越长，KV cache 越大；每生成一个 token，都要从显存中读大量历史 K/V。此时瓶颈往往不是算力，而是内存带宽。

MQA，multi-query attention，做了一个激进简化：保留多个 query head，但所有 head 共享一组 K 和 V。这样 KV cache 大幅减少，推理速度和长上下文能力更好。

GQA，grouped-query attention，是折中方案：多个 query head 分成若干组，每组共享一组 K/V。它比 MQA 表达能力更强，又比标准 MHA 更省 KV cache。许多现代大模型采用 GQA，因为它在质量和推理成本之间更平衡。

因此，attention head 的设计不只是训练问题，更是部署问题。模型发布后，大量成本来自推理；GQA/MQA 的价值主要体现在降低推理内存访问和提升吞吐。

## 6. 关键超参数的经验法则

### 6.1 MLP 中间维度

如果使用普通 ReLU/GELU MLP，一个经典选择是：

```text
d_ff = 4 * d_model
```

如果使用 SwiGLU/GeGLU 等 gated MLP，为了参数量接近，常用：

```text
d_ff ≈ 8/3 * d_model
```

Kaplan 等 scaling law 工作中的消融显示，MLP ratio 在一个相当宽的范围内都还可以，但 4 倍附近是合理默认值。T5 曾使用非常夸张的 64 倍 d_ff，说明规则不是铁律；但后续 T5 v1.1 又回到更标准的 GLU ratio，说明常规默认值仍然很有竞争力。

### 6.2 Attention head 维度

常见做法是让：

```text
d_model = n_heads * d_head
```

也就是说，增加 head 数时，不让 attention 总维度无限增长，而是把 `d_model` 切成多个 head。大多数 GPT、PaLM、LLaMA 类模型都接近这个 1:1 设定。理论上，head 维度太小可能造成低秩瓶颈，但实践中这一默认设置表现良好。

### 6.3 宽深比

模型容量可以通过变宽，也可以通过变深。宽度通常由 `d_model` 控制，深度由 layer 数控制。经验上，许多模型落在类似：

```text
d_model / n_layers ≈ 100 到 128
```

的范围。这个比例不是定律，但 Kaplan 等实验显示，在多个参数规模下，宽深比的最优区域变化不算剧烈。

系统因素也会影响宽深比。更深的模型适合 pipeline parallel，把不同层切到不同设备；更宽的模型适合 tensor parallel，把大矩阵切到多个 GPU。也就是说，超参数不只由 loss 决定，还会被集群网络、并行策略和显存限制影响。

### 6.4 词表大小

早期英文模型常用 30k 到 50k token 词表。现代生产模型，尤其多语言模型，常用 100k 到 250k 甚至更大的词表。

更大的词表带来几个好处：

- 多语言文本被切成更少 token；
- 推理成本对低资源语言更友好；
- emoji、代码、特殊符号等覆盖更好；
- 大模型通常能更好利用大词表。

如果只训练英文小模型，较小词表仍可行；如果目标是通用、多语言、面向生产的模型，大词表已经成为趋势。

## 7. Dropout、weight decay 与训练稳定性

预训练和传统监督学习不同：数据巨大，通常训练不到完整多 epoch，因此过拟合不是主要矛盾。这解释了为什么 dropout 在现代 LLM 预训练中逐渐不流行。

但 weight decay 仍然常见。它在这里的作用也不完全是传统意义上的“防止过拟合”。实验观察表明，weight decay 会和学习率 schedule，尤其 cosine decay，产生复杂交互：高学习率阶段可能看起来训练较慢，但当学习率逐渐下降时，带 weight decay 的模型可能快速改善，最终得到更好的训练 loss 和验证 loss。

所以在 LLM 预训练里，weight decay 更像是一种优化动力学工具，而不仅是正则化工具。

## 8. 大模型训练稳定性：softmax 是重点风险区

随着模型变大、训练更久，loss spike、gradient norm spike 会越来越重要。现代架构改进中，一个明显趋势是围绕 softmax 做稳定性处理，因为 Transformer 中有两个关键 softmax：

1. 输出层 vocabulary softmax；
2. attention 中的 softmax。

### 8.1 输出 softmax 的 z-loss

输出 softmax 会计算：

```text
p(x) = exp(logit_x) / Z
```

其中 `Z` 是所有词表项 exponentiated logits 的和。若 Z 数值过大或不稳定，softmax 会带来数值问题。z-loss 的思想是加入一个辅助项，鼓励 `log Z` 接近 0，也就是让 normalizer 接近 1。

PaLM 使用过这种技巧，后续一些模型也采用它来稳定训练。它的目的不是提升表达能力，而是让输出 softmax 的数值范围更可控。

### 8.2 Attention softmax 的 QK norm

attention softmax 的输入来自 QK 内积。如果 query/key 的范数过大，logits 会变得很极端，softmax 可能饱和或产生不稳定梯度。QK norm 的做法是在 Q 和 K 进入内积前，对它们做 normalization。

这相当于直接控制 softmax 的输入尺度。它最早在 vision transformer 和 multimodal 训练稳定性中很有用，后来被一些文本 LLM 吸收。一个值得注意的现象是：normalization 在现代模型里不断扩展位置，从 block 前 norm，到子层后 norm，再到 Q/K norm，说明“控制激活尺度”是训练大模型的核心手段之一。

### 8.3 Logit soft capping

另一种方法是对 attention logits 做 soft cap，例如用 tanh 把过大的 logits 平滑限制在某个范围内。Gemma 2 等模型使用过类似技巧。它能控制极端值，但不一定总是提升效果；一些实验中，QK norm 比 soft capping 更稳妥。

## 9. 长上下文 attention：局部窗口与稀疏结构

完整 self-attention 的成本随序列长度平方增长。为了支持更长上下文，模型可以采用结构化 attention，例如：

- sliding window attention：每层只看附近窗口；
- sparse attention：设计局部和跨块连接；
- 周期性 full attention：不是每层都做全局 attention，而是隔几层做一次。

近期一些模型采用混合结构：例如每四个 block 中，某一层做 full attention，且不使用位置编码；其他层做带 RoPE 的 sliding window attention。这样有两个好处：

1. 大多数层只处理局部窗口，系统成本可控；
2. 超长距离信息通过无位置编码的 full attention 传播，减少 RoPE 长度外推压力。

这类设计说明，长上下文能力不只是“把 RoPE 拉长”，还涉及 attention pattern、位置编码和系统成本之间的联合设计。

## 10. 实用默认配置

如果你要训练一个标准 dense decoder-only LLM，可以从以下配置开始：

- block：pre-norm Transformer；
- norm：RMSNorm；
- linear：默认无 bias；
- position：RoPE，作用于 Q/K；
- MLP：SwiGLU；
- MLP ratio：约 `8/3 * d_model`；
- attention：训练小模型可用 MHA，面向推理部署优先考虑 GQA；
- head 维度：满足 `d_model = n_heads * d_head`；
- 宽深比：参考 `d_model / n_layers ≈ 100-128`；
- dropout：大规模预训练通常可不用或很小；
- weight decay：保留，并和学习率 schedule 一起调；
- 稳定性：关注 gradient norm、loss spike，可考虑 z-loss、QK norm、额外 norm 或 logit soft cap。

## 小结

这节课的核心结论是：现代 LLM 架构不是由单个突破决定的，而是许多经验选择逐渐收敛的结果。pre-norm 和干净的 residual stream 让深层网络更容易训练；RMSNorm、无 bias 和 GQA 体现了对内存移动和推理成本的重视；SwiGLU、RoPE 和合理的超参数比例提供了稳定有效的默认性能；z-loss、QK norm 等技巧则解决大规模训练中越来越突出的数值稳定性问题。

如果只记住一句话：训练 Transformer 不是简单堆层数和参数量，而是在架构、超参数、优化动力学和硬件效率之间做一组相互配合的选择。现代 LLM 的“默认配方”之所以重要，是因为它们已经被许多大规模训练实验验证过，能让你少踩很多昂贵的坑。


---


# Stanford CS336 第 4 讲教程：Mixture of Experts（MoE）

## 学习目标

学完本讲，你应该能够：

1. 解释 Mixture of Experts（MoE，专家混合）相对 dense model（稠密模型）的核心优势：用接近相同的激活计算量获得更多总参数。
2. 理解 router / gating（路由器/门控）、expert（专家）、top-k routing（Top-K 路由）、load balancing（负载均衡）等关键术语。
3. 写出 MoE 前向传播的基本公式与伪代码。
4. 分析 MoE 在训练和推理中的成本：FLOPs、显存、通信、负载不均、token dropping。
5. 理解现代 MoE 系统（如 DeepSeek、Mixtral、Grok、Llama 4）常见的工程权衡。

## 先修知识

本讲默认你已经了解：Transformer 的基本结构，尤其是 self-attention 和 FFN/MLP；softmax、top-k、残差连接（residual connection）；语言模型训练中的 next-token prediction；以及基本的 GPU/TPU 并行训练概念。

## 本讲地图

MoE 的主线可以概括为一句话：把 Transformer 中昂贵的 FFN 层替换成多个“专家”FFN，并让每个 token 只激活其中少数几个专家。

本讲依次讨论：

- 为什么 MoE 在相同 FLOPs 下通常优于 dense model；
- router 如何为 token 选择 expert；
- expert 应该多大、多少个，是否需要 shared expert；
- 为什么 load balancing 是训练 MoE 的关键；
- MoE 的系统代价：通信、显存、并行、token dropping；
- DeepSeek 系列如何把这些想法组合成现代大规模 MoE 架构。

## 核心概念

### 1. Dense model 与 MoE 的区别

在普通 Transformer 中，每一层通常包含 attention 和 FFN。dense model 的 FFN 是一个固定的大 MLP：每个 token 都经过同一个 FFN。

MoE 把这个 FFN 替换成多个 expert。每个 expert 通常也是一个 FFN，但每个 token 不会经过所有 expert，而是由 router 选择其中 K 个。若每个 token 只激活 1 或 2 个 expert，则计算量主要取决于“activated parameters”（被激活参数），而不是模型总参数。

因此，MoE 的优势是：

- 总参数更多：模型有更大容量，可以记忆和表达更多模式；
- 激活参数较少：每个 token 只用一小部分参数，FLOPs 不随专家总数线性增长；
- 天然适合 expert parallelism（专家并行）：不同 expert 可放在不同设备上。

但它也带来复杂性：路由是离散选择，不易优化；不同 expert 负载可能严重不均；跨设备发送 token 会产生通信开销。

### 2. Expert 并不一定是“语义专家”

“Mixture of Experts”这个名字容易误导。它并不保证一个专家负责代码、一个负责数学、一个负责中文。expert 更准确地说是若干可被稀疏激活的子网络。它们可能形成某种 specialization（专门化），但这种专门化通常不是人工可解释的领域划分。

更好的心智模型是：MoE 在每一层提供了多条可选的非线性变换路径。router 根据当前 hidden state 选择其中几条路径。由于 hidden state 已经包含上下文、位置、前面层的计算结果，所以同一个表面 token 在不同语境下也可能被送往不同 expert。比如 “Python” 在编程上下文和动物上下文中，路由结果可能不同；但这并不意味着某个 expert 可以被简单命名为“编程专家”。

### 3. Router / Gating

router 是一个很轻量的模块，输入 token 的 hidden state，输出该 token 对每个 expert 的 affinity score（亲和分数）。常见做法是一个线性投影加 softmax 或 sigmoid，然后选择分数最高的 K 个 expert。

router 的输出也常被称为 gating weights（门控权重），用于加权组合多个 expert 的输出。

### 4. Top-K Routing

现代大规模 MoE 基本收敛到 token choice top-k routing：每个 token 自己选择分数最高的 K 个 expert。

也存在其他方案：

- expert choice：每个 expert 选择自己要处理的 token，天然负载均衡，但可能不是 token 的最佳选择；
- global assignment：解一个全局匹配/最优传输问题，更优雅但计算代价高；
- hashing routing：用哈希函数固定分配 token，意外地也能带来增益，但不如学习式路由灵活；
- RL routing：用强化学习处理离散选择，原则上合理，实践中成本和方差太高。

## 逐步教程

### 第一步：把 FFN 替换成专家池

普通 FFN 可写成：

```text
h_out = h + FFN(h)
```

MoE 层则写成：

```text
h_out = h + sum_{i in TopK(router(h))} g_i(h) * Expert_i(h)
```

其中：

- `h` 是当前 token 的 hidden state；
- `Expert_i` 是第 i 个 FFN；
- `router(h)` 给出 token 对所有 expert 的分数；
- `TopK` 只保留 K 个 expert；
- `g_i(h)` 是门控权重，可归一化，也可不完全归一化，取决于实现。

如果 K=1，计算量接近一个 dense FFN；如果 K=2，大致相当于激活两个 FFN，FLOPs 约翻倍。但模型总参数可以远大于 dense model。

### 第二步：理解专家数量与大小

早期 MoE 常把 dense FFN 复制成多个同样大小的 expert。后来 DeepSeek 等系统发现 fine-grained experts（细粒度专家）非常有效：把每个 expert 做得更小，但数量更多。

例如，原本一个 FFN 的中间维度是 hidden size 的 4 倍。细粒度 MoE 可以把每个 expert 的中间维度切成原来的 1/2、1/4 甚至更小，然后激活更多个 expert。这样可以增加路由组合的灵活性，同时控制 FLOPs。

另一个设计是 shared expert（共享专家）：无论 router 选择什么，每个 token 都经过一个或几个共享 FFN。动机是让模型保留一部分通用处理能力，而不是所有计算都依赖稀疏路由。DeepSeek 系列使用过 shared expert，但其他模型的消融实验显示其收益并不总是稳定，因此这是一个工程选择。

### 第三步：为什么 load balancing 必不可少

如果没有约束，router 很容易陷入坏的局部最优：所有 token 都被送到少数几个 expert，其他 expert 几乎不训练，成为 dead experts（死亡专家）。这不仅浪费显存，也使 MoE 退化成一个小得多的模型。

因此训练 MoE 时通常加入 auxiliary load balancing loss（辅助负载均衡损失）。Switch Transformer 中常见形式是：

```text
L_balance = alpha * N * sum_i f_i * p_i
```

其中：

- `N` 是 expert 数量；
- `f_i` 是实际被路由到 expert i 的 token 比例；
- `p_i` 是 router 分配给 expert i 的平均概率；
- `alpha` 是损失权重。

直觉是：如果某个 expert 已经拿到太多 token，它对应的路由概率会被压低，从而鼓励 token 分散到其他 expert。

负载均衡不仅是系统优化，也是建模优化。即使不考虑 GPU 利用率，也需要它来避免专家坍缩。

### 第四步：训练中的稳定性问题

MoE 难训练的原因主要有三点：

1. top-k 是离散选择，未被选中的 expert 没有梯度；
2. router 可能过早偏向少数 expert；
3. softmax router 在低精度训练中可能引入数值不稳定。

常见稳定化技巧包括：

- router 计算使用 float32；
- 对 router logits 加 z-loss，约束 softmax normalizer；
- 加 load balancing loss；
- 有时加入噪声或 jitter 促进探索；
- 微调时使用更多数据，避免 MoE 因总参数巨大而过拟合。

DeepSeek V3 提出 auxiliary-loss-free balancing：为每个 expert 维护一个偏置 `b_i`。如果 expert i 最近拿到的 token 太少，就增加 `b_i`；如果太多，就减少 `b_i`。这个偏置只用于路由选择，不作为最终 gating weight。实践中 DeepSeek V3 仍保留了 sequence-wise auxiliary loss，用于控制单条序列内的负载不均。

### 第五步：训练与推理成本

MoE 的 FLOPs 很有吸引力，但真实系统成本不只看 FLOPs。

训练成本包括：

- 激活 expert 的矩阵乘法成本；
- 存储所有 expert 参数的显存成本；
- router 与 load balancing 的额外计算；
- token dispatch 和 combine 的通信成本；
- 负载不均导致的设备空等。

推理成本也类似。虽然每个 token 只激活少数 expert，但所有 expert 的权重必须在某处存放。若 expert 分布在多 GPU 上，就需要 all-to-all communication：先把 token 发送到对应 expert 所在设备，计算后再把结果送回。

如果某个 expert 收到太多 token，系统可能触发 token dropping：超过容量的 token 不经过该 expert，直接靠 residual connection 传下去。这会造成训练或推理结果依赖 batch 中其他请求，从而引入看似奇怪的非确定性。

### 第六步：Expert Parallelism

expert parallelism 是 MoE 的重要系统优势。因为 expert 是天然分块的，可以把不同 expert 放在不同设备上。流程大致是：

1. 每个设备上有一部分 token hidden states；
2. router 决定每个 token 要去哪些 expert；
3. all-to-all 把 token 发到对应设备；
4. expert 执行 FFN；
5. all-to-all 把输出发回原位置；
6. 按 gating weight 合并结果。

这给大模型并行增加了一条轴：除了 data parallelism、tensor/model parallelism、pipeline parallelism，还可以用 expert parallelism。但通信拓扑、batch size、expert 数量、capacity factor 都会影响效率。

一个重要判断标准是：expert 的计算必须“足够厚”，才能掩盖 all-to-all 通信成本。如果每个 expert 太小、每次只收到很少 token，GPU 上会出现许多小矩阵乘法和通信碎片，实际 wall-clock time 可能并不理想。因此现代实现会使用 fused kernels、block-sparse matrix multiplication、MegaBlocks 之类的库，把多个专家计算组织成更适合硬件执行的批量操作。

### 第七步：DeepSeek 系列的演化

DeepSeek MoE 是本讲反复引用的现代案例。早期 DeepSeek MoE 已经采用了两个关键设计：fine-grained experts 和 shared experts，并使用 top-k routing 加辅助负载均衡。DeepSeek V2 在总体结构上变化不大，但规模扩大到数百亿激活参数，并加入 top-M device selection：先限制 token 可以访问的设备集合，再在这些设备内部选 expert，以降低跨设备通信。

DeepSeek V3 继续沿用 MoE 主体，但调整了 router：使用更温和的 sigmoid 风格分数，并引入按 expert 负载在线更新的 bias，实现所谓 auxiliary-loss-free balancing。同时它仍保留 sequence-wise auxiliary loss，以防推理时单条异常序列压垮少数 expert。这个演化说明：MoE 的基本架构并不复杂，真正的进步往往来自训练稳定性、通信控制和负载均衡细节。

## 公式与伪代码

### Top-K MoE 前向传播

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

### DeepSeek V3 风格偏置更新

```text
if load_i < target_load:
    b_i = b_i + gamma
else:
    b_i = b_i - gamma

routing_score_i = router_score_i + b_i
```

注意：`b_i` 用于决定 top-k，但最终合并 expert 输出时通常仍使用原始 gating score。

## 常见误区

1. “Expert 就是人类可解释的领域专家。”
   不一定。expert 是稀疏激活的子网络，可能有专门化，但通常不是清晰的代码/数学/中文专家。

2. “MoE 参数更多，所以一定更贵。”
   总参数更多，但每个 token 只激活少量参数。FLOPs 取决于 activated parameters，而显存取决于 total parameters。训练或部署 MoE 时，必须同时报告总参数、激活参数、每 token 激活 expert 数量，否则很容易误读模型规模。

3. “只要有很多 expert 就会更好。”
   更多 expert 会增加显存和通信复杂度。如果路由不均，很多 expert 会死亡；如果 expert 太细，通信碎片化也会变严重。真正有意义的是“更多可用且被充分训练的 expert”，而不是配置文件里写着更多 expert。

4. “softmax 后 top-k 可以去掉。”
   不能随便去掉。若所有 expert 都参与计算，MoE 就失去稀疏计算优势，训练和推理成本会爆炸。softmax 的主要作用是产生可比较、可加权的分数；top-k 的作用是强制稀疏激活，两者解决的是不同问题。

5. “Load balancing 只是为了 GPU 利用率。”
   不只是。它还防止 expert collapse，让所有参数都得到训练。没有负载均衡时，模型可能看起来仍在下降 loss，但实际上只训练了少数 expert，浪费了大部分容量。

6. “MoE 推理一定比 dense 快。”
   不一定。若模型太大导致权重分布在多机多卡上，通信和调度可能抵消稀疏计算的收益。MoE 适合在足够大规模、足够成熟的推理系统中发挥优势；在单卡小模型场景下，dense model 可能更简单、更稳定。

## 实践练习

1. 手写一个 toy MoE 层：输入二维向量，4 个 expert，每次 top-2 激活，观察不同 token 的路由。
2. 关闭 load balancing loss，记录每个 expert 的 token 数量，看看是否出现 expert collapse。
3. 比较 K=1 与 K=2：训练 loss、计算量、expert 利用率有何变化？
4. 实现 capacity factor：每个 expert 最多接收固定数量 token，超过则 drop，观察输出是否受 batch 组成影响。
5. 阅读 DeepSeek V3 技术报告，找出其 MoE 之外的两个关键设计：MLA（Multi-head Latent Attention）和 MTP（Multi-token Prediction）。

## 总结

MoE 是现代高性能语言模型的重要架构。它的核心思想很简单：用 router 为每个 token 选择少数 expert，从而在不同比例上解耦 total parameters 与 activated parameters。这样可以在相同训练 FLOPs 下获得更低 loss，或者在相似推理计算下使用更大容量模型。

真正困难的是工程与优化：离散 top-k 路由难以训练；expert 可能负载不均；跨设备通信会成为瓶颈；推理时还可能因 token dropping 引入非确定性。现代系统通过 top-k routing、fine-grained experts、shared experts、load balancing loss、expert parallelism、auxiliary-loss-free balancing 等技巧，使 MoE 成为大规模模型训练和部署的主流选择。

## 下一讲衔接

本讲主要讨论 MoE 架构与训练。下一步需要更深入理解大模型系统层面的并行策略：data parallelism、tensor parallelism、pipeline parallelism、expert parallelism 如何组合，以及通信带宽、显存、batch size 如何共同决定真实训练吞吐。理解这些系统细节后，才能真正判断 MoE 在某个硬件集群上是否“划算”。如果你继续学习系统部分，建议始终把“数学上的 FLOPs”和“机器上的时间”分开思考：MoE 的论文曲线常常展示前者，而工程成败往往取决于后者。


---


# Stanford CS336 Lecture 5：GPU 中文教程

大语言模型之所以能训练到今天的规模，核心原因之一是硬件吞吐量的持续增长，尤其是 GPU 的普及。本讲的重点不是把 CUDA API 背下来，而是建立一个性能直觉：GPU 很擅长并行矩阵乘法，但常常不是“算不动”，而是“数据搬不动”。理解 GPU 架构、执行模型和内存层次，才能解释为什么同一个矩阵乘法在某些维度上飞快、在另一些维度上突然变慢，也才能理解 FlashAttention 这类算法为什么有效。

## 1. CPU 与 GPU 的设计目标

CPU 优化的是低延迟。它通常有复杂的控制逻辑、分支预测、缓存体系和较高的单线程性能，目标是让一个任务尽快完成。GPU 优化的是高吞吐。它牺牲单个线程的灵活性和低延迟，把芯片面积更多用于大量算术单元，让成千上万个相似任务同时推进。

这正好适合深度学习：训练 Transformer 时，大量工作都是矩阵乘法、逐元素运算、归约和张量变换。这些操作结构规则、数据量巨大，可以拆成很多相似的小任务并行执行。随着 Dennard scaling 和单线程性能增长放缓，深度学习的扩展越来越依赖并行硬件，而 GPU 正是这种并行扩展的代表。

## 2. GPU 的基本架构：SM、SP 与 Tensor Core

可以把一块 GPU 看成由许多 Streaming Multiprocessor（SM）组成。SM 类似 GPU 上的基本执行单元：每个 SM 有自己的调度和控制逻辑、寄存器、共享内存以及大量执行单元。SM 内部有许多更细粒度的处理单元，可以对不同数据执行同一条指令。

现代 NVIDIA GPU 还包含 Tensor Core，这是一类专门为矩阵乘法设计的硬件单元。从 V100 开始，矩阵乘法吞吐量与普通浮点运算吞吐量之间出现巨大差距：如果你的神经网络大部分时间都能落在矩阵乘法上，就能吃到 Tensor Core 的红利；如果设计了大量非矩阵乘法的复杂操作，即使理论 FLOPs 不高，也可能跑得很慢。

这也是 LLM 架构偏爱线性层、注意力中的 QK^T 和 PV、MLP 中大矩阵乘法的原因：这些操作与 GPU 的硬件能力高度匹配。

## 3. SIMT、thread、warp 与 block

GPU 的执行模型通常称为 SIMT：Single Instruction, Multiple Threads。也就是说，同一组线程在同一时刻执行同一条指令，但处理不同的数据。

CUDA 编程中常见三个层次：

- thread：最小的逻辑执行单位。
- warp：一组通常为 32 个连续线程，它们一起执行同一条指令。
- block：一组线程，通常被分配到某个 SM 上执行。

这个模型带来一个重要限制：同一个 warp 内最好不要出现严重分支分歧。例如 32 个线程中一半走 if 分支，另一半走 else 分支，GPU 不能真正同时执行两条不同路径，而是先让一部分线程执行、另一部分暂停，再反过来执行。这样会降低有效利用率。因此，高性能 GPU kernel 通常追求规则的数据访问和规则的控制流。

## 4. 内存层次：性能优化的主战场

GPU 的计算单元非常快，但内存速度差异巨大。可以按从快到慢理解：

- register：每个线程私有，最快，适合保存临时标量。
- shared memory / L1：位于 SM 内部，延迟很低，线程块内可以共享。
- L2 cache：在芯片上，但不在单个 SM 内，速度较慢。
- global memory / HBM：位于芯片外的高带宽显存，容量大但延迟高。

访问 shared memory 可能只需几十个 cycle，而访问全局显存可能需要数百个 cycle。如果 kernel 不断从 HBM 读写中间结果，计算单元就会等待数据，吞吐量很难上去。

一个优秀 GPU 算法的基本原则是：尽量少访问 global memory；一旦把数据搬进 SM，就在 shared memory 或寄存器中做尽可能多的计算；最后只把必要结果写回 HBM。

## 5. Roofline 模型：算力瓶颈还是内存瓶颈

Roofline 模型用来判断程序性能受什么限制。横轴常理解为 arithmetic intensity，即每读取一个字节数据能做多少 FLOPs；纵轴是实际吞吐。

当 arithmetic intensity 很低时，程序在左侧的 memory-bound 区域：算术单元还没吃饱，性能主要由内存带宽决定。很多逐元素操作就是这样，例如 ReLU、加法、LayerNorm 的某些部分，它们读写很多数据，但每个元素只做少量计算。

当 arithmetic intensity 足够高时，程序进入 compute-bound 区域：矩阵乘法足够大，数据复用充分，Tensor Core 被充分利用，吞吐接近硬件峰值。

LLM 训练中，大矩阵乘法通常更容易 compute-bound，而小 batch、小矩阵、逐元素操作、归约、频繁写回中间张量则容易 memory-bound。优化的核心就是把更多工作推向 roofline 的右上方。

## 6. 矩阵乘法为什么需要 tiling

朴素矩阵乘法 C = A × B 中，每个 C[i,j] 需要读取 A 的一行和 B 的一列。如果每个线程都直接从 global memory 读取所需元素，会产生大量重复读取：同一个 A 元素会被多个输出复用，同一个 B 元素也会被多个输出复用。若每次都从 HBM 取，显然浪费。

Tiling 的思路是把 A、B、C 切成小块。一个 block 负责计算 C 的一个 tile。它先把 A 和 B 的对应 tile 从 global memory 搬到 shared memory，然后在 shared memory 中反复使用这些数据，累加 partial sum。处理完当前 tile 后，再加载下一个 tile。

这样做有两个收益：

1. 全局内存读取次数减少。tile 大小为 T 时，理想情况下可以把部分 global memory 访问减少约 T 倍。
2. 访问模式更规则。加载 tile 时可以安排连续线程读取连续地址，便于 memory coalescing。

不过 tile 大小不是越大越好。它受 shared memory 容量、寄存器数量、warp 调度、Tensor Core 形状和矩阵维度整除性的共同限制。矩阵维度如果刚好是 tile 大小、warp 大小或 burst section 的倍数，通常会更快；如果多出 1 个元素，可能需要额外 tile，导致很多 SM 处理“稀疏边角块”，吞吐突然下降。

## 7. Memory coalescing、padding 与奇怪的性能波动

DRAM 通常不是一次只返回一个标量，而是按连续块读取。若同一 warp 的线程访问相邻地址，硬件可以把这些访问合并成更少的内存事务，这叫 memory coalescing。若线程访问分散地址，就会触发多次读取，带宽利用率下降。

这解释了很多看似玄学的现象：矩阵按行访问还是按列访问，性能可能差很多；词表大小、hidden size、batch size 是否是 8、16、32、64、128 的倍数，也会影响吞吐。Karpathy 曾提到把 nanoGPT 的 vocab size padding 到 64 的倍数能带来明显加速，本质上就是让矩阵形状更适合 GPU 的 tile、warp 和内存对齐。

另一个现象是 wave quantization。假设 A100 有 108 个 SM，如果一次矩阵乘法被切成 98 个 tile，那么一波即可让大部分 SM 工作；如果维度略增导致 tile 数变成 120，前 108 个 tile 先跑，剩下 12 个 tile 再单独跑一小波，后半段 SM 利用率很低。于是矩阵只增大一点，性能却可能掉一大截。

## 8. 降低精度、融合与重计算

为了减少内存压力，常用三类技巧。

第一是低精度。FP16、BF16、FP8 或 int8 会减少每个元素的字节数，使同样的带宽能搬运更多数据，也能使用更快的 Tensor Core。训练中通常采用 mixed precision：输入和权重用 16 位，乘法累加用 FP32 accumulator，以兼顾速度和数值稳定性。

第二是 operator fusion。若代码先计算 sin(x)，写回 HBM，再读出计算平方，再写回，内存往返会非常多。融合 kernel 会把多个逐元素操作放在一次 kernel 中完成，中间值保存在寄存器或 shared memory，只在最后写回结果。torch.compile、Triton 和手写 CUDA kernel 都常用于这种优化。

第三是 recomputation。反向传播需要前向激活。朴素做法是把所有激活存到 HBM，反向时再读回来。但如果某些激活计算便宜、读取昂贵，可以在反向时重新计算它们，用额外 FLOPs 换更少的内存读写。这不仅能省显存，也能在 memory-bound 场景下提速。

## 9. FlashAttention：把这些思想组合起来

标准 attention 包含 QK^T、softmax、再乘 V。问题是注意力矩阵大小为 n × n，如果把完整 score 和 softmax 结果都 materialize 到 HBM，长上下文时内存读写会非常昂贵。

FlashAttention 的关键不是减少 attention 的数学计算量，而是减少 HBM 访问。它使用 tiling：把 Q、K、V 分块搬到 SRAM/shared memory，在块内计算 QK^T 和后续累加。难点是 softmax 是行级全局操作，必须知道整行的最大值和归一化分母。FlashAttention 使用 online softmax：按块维护当前行的 running max 和归一化和，每来一个 tile 就更新这些统计量，因此不需要把完整 n × n 矩阵写回显存。

反向传播中，FlashAttention 还会对 softmax 相关量做重计算，避免保存 n × n 的中间激活。于是它结合了本讲的多个核心技巧：tiling、shared memory 复用、operator fusion、online softmax 和 recomputation。结果是精确 attention，但 HBM 访问显著下降，长序列 Transformer 训练和推理都更快。

## 10. 总结：LLM 训练为何依赖 GPU

LLM 训练依赖 GPU，不只是因为 GPU FLOPs 高，更因为 Transformer 的主要计算形式天然适合 GPU：大规模矩阵乘法、规则张量操作、可批处理的并行数据流。Tensor Core 让矩阵乘法成为“被硬件祝福”的操作，混合精度进一步放大吞吐。

但现代 GPU 的真正瓶颈越来越多地来自内存移动，而不是纯计算。高性能实现要关注：是否让 warp 访问连续内存；是否避免不必要的 HBM 读写；是否用 tiling 提高数据复用；矩阵维度是否对齐；是否能 fusion；是否能用重计算换内存；是否让 tile 数和 SM 数匹配良好。

因此，理解 GPU 的核心不是记住某个型号的参数，而是形成一套判断性能瓶颈的思维：计算是否足够密集？数据是否被重复从 HBM 搬运？warp 是否分歧？访问是否合并？tile 是否对齐？这些细节共同决定了 LLM 训练能否真正把昂贵的 GPU 用满。

## 11. 实践检查清单：看到慢代码时先问什么

当你在训练或推理 LLM 时发现 GPU 利用率不高，可以按下面顺序排查。第一，看是不是 CPU 在拖后腿：数据加载、tokenization、日志打印、频繁 `.item()` 都可能让 GPU 等待。第二，看是不是 kernel 太碎：如果一个 Transformer block 中出现大量很短的小 kernel，说明许多逐元素操作没有融合，kernel launch 和 HBM 往返会吞掉时间。第三，看矩阵形状是否适合 Tensor Core：hidden size、vocab size、batch×sequence 是否对齐到硬件喜欢的倍数。第四，看显存占用是否迫使 batch 太小：batch 太小会降低矩阵乘法算术强度，也会让并行度不够。第五，用 profiler 确认瓶颈，而不是只看 `nvidia-smi` 的利用率；后者只能给粗略信号，不能告诉你到底是哪个 kernel 慢。

## 12. 一个贯穿例子：为什么“同样的模型”速度可以差很多

假设两个实现都训练同一个 Transformer，参数量、batch size 和 dtype 完全相同。实现 A 直接用许多 PyTorch 小操作拼接 attention、MLP、residual 和 normalization；实现 B 使用 fused layer norm、FlashAttention、fused optimizer，并把词表和 hidden size padding 到更友好的维度。两者数学上几乎等价，但实现 B 会少写回大量中间张量，减少 CPU/GPU 同步，让矩阵乘法更稳定地落到高效 tile 上。结果可能不是提升 5%，而是提升数十个百分点甚至更多。

这就是本讲想建立的工程直觉：模型架构论文里的一行公式，落到 GPU 上会变成很多 kernel、很多内存事务和很多调度决策。做 LLM Infra 不能只懂模型，也要能把公式翻译成硬件成本：哪些张量会被 materialize，哪些中间值可以重算，哪些操作应该融合，哪些维度会触发额外一波 tile。掌握这套直觉后，后面的 Triton、自定义 kernel、分布式训练和推理服务才会真正连起来。

最后给一个最实用的判断标准：如果优化能减少 HBM 读写、提高 Tensor Core 利用率、降低同步次数或提升 batch/sequence 的有效并行度，它通常值得尝试；如果只是把代码写得更底层但没有改变数据移动路径，收益往往有限。


---


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


---


# Stanford CS336 Lecture 7 中文教程：Parallelism 1

## 学习目标

读完本讲，你应该能够：

1. 解释为什么训练大语言模型（LLM）必须从单 GPU 扩展到多 GPU、多节点甚至整个数据中心。
2. 区分 data parallelism、model parallelism、tensor parallelism、pipeline parallelism 与 activation/sequence parallelism 的核心思想。
3. 理解 all-reduce、reduce-scatter、all-gather 等 collective communication 在分布式训练中的作用。
4. 判断不同并行策略在计算、显存、通信带宽和 batch size 之间的取舍。
5. 根据硬件拓扑给出一个合理的 LLM 训练并行配置思路。

## 本讲地图

本讲讨论的不是“如何让单个 kernel 更快”，而是“模型太大、训练太慢时，如何把训练任务切到很多机器上”。主线如下：

- 先看动机：单 GPU 的 FLOPs 和显存都不够，必须使用 multi-machine parallelism。
- 再看通信基础：GPU 之间并不是等价连接，节点内 NVLink/NVSwitch 很快，跨节点 InfiniBand 或以太网更慢。
- 然后介绍 data parallelism：复制模型，切分数据，并同步梯度；再用 ZeRO/FSDP 降低显存占用。
- 接着介绍 model parallelism：不再复制完整模型，而是把模型本身切开，包括 pipeline parallelism 和 tensor parallelism。
- 最后讨论 activation memory、sequence parallelism、通信/计算权衡，以及实际大模型训练中的组合规则。

## 核心概念

### 1. 为什么 LLM 训练需要并行？

大模型训练有两个硬约束：compute 和 memory。

Compute 指训练所需的总浮点运算量。模型越大、token 越多，训练所需 FLOPs 越高。单个 GPU 即使每代都变快，也无法立刻满足前沿模型的训练需求。

Memory 指 GPU 显存。参数本身只是显存的一部分。训练时还要存：

- parameters：模型参数，通常 BF16/FP16，约 2 bytes/parameter；
- gradients：梯度，约 2 bytes/parameter；
- optimizer states：Adam 的一阶、二阶矩，以及 master weights，常常占更多；
- activations：前向传播中为反向传播保存的中间结果。

一个常见估算是，使用 Adam 训练时，单个参数可能需要约 16 bytes 的训练状态。因此，7B、70B、甚至更大模型无法简单放进一张 GPU。

### 2. 通信原语：collective communication

分布式训练大量依赖 collective communication：

- all-reduce：每个 rank 都有一份张量，先做求和/平均等 reduce，再把结果发回所有 rank。数据并行同步梯度常用它。
- reduce-scatter：先 reduce，再把结果按 shard 分给不同 rank。
- all-gather：每个 rank 有一个 shard，把所有 shard 收集起来，使每个 rank 都得到完整张量。
- broadcast：一个 rank 的数据复制给所有 rank。

重要等价关系：

```text
all-reduce ≈ reduce-scatter + all-gather
```

在 bandwidth-bound 场景下，二者通信量近似等价。这个事实解释了 ZeRO 为什么能在不明显增加通信量的情况下节省显存。

### 3. 硬件拓扑决定并行策略

并行算法不能脱离硬件。典型 NVIDIA 训练节点中，一台机器可能有 8 张 GPU，节点内通过 NVLink/NVSwitch 高速互连；跨机器则通过 InfiniBand 等网络连接，带宽和延迟明显更差。

因此经验规则是：

- bandwidth-hungry 的 tensor parallelism 通常放在节点内；
- data parallelism 可以跨更慢的网络，因为每个 step 只同步一批梯度或参数 shard；
- pipeline parallelism 通信的是 activation，且多为 point-to-point，有时适合跨节点或跨较慢链路。

## 逐步教程

开始设计并行方案时，可以先问三个问题。第一，单张卡是否能放下模型参数、梯度、优化器状态和峰值 activation；如果不能，必须先切模型或切训练状态。第二，通信链路在哪里最快；节点内高速互连适合频繁同步，跨节点慢链路应尽量少同步或只做点到点传输。第三，当前训练允许的有效 batch size 有多大；如果 batch 已经接近临界值，继续用数据并行堆 GPU 可能只会增加通信而不提升收敛速度。

一个实用思路是把并行策略看成不同维度的切分。Data parallelism 切的是样本维度，优先提高吞吐；tensor parallelism 切的是每层矩阵宽度，优先解决单层太宽、参数太大的问题；pipeline parallelism 切的是网络深度，优先解决层数太多、整网放不下的问题；sequence parallelism 切的是序列维度，专门处理 activation 中难以被 tensor parallelism 降下来的部分。实际系统通常不是四选一，而是把它们叠加成所谓 3D/4D parallelism。

## Step 1：Data Parallelism：复制模型，切分数据

Data parallelism（数据并行）是最自然的并行方式：每张 GPU 保存完整模型参数，但处理不同的数据样本。

设全局 batch size 为 B，GPU 数为 M，则每张 GPU 处理 B/M 个样本。每张 GPU 独立做 forward/backward，得到本地梯度，然后用 all-reduce 对梯度求平均，最后每张 GPU 执行相同的 optimizer step。

SGD 更新可写作：

```text
θ_{t+1} = θ_t - η · (1/B) · Σ_{i=1}^B ∇ℓ(x_i; θ_t)
```

数据并行只是把这个求和拆给不同 GPU 做。

优点：

- 计算扩展性好：batch 足够大时，更多 GPU 可以处理更多样本。
- 实现简单，对模型结构不敏感。

缺点：

- 显存不省：每张 GPU 都存完整参数、梯度和 optimizer state。
- 通信量与参数量相关：每个 step 需要同步梯度。
- 受 batch size 限制：GPU 数不能无限超过有效 batch size。

### Batch size 是一种资源

当 batch size 很小时，数据并行无法继续扩展，因为每张 GPU 至少要有可用样本。即使能增大 batch，优化效果也存在 critical batch size：超过某个点后，继续增大 batch 对训练速度的收益递减，因为瓶颈从“梯度噪声”变成了“梯度更新步数”。

所以，batch size 不是随便可用的无限资源。它要在 data parallelism、pipeline parallelism 和 gradient accumulation 之间分配。

## Step 2：ZeRO/FSDP：让数据并行也节省显存

朴素数据并行浪费显存，因为每张 GPU 都复制完整训练状态。ZeRO（Zero Redundancy Optimizer）逐步切分这些状态。

### ZeRO Stage 1：切 optimizer state

每张 GPU 仍保存完整 parameters 和 gradients，但 Adam 的 optimizer states 被切成 shards。每张 GPU 只负责更新自己拥有的参数 shard。

流程：

1. 每张 GPU 计算完整梯度。
2. 用 reduce-scatter 把梯度按参数 shard 汇总到对应 GPU。
3. 每张 GPU 用自己的 optimizer state 更新对应参数 shard。
4. 用 all-gather 把更新后的参数 shard 收集回每张 GPU。

通信上，reduce-scatter + all-gather 近似等价于原来的 all-reduce，因此 Stage 1 几乎是“免费”的显存优化。

### ZeRO Stage 2：再切 gradients

Stage 2 不再保留完整梯度。反向传播时，每算出一层梯度，就立刻 reduce 到负责该参数 shard 的 GPU，并释放本地临时梯度。

这样避免在显存中实例化完整 gradient vector。通信总量仍接近原始数据并行，但调度更复杂。

### ZeRO Stage 3 / FSDP：连 parameters 也切

FSDP（Fully Sharded Data Parallel）基本对应 ZeRO Stage 3：parameters、gradients、optimizer states 全部 sharded。

核心思想是“按需 all-gather 参数”：

1. 某层 forward 前，all-gather 该层参数。
2. 完成该层计算后释放参数。
3. backward 时再次按需 all-gather 参数。
4. 算出梯度后 reduce-scatter 到负责该 shard 的 GPU。

通信量从约 2×parameters 增加到约 3×parameters，但通过 overlap communication and computation（通信与计算重叠）、prefetch（预取）等技术，实际开销可接受。

FSDP 的优势是通用：它不需要深入理解 Transformer 结构，常作为大模型训练的默认显存优化手段。

## Step 3：Model Parallelism：把模型本身切开

当模型或 activation 仍然放不下时，需要 model parallelism（模型并行）。与 FSDP 不同，模型并行的目标不是“临时收集完整参数”，而是让模型的不同部分固定生活在不同 GPU 上，通信的主要对象变成 activations。

本讲重点介绍两种：pipeline parallelism 和 tensor parallelism。

## Step 4：Pipeline Parallelism：按层切模型

Pipeline parallelism（流水线并行）沿深度方向切模型：例如 GPU0 放前几层，GPU1 放中间层，GPU2 放后几层。forward 时 activation 从前往后传，backward 时 activation gradients 从后往前传。

朴素做法会产生严重 bubble：某个时刻只有一张 GPU 在工作，其他 GPU 空闲。为减少 bubble，通常把 batch 切成多个 micro-batches，让不同 micro-batch 像工厂流水线一样同时处在不同 stage。

bubble 开销的常见近似：

```text
bubble_ratio ≈ (pipeline_stages - 1) / micro_batches
```

因此 micro-batch 越多，pipeline 越满。但这又消耗 batch size 资源。

优点：

- 参数和部分 activation 按层分布，显存扩展好。
- 通信多为相邻 stage 的 point-to-point activation 传输。
- 适合跨较慢网络链路使用。

缺点：

- 调度复杂，尤其是 1F1B、interleaved pipeline、zero-bubble pipeline 等高级策略。
- bubble 会降低 GPU 利用率。
- 工程实现非常难，常需要深度介入 autograd 和 runtime 调度。

Zero-bubble pipeline 的一个技巧是把 backward 拆成两类工作：

- B：反传 activation gradient，存在严格依赖；
- W：计算 weight gradient，依赖较少，可以挪到 bubble 里执行。

这样可用原本空闲的时间计算参数梯度，提高利用率。

## Step 5：Tensor Parallelism：按矩阵宽度切模型

Tensor parallelism（张量并行）沿宽度方向切矩阵乘法。Transformer 的主要计算来自大矩阵乘法，因此可以把权重矩阵切成多个子矩阵，由多张 GPU 分别计算 partial results，再通过 collective communication 合并。

例如 MLP 中：

```text
Y = GeLU(XA)
Z = YB
```

可以把 A、B 切成 A1/A2、B1/B2。每张 GPU 处理一部分矩阵乘法，必要时 all-reduce 合并结果。

优点：

- 不消耗 batch size。
- 没有 pipeline bubble。
- 对 Transformer 这类以矩阵乘法为主的模型很自然。

缺点：

- 每层都可能有同步屏障。
- 通信的是 activation，频率高、带宽需求大。
- 通常只适合节点内高速互连，例如 8 张 GPU 的 NVLink/NVSwitch 环境。

经验规则：tensor parallel size 常设为节点内 GPU 数，例如 8。超过节点后跨慢链路做 tensor parallel，吞吐可能明显下降。

## Step 6：Activation 与 Sequence Parallelism

前面主要处理了参数、梯度和 optimizer state，但 activation memory 也会成为大问题。它的峰值通常出现在反向传播早期：此时许多前向 activation 还没释放，梯度又开始累积。序列越长、batch 越大，这部分越明显；长上下文训练尤其容易被 activation 卡住。

Transformer 每层 activation 近似包含两类项：

```text
activation_memory_per_layer ≈ S · B · H · 34 + 5 · A · S² · B
```

其中 S 是 sequence length，B 是 batch size，H 是 hidden size，A 是 attention heads。右侧 S² 项来自 attention softmax 等二次复杂度部分，可通过 FlashAttention 和 recomputation 大幅降低。

Tensor parallelism 能切分许多矩阵乘相关 activation，但 layer norm、dropout、残差流输入等 pointwise operations 仍可能保留完整 activation。Sequence parallelism 的做法是沿 sequence dimension 切这些 pointwise activation：不同 GPU 负责不同 token position。

这会引入额外 all-gather / reduce-scatter，但可进一步降低 activation memory。结合 activation recomputation，常能用更多计算换取更低显存，从而允许更大 batch 或更大模型。

## 公式与伪代码速查

### 数据并行训练伪代码

```python
for batch in data:
    local_batch = shard(batch, rank)
    loss = model(local_batch)
    loss.backward()
    all_reduce(model.gradients)   # 同步所有 rank 的梯度
    optimizer.step()
    optimizer.zero_grad()
```

### FSDP 思想伪代码

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

### 并行策略组合规则

```text
先保证模型能放进显存：
    1. 节点内优先用 tensor parallelism
    2. 仍放不下时，用 FSDP/ZeRO-3 或 pipeline parallelism

模型能放下后：
    3. 剩余 GPU 用 data parallelism 扩展吞吐
    4. 如果通信太频繁，用 gradient accumulation 增大有效 batch
```

## 常见误区

1. “GPU 越多越快。”
   错。通信、同步、pipeline bubble 和 batch size 限制都会让扩展效率下降。

2. “Data parallelism 能解决显存问题。”
   朴素 DDP 不能。只有 ZeRO/FSDP 这类 sharding 技术才能显著减少参数相关显存。

3. “FSDP 就是 model parallelism。”
   不完全是。FSDP 虽然切参数，但计算时会按需 all-gather 参数；model parallelism 更强调参数固定分布，主要传 activation。

4. “Tensor parallelism 可以随便跨节点扩展。”
   通常不行。它每层都通信，对带宽和延迟极敏感，最好放在节点内高速互连上。

5. “Pipeline parallelism 概念简单，所以实现简单。”
   恰好相反。高效 pipeline 调度、micro-batch、1F1B、zero-bubble、autograd 介入都很复杂。

6. “Activation memory 不重要。”
   对长序列、大 batch、大模型训练，activation 可能成为主要显存瓶颈。

## 练习

1. 假设一个模型有 P 个参数，使用 Adam 训练，每个参数训练状态约 16 bytes。估算 7B 参数模型仅参数相关训练状态需要多少显存。

2. 用自己的话解释为什么：

```text
all-reduce ≈ reduce-scatter + all-gather
```

并说明这个等价关系如何帮助 ZeRO Stage 1。

3. 一个 8 GPU 节点内 NVLink 很快，跨节点网络较慢。你会把 tensor parallelism、pipeline parallelism、data parallelism 分别放在哪些范围？为什么？

4. 如果 pipeline stages = 8，micro-batches = 32，估算 bubble_ratio。micro-batches 减半后会发生什么？

5. 解释 FSDP 与 tensor parallelism 的主要区别：它们分别通信什么？分别依赖什么硬件特性？

## 总结

LLM 训练并行的本质是同时管理四种稀缺资源：显存、计算、通信带宽/延迟和 batch size。Data parallelism 简单且适合扩展吞吐，但受 batch size 和显存复制限制；ZeRO/FSDP 通过 sharding optimizer states、gradients、parameters 让数据并行也能节省显存；pipeline parallelism 沿层切模型，通信较温和但有 bubble 和复杂调度；tensor parallelism 沿矩阵宽度切模型，不消耗 batch size，但需要高速互连；sequence parallelism 和 activation recomputation 则进一步处理 activation memory。

实际训练中没有单一最佳方案。常见做法是：节点内用 tensor parallelism，必要时结合 sequence parallelism；模型仍放不下时加入 FSDP 或 pipeline parallelism；最后用 data parallelism 吃掉剩余 GPU，并用 gradient accumulation 调节通信频率。理解这些策略的通信对象和硬件需求，是设计高效 LLM 训练系统的关键。

从工程角度看，好的并行方案不是追求某个术语，而是让昂贵 GPU 尽量少等待。通信可以被预取和重叠，显存可以用分片和重计算换取，batch size 可以用来摊薄同步成本，但每个选择都会把压力转移到别的资源上。Lecture 7 的核心结论正是：大规模训练是系统设计问题，算法、硬件拓扑、优化器状态、activation 生命周期和调度策略必须一起考虑。


---


# Stanford CS336 Lecture 8：并行训练（二）中文教程

本讲继续讨论大模型训练的系统层问题：当单张 GPU 已经无法容纳模型、优化器状态或足够大的 batch 时，如何把计算和数据分布到多张 GPU、甚至多个节点上，并尽量避免通信成为瓶颈。核心原则仍然和单 GPU 优化一致：让昂贵的计算单元尽可能忙起来，提高算术强度，减少不必要的数据搬运。

## 1. 多 GPU 训练的硬件视角

可以把训练集群理解成一个层次化存储与通信系统：

- GPU 内部：SM 执行计算，L1/shared memory 很快但很小，HBM 容量更大但更慢。
- 单节点多 GPU：GPU 之间通过 PCIe 或 NVLink/NVSwitch 连接。NVLink 是 NVIDIA 为 GPU 间高带宽通信设计的专用链路，通常远快于 PCIe。
- 多节点：节点之间还要经过网卡、交换机等网络设备，带宽更低、延迟更高。

因此，跨 GPU 通信比访问本地 HBM 更贵，跨节点通信又比同节点通信更贵。分布式训练工程的目标不是“完全避免通信”，而是：

1. 减少通信量；
2. 把通信安排在合适的位置；
3. 尽量与计算重叠；
4. 根据硬件拓扑选择并行策略。

在 NVIDIA 生态中，底层通信通常由 NCCL 负责。NCCL 会根据 GPU 拓扑选择通信路径，并把高层 collective 操作编译成底层 CUDA kernel 和数据传输。PyTorch 的 `torch.distributed` 则在 Python 层提供更方便的接口，例如 `all_reduce`、`all_gather`、`reduce_scatter` 等。

## 2. Collective 通信原语

分布式训练里经常使用 collective operations。假设一共有 `world_size` 个进程或设备，每个设备编号为一个 `rank`。

常见原语包括：

- `broadcast`：一个 rank 上的数据复制到所有 rank。
- `scatter`：一个 rank 上的不同切片分发到不同 rank。
- `gather`：多个 rank 的数据收集到一个 rank。
- `reduce`：多个 rank 的数据先做求和、平均、最大值等归约，再放到一个 rank。
- `all_gather`：每个 rank 都得到所有 rank 的数据拼接结果。
- `reduce_scatter`：先对各 rank 的输入做归约，再把归约后的不同切片分给不同 rank。
- `all_reduce`：每个 rank 都得到所有 rank 数据归约后的完整结果。

一个重要等价关系是：

```text
all_reduce = reduce_scatter + all_gather
```

例如数据并行训练中，每张 GPU 在不同数据切片上算出梯度，然后通过 `all_reduce` 求平均，使所有 GPU 的参数更新保持一致。很多高级策略会把 `all_reduce` 拆成 `reduce_scatter` 和 `all_gather`，以便减少峰值显存或更好地与计算重叠。

使用这些原语时要特别注意同步关系。collective 操作要求参与同一个 process group 的 rank 都以一致的顺序调用；如果某个 rank 少调用一次 `all_reduce`，其他 rank 可能永久等待，表现为程序 hang 住。

## 3. Benchmark：不要只看理论带宽

H100 的 NVLink 理论带宽很高，但实际训练中能达到多少，取决于张量大小、rank 数量、通信模式、NCCL 算法、节点拓扑和是否跨节点。课程中用大张量测试 `all_reduce`，得到的实际带宽明显低于硬件标称值；`reduce_scatter` 的表现也可能与简单估算不完全一致。

工程上应养成习惯：

- 对目标集群实际 benchmark，而不是只读产品规格；
- 分别测试同节点和跨节点通信；
- 关注不同消息大小下的吞吐和延迟；
- 用 warmup、`torch.cuda.synchronize()`、barrier 等方式避免计时误差；
- 区分“算法需要传输的字节数”和“墙钟时间”。

通信性能很难仅凭公式精确预测，因为 NCCL 会使用 ring、tree、分层通信、网络内归约等实现细节。因此实际调优必须依赖 profiling 和 benchmark。

## 4. 数据并行 DDP：切 batch，同步梯度

数据并行是最直观的并行方式：每张 GPU 保存完整模型，但处理不同的 batch 切片。每个 rank 独立执行 forward 和 backward，得到本地梯度，然后对所有参数梯度做 `all_reduce` 平均。

训练步骤可以概括为：

1. 把全局 batch 按 rank 切成 local batch；
2. 每个 rank 用同一份模型参数处理自己的 local batch；
3. backward 得到本地梯度；
4. 对每个参数的梯度执行 `all_reduce(mean)`；
5. 每个 rank 用相同梯度执行 optimizer step。

这样每个 rank 的 loss 可能不同，因为数据不同；但同步梯度后，参数仍保持一致。

DDP 的优点是实现简单、计算扩展性好；缺点是每张 GPU 都要保存完整参数、梯度和优化器状态。当模型变大时，显存首先成为瓶颈。另一个工程要点是，`all_reduce` 本身也是同步点；如果某些 rank 速度较慢，其他 rank 会等待，这就是 straggler 问题。

## 5. ZeRO 与 FSDP：切参数、梯度和优化器状态

为了突破 DDP 的显存限制，可以对模型状态做分片。ZeRO（Zero Redundancy Optimizer）把训练状态拆成几个层级：

- ZeRO-1：分片 optimizer states，例如 Adam 的一阶、二阶动量；
- ZeRO-2：进一步分片 gradients；
- ZeRO-3：连 parameters 也分片。

FSDP（Fully Sharded Data Parallel）可以看作 PyTorch 中类似 ZeRO-3 的实现：每个 rank 只常驻一部分参数，需要计算某层时，通过 `all_gather` 临时收集该层完整参数；反向传播结束后，用 `reduce_scatter` 把梯度归约并重新切片。

FSDP 的基本权衡是：

- 好处：显著降低每张 GPU 的常驻显存，使更大模型可以训练；
- 代价：forward/backward 期间要频繁 `all_gather` 参数和 `reduce_scatter` 梯度；
- 工程重点：合理设置 wrapping 粒度、prefetch、bucket size，避免通信碎片化。

如果切得太细，每层通信开销和调度开销会变大；如果切得太粗，峰值显存又会上升。实际训练中常按 Transformer block 作为 FSDP 单元，并结合混合精度、activation checkpointing 和 CPU/offload 策略。

## 6. Activation checkpointing：用重算换显存

反向传播需要 forward 期间保存 activations。长序列、大 batch、深层 Transformer 都会让 activation 显存非常大。Activation checkpointing 的思路是：只保存部分中间结果，backward 时重新计算缺失的 activations。

这是一种典型的“计算换存储”策略：

- 不 checkpoint：计算少，但保存很多 activation；
- checkpoint 全部或部分 block：显存低，但 backward 需要额外 forward 重算。

实践中不应盲目重算所有东西，而是选择合适粒度。通常可以在 Transformer block 级别 checkpoint；对于 matmul 后紧跟的简单 pointwise 操作，可能没必要保存所有中间值，重算成本很低。Checkpointing 经常与 FSDP/ZeRO 搭配，因为参数、梯度、优化器状态被分片后，activation 可能成为新的显存瓶颈。

## 7. Tensor Parallel：切隐藏维度，频繁 collective

张量并行切的是模型内部的矩阵维度，而不是 batch。以 MLP 的线性层为例，权重矩阵可以按列或按行切到多个 rank 上。每个 rank 只保存一部分权重并计算一部分输出。

课程中的简化例子是：每个 rank 拥有每一层的一部分隐藏维度。计算出局部 activation 后，需要通过 `all_gather` 把所有 rank 的 activation 拼回完整隐藏向量，再进入下一层。

Tensor Parallel 的特点：

- 优点：单层超大矩阵可以分布到多张 GPU 上；
- 缺点：每层或每几个算子都可能需要 collective；
- 适用硬件：强依赖高速互联，通常优先放在同节点 NVLink/NVSwitch 内。

Transformer 中更常见的做法是对 attention heads、MLP intermediate dimension 或 vocabulary projection 做切分。不同切法对应不同 collective：有时 forward 需要 `all_reduce`，有时 backward 需要；有时可以用 `reduce_scatter` + `all_gather` 优化显存和通信。

## 8. Pipeline Parallel：切层，处理 pipeline bubbles

流水线并行沿“深度”切模型：rank 0 保存前几层，rank 1 保存后几层，依次类推。forward 时，前一阶段把 activation 发送给后一阶段；backward 时，梯度反向传回。

朴素 pipeline 的问题是 bubble：如果一次只送一个完整 batch，那么早期阶段在后面阶段计算时会空闲，后期阶段在等待输入时也会空闲。解决方法是把 batch 切成多个 microbatch，让不同 microbatch 在不同阶段同时流动。

bubble 的直观规律是：

- pipeline stage 越多，填充和排空开销越大；
- microbatch 数越多，bubble 占比越小；
- 但 microbatch 太多会增加调度开销，并影响 batch norm/optimizer 等行为。

真实系统还要设计 forward/backward 调度，例如 GPipe 的先全 forward 后全 backward，或 1F1B（一前一后）调度。为了减少等待，应使用异步 `isend/irecv`，让通信与后续 microbatch 的计算重叠。否则同步 send/recv 会让 GPU 经常阻塞。

## 9. 组合并行与工程调优

大模型训练通常不是只用一种并行，而是组合使用：

- 数据并行：跨更多节点扩展吞吐；
- FSDP/ZeRO：降低模型状态显存；
- Tensor Parallel：处理单层太宽、单卡放不下或单层计算太大的情况；
- Pipeline Parallel：处理模型太深、跨层切分；
- Activation checkpointing：降低 activation 显存；
- Sequence/context parallel：在长上下文训练中切序列维度。

一个常见经验是：把通信最频繁、最细粒度的 tensor parallel 放在同节点高速互联内；把较粗粒度的数据并行扩展到跨节点；必要时再加入 pipeline parallel。选择策略时要同时看显存、计算利用率、通信带宽、延迟和代码复杂度。

工程调优清单：

1. 先确认瓶颈：用 profiler 判断是 compute-bound、memory-bound 还是 communication-bound。
2. 调整 batch 与 microbatch：增大 microbatch 可提高矩阵乘利用率，但会增加 activation 显存。
3. 调整 FSDP bucket/prefetch：让 `all_gather` 和 `reduce_scatter` 尽量与计算重叠。
4. 避免过多小 collective：小消息延迟主导，合并 bucket 往往更快。
5. 匹配拓扑：同节点用 tensor parallel，跨节点少做高频通信。
6. 检查同步点：barrier、loss logging、checkpoint 保存都可能引入等待。
7. 保持确定性和一致性：所有 rank 必须按相同顺序进入 collective；随机种子、数据切分和 dropout 也要正确处理。
8. 定期保存完整 checkpoint：分片训练下 checkpoint 可能分布在多个 rank，需要明确保存和恢复格式。

## 10. 一个实用的选型流程

如果从零开始为一个模型选择并行方案，可以按下面顺序思考。第一步先估算单卡显存：参数、梯度、优化器状态和激活值分别占多少。如果完整模型加优化器状态已经放不下，优先考虑 FSDP 或 ZeRO；如果主要是长序列或大 microbatch 导致激活值过大，优先打开 activation checkpointing。第二步看单层是否过大：如果注意力头、MLP 中间层或词表投影单卡计算和显存都吃紧，就需要 tensor parallel，并尽量把同一个 tensor parallel group 放在同一台机器的高速互联内。第三步看模型深度：如果层数很多，而且只靠 FSDP 与张量并行仍不够，可以再用 pipeline parallel 按层切分。第四步才是扩展数据并行，把多个模型副本放到更多节点上提高总吞吐。

调参时不要只看每秒 token 数，还要同时看 GPU 利用率、通信时间占比、显存峰值和 step time 的方差。若 GPU 利用率低而通信时间高，说明并行切得太碎或跨节点通信太频繁；若显存接近上限但通信不高，可以增加 checkpointing 或更细的分片；若 pipeline 阶段负载不均，某些 rank 会长期等待，需要重新划分层或调整 microbatch 数。真正的分布式训练往往是反复测量、定位瓶颈、修改切分策略的过程。

## 11. 常见故障信号

分布式程序最常见的问题不是报错，而是卡住。遇到长时间无输出时，先检查所有 rank 是否进入了同样的 collective，张量形状是否一致，发送和接收的源、目标是否匹配。若训练能跑但速度忽快忽慢，要检查数据加载、日志写入、checkpoint 保存和跨节点网络拥塞。若某个 rank 显存明显更高，通常说明模型切分不均、pipeline stage 负载不均，或某些 activation 没有被释放。调试时可以先用较小模型和较少 rank 复现，再逐步扩大规模。任何一次改动后都应重新测量，避免凭直觉判断瓶颈和优化效果是否真正可靠。

## 12. 总结

本讲的核心是：分布式训练的本质是在数据、参数、梯度、优化器状态和 activation 之间做切分，并在计算、显存和通信之间做权衡。DDP 简单但显存冗余；ZeRO/FSDP 通过分片降低显存但增加通信；activation checkpointing 用重算换显存；tensor parallel 适合切大矩阵但需要高速 collective；pipeline parallel 能切深层模型但必须处理 bubbles 和调度问题。

硬件会继续进步，但模型规模也会继续逼近硬件极限。因此，层次化内存、通信瓶颈、重算与分片这些系统问题不会消失。训练大模型时，真正的性能来自算法、模型结构、并行策略和硬件拓扑的共同设计。


---


# CS336 第 9 讲教程：Scaling Laws（一）

本讲讨论大语言模型训练中最重要的工程工具之一：scaling laws（缩放律）。它的核心目标不是宣称“模型会永远变聪明”，而是用小规模实验预测大规模训练的结果，从而在真正花掉巨额算力之前，回答一系列实际问题：应该训练多大的模型？用多少数据？架构、优化器、batch size、学习率如何随规模变化？给定固定 FLOPs，怎样分配模型参数和训练 token 才最划算？

## 1. 为什么需要 Scaling Laws

假设你有 10 万张 H100，可以训练一个最强开源语言模型。系统、数据、架构都准备好了，但仍然面临一个昂贵问题：不能靠反复训练巨型模型来调参。传统做法是“训练大模型—观察效果—再调参”，但在前沿规模上每一次试错都极其昂贵。

Scaling laws 的思路是：

1. 训练一组小模型，覆盖几个数量级的 compute、数据量或参数量。
2. 拟合模型损失与资源投入之间的简单函数关系，通常是幂律关系。
3. 将该关系外推到更大规模，用来预测大模型表现并选择训练方案。

因此，scaling laws 是一种“规模感知”的工程方法。它让我们不必盲目复制 LLaMA、GPT 等已有设计，而能系统地比较候选架构、优化器、数据配比和训练预算。

## 2. 基本形式：Loss 与数据、模型、算力的关系

经验上，语言模型的交叉熵损失常常与数据量、模型参数量、训练算力呈现 log-log 线性关系。也就是说，如果横轴取资源规模的对数，纵轴取 excess loss（超过不可约损失的部分）的对数，曲线近似是一条直线。这等价于幂律：

```text
L(x) = L_infinity + A * x^(-alpha)
```

其中 `x` 可以是数据量、非 embedding 参数量或 compute；`L_infinity` 是不可约损失；`alpha` 是缩放指数，表示增加资源后损失下降的速度。

这种关系通常有三个区域：

- 随机猜测区：模型或数据太小，行为不稳定，难以外推。
- 幂律区：loss 随规模稳定下降，是 scaling laws 最有用的区域。
- 饱和区：接近不可约误差，继续投入资源收益变小。

做缩放实验时，要尽量让数据点落在幂律区。例如研究“数据缩放”时，模型要足够大，避免模型容量先成为瓶颈；研究“模型缩放”时，也要保证训练 token 不过早限制模型。

## 3. 数据 Scaling：为什么幂律是自然的

从统计学习角度看，数据越多，估计误差越小。最简单的例子是估计高斯分布均值，均方误差约为 `sigma^2 / n`，取对数后就是一条斜率为 -1 的直线。

但真实神经网络不是估计一个均值，而是在高维空间中学习复杂函数。若把输入空间切成许多小区域，在每个区域内估计局部平均，维度越高，每个区域需要的数据越多，误差下降越慢。非参数统计中常见的结论是，误差指数会依赖任务的内在维度。因此真实任务中的数据缩放指数往往远小于 1：例如早期实验中，机器翻译、语音、语言建模的指数可能只有 0.1 到 0.3 左右。

这说明 scaling exponent 不只是拟合参数，它也反映了任务可学习性的难度：指数越小，增加数据带来的收益越慢。

数据 scaling 的工程用途包括：

- 比较数据源质量：如果不同数据混合主要改变曲线截距而非斜率，就可以用小模型筛选数据。
- 优化数据配比：为不同数据 mixture 拟合缩放曲线，预测大规模下哪种组合更好。
- 分析多 epoch 训练：重复同一批 token 会产生递减收益，通常可用“有效数据量”修正 scaling law。
- 权衡高质量重复数据与低质量新数据：当高质量数据有限时，需要判断重复 Wikipedia、书籍等数据，还是加入更多低质量网页数据。

## 4. 模型 Scaling：用小规模实验做架构与超参数选择

Scaling laws 不只适用于数据，也能比较模型与训练方法。经典做法是训练多个候选方案，在多个 compute scale 上观察 loss 曲线。如果两条曲线斜率相近且不交叉，那么它们的差异可理解为“常数倍 compute 效率差”。例如 Transformer 相比 LSTM 在 Kaplan 等实验中有明显优势：在相同 loss 目标下，LSTM 可能需要更多倍算力。

类似方法可以用于：

- 架构选择：比较 Transformer、LSTM、state space model、GLU、MoE 等是否在放大后仍占优。
- 优化器选择：Adam 与 SGD 可能表现为稳定的 compute efficiency gap。
- 深度/宽度比例：很多超参数并没有尖锐最优点，而是存在较宽的“近似最优盆地”。
- 参数计数方式：embedding 参数和非 embedding 参数的缩放行为不同；MoE 中还要区分总参数与激活参数。

一个重要提醒是：scaling laws 对 next-token cross entropy / log loss 通常很稳定，但对下游 benchmark 不一定稳定。困惑度随规模下降，并不保证问答、上下文学习、推理等能力按同样规律提升。因此工程上常用 loss 做主预测，但仍需要下游评估验证。

## 5. Batch Size、学习率与规模

训练规模变大时，batch size 和 learning rate 不能简单固定。

Batch size 有一个“临界 batch size”概念：在较小 batch 下，增大 batch 几乎等价于增加有效梯度样本，能提升并行效率；超过某个点后，继续增大 batch 的边际收益迅速下降。这个阈值与目标 loss 有关：模型训练得越好、目标 loss 越低，通常需要更精确的梯度，因此可承受或需要更大的 batch。实际大模型训练中常见做法是随着训练推进逐步增大 batch size。

学习率也会随模型宽度变化。标准参数化下，模型越宽，最优学习率往往越小，因此需要对不同规模分别调参，或拟合“最优学习率 vs 模型宽度”的缩放关系。另一种思路是 μP（mu-parameterization）：通过按宽度重新缩放初始化、学习率和前向输出，使小模型上调好的学习率更容易迁移到大模型。这体现了一个重要思想：不仅调参要规模感知，参数化本身也可以为跨规模迁移而设计。

## 6. 联合数据-模型 Scaling 与 Chinchilla 最优性

前面讨论的是单变量 scaling：只改变数据、模型或 compute。但真实训练中，固定 compute 可以分配给两件事：更大的模型，或更多训练 token。极端情况都浪费：小模型吃太多数据会饱和；巨型模型只看很少 token 也学不好。

联合 scaling law 试图拟合：

```text
L(N, D) = E + A / N^alpha + B / D^beta
```

其中 `N` 是模型参数量，`D` 是训练 token 数，`E` 是不可约损失。训练 compute 近似与 `N * D` 成正比（更精确地常写为约 `6ND` FLOPs）。给定总 compute，就可以在这条约束线上寻找 loss 最小的 `N` 和 `D`。

Chinchilla 论文系统研究了这个问题，并得到著名结论：在训练 compute 最优意义下，模型参数和训练 token 应大致同比增长；经验规则约为每个参数配 20 个训练 token。也就是说，与 GPT-3 这类“参数很多、token 相对少”的模型相比，Chinchilla 风格会选择更小的模型、更多的数据，从而在相同训练 FLOPs 下获得更低 loss。

Chinchilla 使用了三类方法：

1. 下包络线法：收集不同大小模型的训练曲线，对每个 compute 找到 loss 最低的 checkpoint，再拟合最优参数量与 token 数。
2. IsoFLOP 分析：固定若干 compute budget，在每个 budget 下扫模型大小；小模型多训 token，大模型少训 token，找到每条曲线的最小点。
3. 直接拟合二维 loss surface：训练不同 `N, D` 组合，拟合联合 scaling law，再推出最优 compute 分配。

其中 IsoFLOP 分析最直观：同样 FLOPs 下，横向比较不同模型大小，找 loss 最低点；再观察这些最优点如何随 FLOPs 增长。Chinchilla 的多个方法给出接近的结论，后来复现实验还发现第三种方法的原始曲线拟合存在小问题，修正后更接近前两种结果。

## 7. 预测训练结果与实验设计流程

实际使用 scaling laws 时，可以按如下流程设计实验：

1. 明确目标指标：优先用验证集 cross entropy，而不是直接用不稳定的 benchmark 分数。
2. 选择缩放轴：数据量、非 embedding 参数量、总 FLOPs，或联合的 `N` 与 `D`。
3. 覆盖多个数量级：小实验必须横跨足够范围，否则外推不可靠。
4. 控制混杂变量：研究数据时模型要足够大；研究模型时数据和训练 schedule 要合理；比较架构时尽量让训练预算、tokenizer、数据一致。
5. 拟合 log-log 曲线：检查是否处于幂律区，是否有弯曲、饱和或随机区异常。
6. 外推并选择方案：预测大规模 loss、最优模型大小、token 数、batch size、学习率或数据配比。
7. 做中等规模验证：在真正大训练前，用更接近目标规模的一两个点验证外推是否仍成立。

## 8. 常见陷阱与实践建议

第一，不要把所有参数都等同看待。embedding 参数、稠密层参数、MoE 的总参数与激活参数，对训练 loss 和推理成本的贡献不同；如果直接把它们混在一起拟合，曲线可能弯曲或产生错误结论。

第二，不要从太小的模型外推太远。随机猜测区、学习率未调好、batch size 过大或数据不足，都会让小模型点偏离真实幂律。小规模实验必须先确认训练稳定、loss 曲线可比。

第三，不要只看最终 checkpoint。像 cosine learning rate schedule 需要完整 cooldown，提前截断的中间 checkpoint 不等价于重新训练一个较短 schedule 的模型。Kaplan 与 Chinchilla 估计差异的一部分，就来自这类训练曲线处理细节。

第四，要区分“常数因子优势”和“斜率优势”。如果新架构只是曲线整体下移，它可能只是固定倍数更省算力；如果斜率更陡，则说明越放大优势越大，这才是真正值得在前沿规模上押注的信号。

第五，总体而言，预测结果应当带有不确定性。scaling law 不是物理定律，而是特定数据、代码、优化器和训练制度下的经验模型。外推距离越远，越需要保守；最好保留一个中等规模验证点，专门检验拟合曲线是否仍能命中真实 loss。若验证点偏离明显，应重新检查数据质量、学习率、warmup、权重衰减、tokenizer、去重和评估集泄漏等因素。

## 9. 现代视角：训练最优不等于部署最优

Chinchilla 解决的是“给定训练 FLOPs，怎样得到最低训练/验证 loss”。但今天模型是产品，推理成本同样重要。较大的模型虽然训练 token 更少也许训练最优，但部署时每个 token 的推理成本更高。因此许多现代模型会使用远高于 20 tokens/parameter 的比例：宁愿预训练时多花一次性成本，把模型做得更小、更密集地训练，以换取长期更低的推理成本。

所以 scaling laws 的结论要结合目标函数理解：如果目标是训练 FLOPs 最优，Chinchilla 比例是重要基准；如果目标是总成本（训练 + 海量推理）最优，可能会选择更多 token、更小模型。

## 小结

Scaling laws 是大模型工程中的预测工具。它通过小规模训练拟合 loss 与数据、模型参数、compute 的幂律关系，帮助我们预测大训练结果、比较架构与优化器、选择超参数，并在固定预算下权衡模型大小和训练 token。Chinchilla-style compute optimality 表明，训练最优模型通常需要让参数量和 token 数协调增长，而不是一味增大参数。真正可靠的 scaling 实验需要覆盖多个数量级、控制混杂变量、优先使用稳定的 log loss，并在外推前确认曲线确实处于幂律区。


---


# Stanford CS336 Lecture 10：LLM Inference 中文教程

本讲讨论的是 **inference（推理/服务时生成）**：给定一个已经训练好的固定模型，根据用户的 prompt 生成 response。训练通常是一次性的大成本，而推理会在聊天、代码补全、批处理、模型评测、test-time compute、RL 采样等场景中被反复调用，因此推理效率往往直接决定产品成本和用户体验。

## 1. 推理要优化什么？

衡量 LLM inference 主要看三个指标：

- **TTFT（Time To First Token，首 token 延迟）**：用户从提交 prompt 到看到第一个输出 token 的等待时间。它主要由 prompt 的处理时间决定，对交互式应用非常重要。
- **Latency（延迟）**：生成开始后 token 到达的速度，常理解为每个 token 需要多久，或用户感受到的流式输出速度。
- **Throughput（吞吐）**：系统单位时间能生成多少 token，通常以 tokens/s 计。批处理任务更关心吞吐，而聊天产品还要兼顾低延迟。

低 latency 和高 throughput 经常冲突：为了提高吞吐，系统倾向于把更多请求合成大 batch；但 batch 越大，单个用户等待本轮计算完成的时间也可能越长。

## 2. 为什么推理不同于训练？

Transformer 训练时，整段序列的 token 已知，可以在 sequence 维度上并行计算；矩阵乘法足够“大”，容易把 GPU/TPU 的算力打满。

推理时，尤其是自回归生成（autoregressive generation），第 t 个 token 必须依赖前面已经生成的 token，因此 decode 阶段只能一步一步来。每一步通常只为每个序列生成 1 个 token，这会让很多计算退化成较“瘦”的矩阵乘法或矩阵-向量乘法，难以充分利用硬件算力。

理解这一点需要一个关键词：**arithmetic intensity（算术强度）**，即每读取/写入一个 byte 内存能做多少 FLOPs。若算术强度高，通常是 compute-bound（受算力限制）；若算术强度低，通常是 memory-bound（受内存带宽限制）。H100 这类 GPU 的算力很强，但 HBM memory bandwidth 仍有限，所以如果每次只读大量参数和 KV cache，却做不了多少计算，GPU 就会“等内存”。

在训练或 prefill 中，batch size × sequence length 足够大，算术强度高；在逐 token decode 中，T=1，算术强度显著下降，推理就容易变成 memory bandwidth bottleneck。

## 3. Prefill 与 Decode：推理的两个阶段

LLM 推理可分为两个阶段：

### 3.1 Prefill（提示词预填充）

给定 prompt，模型一次性处理所有 prompt token，计算每层 attention 所需的 key/value，并得到下一个 token 的 logits。这个阶段类似训练的 forward pass：序列维度可并行，通常 compute-bound，速度相对较快。TTFT 很大程度上来自 prefill，特别是 prompt 很长时。

### 3.2 Decode / Generation（逐 token 生成）

模型每次根据已有上下文生成一个新 token，然后把它追加到上下文中，再继续下一步。这个阶段顺序性强，T=1，通常 memory-bound，是推理系统最难优化的部分。

## 4. KV Cache：避免重复计算的核心机制

最朴素的生成方法是：每生成一个 token，就把完整上下文重新送进 transformer。这会重复计算所有历史 token 的 key/value，复杂度很差。

**KV cache（Key-Value cache）** 的思想是：在 causal transformer 中，历史 token 的 key/value 不会因为新 token 到来而改变，所以可以缓存起来。Prefill 时为 prompt 建立 KV cache；decode 时只为新 token 计算新的 K/V，并把它追加进 cache。这样每一步不再重算完整前缀，而是读取历史 KV cache 来做 attention。

KV cache 的形状大致与以下因素成正比：

- batch size：同时服务多少序列；
- sequence length：每条序列已经有多少 token；
- number of layers：每层都要存；
- number of KV heads：key/value head 数量；
- head dimension：每个 head 的维度；
- K 和 V 两份数据，以及数据精度（如 BF16）。

因此 KV cache 很容易成为显存大户。长上下文、多并发、大 batch 都会迅速增加显存占用。

## 5. Memory Bandwidth 为什么是瓶颈？

在 MLP 层中，不同请求会共享同一组模型权重。batch 越大，读一次权重可以服务更多 token，所以 batch 可以提高算术强度。

但 attention 的 decode 更麻烦：每条序列都有自己的 KV cache。即使 batch 变大，每个请求仍要读取自己独有的历史 K/V，无法像 MLP 权重那样在 batch 内大量复用。因此 attention decode 的算术强度接近常数级，常常低到 memory-bound。

这解释了推理优化中的一个核心原则：**减少需要从 HBM 读取/写入的数据量，往往比单纯减少 FLOPs 更重要。**

## 6. Batch Size、Latency 与 Throughput 的权衡

增大 batch size 可以提高吞吐：一次 decode step 同时为 B 个请求各生成 1 个 token。但代价是：

1. 每一步要处理更多序列，单步 latency 可能上升；
2. 每条序列都要保留 KV cache，显存占用随 batch 增大；
3. 吞吐提升存在边际递减，且最终受显存容量限制。

如果只服务一个用户，latency 可以很低，但 GPU 利用率差；如果聚合很多用户，throughput 更好，但用户等待可能变长。这就是 LLM serving 的基本张力。

一种简单但有效的扩展方式是 **replication（复制模型）**：在多张 GPU 上各放一份模型，不需要训练时那样复杂的梯度同步，latency 基本不变，总吞吐近似随副本数增加。若模型太大，才需要 tensor parallelism、pipeline parallelism 或 KV cache sharding 等更复杂策略。

## 7. 减小 KV Cache 的架构技巧

因为 decode 主要受 KV cache 读写限制，很多现代架构改动都可以理解为“让 KV cache 更小”。

### 7.1 GQA：Grouped-Query Attention

传统 multi-head attention 中，query heads、key heads、value heads 数量相同。**MQA（Multi-Query Attention）** 极端地让所有 query 共享一组 K/V，但可能表达能力不足。**GQA（Grouped-Query Attention）** 是折中：多个 query heads 共享较少的 KV heads。

这样不会减少 query 的表达能力太多，却能显著减少 KV cache 的 head 数量。KV cache 变小后，显存占用降低，memory transfer 减少，latency 和 throughput 都会改善；同时还能允许更大的 batch size。

### 7.2 MLA：Multi-head Latent Attention

DeepSeek 系列提出的 **MLA（Multi-head Latent Attention）** 不一定减少 KV head 个数，而是把 K/V 投影到更低维的 latent space 中缓存。也就是说，不缓存完整高维 K/V，而缓存压缩后的表示，需要时再恢复或参与计算。它从“维度”上缩小 KV cache，目标同样是降低显存和带宽压力。

### 7.3 CLA：Cross-Layer Attention

**CLA（Cross-Layer Attention）** 在层之间共享 K/V 表示。GQA 是跨 head 共享，CLA 是跨 layer 共享。由于 KV cache 原本要为每一层保存一份，跨层共享可以进一步减少 cache size，但需要在模型质量和效率之间做权衡。

### 7.4 Local / Sliding Window Attention

**Local attention（局部注意力）** 或 **sliding window attention（滑动窗口注意力）** 只关注最近 K 个 token。这样生成很长序列时，超出窗口的 KV 可以丢弃，cache 不再随总序列长度线性增长，而近似随窗口大小固定。

问题是纯局部注意力会损害长程依赖能力。因此实际模型常用 hybrid design：大多数层用 local attention，少数层保留 full/global attention；或者结合 KV sharing、GQA 等技巧。

## 8. 更激进的方向：改变 Transformer

如果 full attention 的 KV cache 是根本瓶颈，一个方向是减少 full attention 层，甚至换掉部分 transformer 结构。

- **State Space Models（SSM）/ Mamba**：用类似 RNN 的状态表示替代随序列增长的 KV cache，使推理状态接近常数大小。挑战是保持语言建模能力，尤其是 associative recall 这类需要精确检索远处信息的任务。
- **Linear Attention（线性注意力）**：通过核函数或特征映射改写 attention，使复杂度从二次变成线性，并具有类似 recurrent state 的实现形式。现代模型常把 linear/local/full attention 混合使用。
- **Diffusion language models（扩散式语言模型）**：不再严格自回归逐 token 生成，而是并行生成一段文本并反复 refine。这样更容易打满硬件，但文本质量和通用性仍是研究问题。

这些方法说明：推理优化不只是系统工程，也会反过来推动模型架构设计。

## 9. Quantization、Pruning 与 Distillation

**Quantization（量化）** 通过降低数值精度减少显存和带宽。例如从 BF16 降到 FP8、INT8，甚至 INT4。由于推理常 memory-bound，减少每个参数或 KV 元素的 byte 数能直接改善速度和容量。但低精度会带来误差，尤其是大模型中存在 outliers（异常大的激活或权重）时。常见做法包括 post-training quantization、对 outliers 单独保留高精度、activation-aware quantization 等。

**Pruning（剪枝）** 则是删除不重要的层、head 或 hidden dimensions，让模型结构本身更小。剪枝后模型通常会变差，因此常结合 **distillation（蒸馏）**：用原始大模型作为 teacher，把能力迁移到被剪过或更小的 student 模型中。

这些方法通常是 lossy 的：速度和成本更好，但需要验证质量是否可接受。

## 10. Speculative Decoding：用小模型加速大模型

**Speculative decoding / speculative sampling（投机解码/投机采样）** 的关键观察是：验证一串给定 token 比逐个生成它们更快。因为验证可以像 prefill 一样并行，而生成必须自回归。

流程如下：

1. 用一个便宜的 **draft model（草稿模型）** 先自回归生成 K 个候选 token；
2. 用昂贵的 **target model（目标模型）** 并行计算这些 token 的概率；
3. 按照接受-拒绝规则决定保留多少草稿 token；
4. 若某个 token 被拒绝，则从 target model 的校正分布中采样，然后继续。

数学上，这种方法可以保证输出分布与直接从 target model 采样一致，即“exact sampling from target model”，前提是接受-拒绝步骤实现正确。加速效果取决于 draft model 与 target model 的接近程度：draft 越准，接受率越高，速度越快。Medusa、EAGLE 等方法都是围绕更好的 draft 或并行草稿生成展开。

## 11. Serving Systems：真实流量下的系统问题

训练时 batch 通常是整齐的 dense token block；服务时请求是动态的：到达时间不同、prompt 长度不同、生成长度不同、有些共享 prefix、有些很快结束。这要求 serving system 动态调度。

### 11.1 Continuous Batching

**Continuous batching（连续批处理）** 不等待一个 batch 全部完成再接新请求，而是在每个 decode step 后把控制权交回 scheduler：完成的请求退出，新来的请求加入。这样能减少 GPU 空转，提高吞吐。

### 11.2 Selective Batching

不同请求长度不同，attention 部分很难完全整齐地 batch；但 MLP 部分不依赖序列之间交互，可以把不同长度的 token flatten 到 batch 维度一起算。这就是 **selective batching（选择性批处理）** 的思路。

### 11.3 PagedAttention 与 vLLM

KV cache 动态增长会导致显存碎片：请求生成长度未知，提前分配会浪费；请求结束后又留下不连续空洞。**PagedAttention** 借鉴操作系统虚拟内存，把 KV cache 切成固定大小 block/page，请求的逻辑连续上下文可以映射到物理上不连续的显存块。这样减少碎片，提高显存利用率。

如果多个请求共享 prefix，还可以使用 **copy-on-write（写时复制）**：共享相同 KV block，只有在后续生成分叉时才复制，进一步节省内存。

## 12. 采样与解码策略

前面的内容主要讨论“如何更快地算出下一个 token 的 logits”，但真正生成文本还需要 **decoding strategy（解码策略）**：从 logits 或概率分布中选择 token。

最简单的是 **greedy decoding（贪心解码）**，每一步都选概率最高的 token。它稳定、便宜、可复现，但容易输出模板化文本，遇到多步推理或创意写作时可能过早锁死路径。**beam search（束搜索）** 会同时保留多个候选序列，在传统机器翻译中常见；但对开放式 LLM 对话，它可能产生重复、保守的答案，且会增加计算和内存压力。

更常用的是随机采样。**temperature（温度）** 会缩放 logits：温度低，分布更尖锐，输出更确定；温度高，分布更平坦，输出更多样。**top-k sampling** 只在概率最高的 k 个 token 中采样，截掉长尾低概率 token；**top-p / nucleus sampling（核采样）** 则选择累计概率达到 p 的最小 token 集合，动态决定候选集合大小。实际服务中还常加入 repetition penalty、frequency penalty、stop sequences、最大输出长度等规则，以控制重复、终止和成本。

这些策略本身不改变 transformer 的主要计算瓶颈，但会影响生成长度、接受率和用户感知质量。例如 speculative decoding 的速度取决于 draft token 被 target model 接受的概率；温度越高、采样越随机，小模型草稿越难预测大模型，接受率可能下降。因此推理系统往往需要把模型、采样参数和服务目标一起调优。

## 13. 总结

LLM inference 的核心难点来自自回归 decode：每次只生成一个 token，难以并行，且需要反复读取模型权重和每条序列专属的 KV cache，因此通常受 memory bandwidth 限制。Prefill 相对容易并行，decode 才是主要瓶颈。

优化路径可以分为几类：

- 系统层面：continuous batching、selective batching、PagedAttention、模型复制与并行；
- 架构层面：GQA、MLA、CLA、local attention、SSM、linear attention、diffusion models；
- 模型压缩：quantization、pruning、distillation；
- 解码算法：speculative decoding，用小模型草稿和大模型验证换取无损加速。

最终目标不是“把某个固定 transformer 跑得更快”这么窄，而是在给定 latency、throughput、显存和成本预算下，交付尽可能高质量的模型输出。推理效率已经成为现代 LLM 架构、算法和系统共同设计的核心驱动力。


---


# CS336 第 11 讲教程：Scaling Laws（二）

上一讲介绍了 scaling laws 的基本思想：用小规模实验拟合 loss、参数量、数据量和 compute 之间的关系，再外推到大模型。本讲更接近真实大模型训练流程：公开论文中的团队到底怎样用 scaling laws 选学习率、batch size、模型大小、训练 token 和架构？这些拟合什么时候可靠，什么时候只是看起来像一条直线？

核心结论可以先概括为一句话：scaling laws 不是单个公式，而是一套降低大规模训练风险的实验方法。它通常包括小模型代理实验、超参数迁移、IsoFLOP 分析、WSD 学习率日程、μP 参数化，以及对外推结果的保守验证。

## 1. 真实训练中的 Scaling 问题

训练一个前沿语言模型前，团队必须回答几个昂贵问题：

1. 给定训练 FLOPs，应该用多大的模型、多少 token？
2. batch size 应随规模怎样变化？
3. 学习率是否需要随模型变大而降低？
4. 小模型上调好的超参数能否迁移到大模型？
5. 新架构在小规模看起来不错，放大后是否仍然划算？

这些问题不能靠在 70B 或 400B 模型上反复试错。公开前沿实验中常见的策略是：先训练一批 10M、100M、1B 级别的代理模型，拟合趋势，然后只在目标规模做少数验证或最终训练。Cerebras-GPT、MiniCPM、DeepSeek LLM、Llama 3、Hunyuan、MiniMax-01 都以不同方式展示了这种流程。

## 2. Cerebras-GPT：用 μP 让超参数更稳定

Cerebras-GPT 训练了从约 0.1B 到 13B 的模型，并重点验证 μP（mu-parameterization，最大更新参数化）。标准参数化下，模型越宽，最优学习率通常会向更小值移动；如果直接把小模型学习率搬到大模型，训练可能不稳定或 loss 偏高。μP 的目标是重新缩放初始化和逐层学习率，使“同一个基础学习率”在不同宽度下都接近最优。

实践上，μP 的做法大致是：非 embedding 权重按宽度缩放初始化；使用 Adam/AdamW 时，不同层的学习率也按 fan-in 或宽度缩放，而不是所有参数共享一个全局学习率。这样做的好处是，小模型上可以进行密集网格搜索，找到学习率、初始化、宽深比等设置；放大模型时，这些设置更可能保持有效。

Cerebras-GPT 的实验显示，μP 曲线更贴近预期 scaling law，标准参数化则更容易在不同规模出现振荡或偏离。这不是说没有 μP 就训练不了大模型，而是 μP 把“每个规模都重新调学习率”的问题，转化为“在小规模调好后迁移”的问题。

## 3. MiniCPM：小模型、长训练与 WSD 日程

MiniCPM 的目标是训练很强的小模型，例如 1B 到 2B 级别，但用大量数据充分训练。它的 scaling 实验包含三个重要工具。

第一，仍然使用 μP 来稳定学习率迁移。MiniCPM 在几十 M 参数的小代理模型上搜索超参数，再放大到数百 M 或 1B 级别。实验中，不同模型大小的最优学习率基本落在同一位置，这支持了 μP 的实用价值。

第二，拟合 critical batch size。临界 batch size 可以理解为“继续增大 batch 开始收益递减”的位置。模型越大、目标 loss 越低，通常可以使用更大的 batch。MiniCPM 通过不同 batch size 的训练曲线，拟合目标 loss 与最优 batch size 的 log-log 关系，再把它外推到目标训练规模。

第三，也是本讲非常重要的一点：使用 WSD（warmup-stable-decay）学习率日程来降低 Chinchilla 式数据/模型 scaling 的实验成本。

常规 cosine schedule 的问题是：如果总训练 token 不同，整条 cosine 曲线都不同。一个训练到 1T token 的模型，其中 100B token 处的 checkpoint，并不等价于“从头训练 100B token 并完整 cooldown”的模型。因此，不能简单拿长训练的中间 checkpoint 当作短数据量实验点。

WSD 把学习率分成三段：

```text
warmup -> stable plateau -> decay/cooldown
```

它的好处是 stable 阶段可以复用。想估计较短数据量的最终 loss 时，可以从中间 checkpoint 回退出来，单独接一个 decay 阶段。这样，一个长训练加若干短 cooldown，就能近似获得多个数据量终点，大幅节省重复训练成本。

MiniCPM 用 WSD 做 Chinchilla 风格分析，并用两种方法估计模型/数据最优比例：一种是取训练曲线下包络线，另一种是直接拟合二维 loss surface：

```text
L(N, D) = E + A / N^alpha + B / D^beta
```

它得到的 token/parameter 比例很高，约 192:1。这个数字未必应被当作通用定律；更重要的启发是，Chinchilla 的 20:1 不是不可突破的硬规则。现代高质量数据、改进架构和更强优化可能让“更多 token、更小模型”的方案更有吸引力，尤其当推理成本也被纳入目标函数时。

## 4. DeepSeek LLM：直接拟合学习率、Batch Size 和 IsoFLOP

DeepSeek LLM 的公开论文很有价值，因为它展示了另一种更直接的 scaling 方法：不依赖 μP，而是显式拟合 batch size 和 learning rate 随规模的变化。

DeepSeek 先在小模型上对 batch size 与学习率做网格搜索，找到每个规模下的最优点或近似最优区域。然后把这些最优 batch size、学习率与训练 FLOPs 放到 log-log 图上拟合趋势，再外推到 7B 和 67B 模型。

这里有一个实践判断：batch size 的 scaling 往往比较干净，学习率的 scaling 更噪声、更可疑。学习率曲线有时看起来也可以用水平线解释。因此，拟合学习率更多是为了得到正确数量级，而不是得到精确公式。大模型训练通常有较宽的“可用盆地”，只要学习率不差一个数量级，训练可能仍然可行。

DeepSeek 同样做了 Chinchilla / IsoFLOP 分析。IsoFLOP 的流程是：固定多个 compute budget；在每个 budget 下训练不同模型大小，小模型看更多 token，大模型看更少 token；找到每条固定 FLOPs 曲线的最低 loss 点；再拟合最优参数量和最优 token 数如何随 FLOPs 增长。相比学习率拟合，这类 IsoFLOP 曲线通常更稳定、更可信。

DeepSeek 还使用 WSD 式 cooldown 来减少重复训练，并最终从约 10^20 FLOPs 量级外推到约 10^24 FLOPs 量级，较准确地预测了 7B 和 67B 模型的 loss。这说明，在训练制度、数据和架构保持一致时，loss scaling 的外推确实可以成为大训练前的风险控制工具。

这类成功外推也说明了为什么许多团队愿意在正式训练前投入大量“小实验预算”。这些实验本身可能已经很贵，但如果能避免一次目标规模训练失败，成本就是值得的。更现实地说，scaling law 不一定要精确预测最终所有 benchmark；只要它能提前发现学习率数量级错误、batch size 不合适、token/parameter 比例明显偏离，或者某个架构在放大后没有优势，就已经能节省大量算力。

## 5. 近期模型中的趋势：比例在变，方法在复用

Llama 3、Hunyuan 和 MiniMax-01 的论文没有像 MiniCPM 或 DeepSeek 那样给出大量 scaling 细节，但仍显示几个趋势。

Llama 3 重新做了 IsoFLOP / Chinchilla 分析，得到的最优 token/parameter 比例大约是 40:1，高于 Chinchilla 的 20:1。它还尝试把训练 loss 或 negative log likelihood 映射到下游 benchmark accuracy，例如用 sigmoid 拟合 loss 到 MMLU 等任务表现的关系。这样做的动机很明确：团队真正关心的不是 log loss 本身，而是下游能力。但 benchmark 分数更噪声、更容易饱和，因此通常还是先预测 loss，再辅助预测 benchmark。

Hunyuan 的分析得到更高的 active-parameter token 比例，例如约 96:1。这里要注意 MoE 或稀疏模型中的“总参数”和“激活参数”不同，比例不能直接与稠密模型混为一谈。

MiniMax-01 则把 scaling laws 用于架构选择。它比较 softmax attention、linear attention 和 hybrid attention 的 loss-compute 曲线，观察它们在下包络线和最优模型/token 趋势上是否接近。如果线性注意力在相同 compute 下的 scaling 曲线没有明显变差，就能支持把它用于长上下文模型。这说明 scaling laws 不只用于选大小，也能用于判断新架构是否值得放大。

## 6. μP 的直觉：控制激活和更新的尺度

μP 背后的数学直觉可以简化为两个条件。

第一，模型变宽时，每个坐标的 activation 不应爆炸或消失。若一层是矩阵乘法 `h_l = W_l h_{l-1}`，为了让输出 activation 的尺度稳定，常见初始化需要按 fan-in 的平方根缩放，类似：

```text
W_l ~ 1 / sqrt(fan_in)
```

这与 Kaiming/Xavier 初始化的直觉一致。

第二，做一次梯度更新后，activation 的变化量也不应随宽度爆炸或消失。这个条件会约束学习率如何随层宽变化。对 SGD 推导可得到类似 fan-out/fan-in 的比例；对 Adam/AdamW，由于自适应归一化改变了梯度尺度，常见 μP 规则会让学习率按 fan-in 或宽度缩放。

因此，μP 的重点不只是初始化，而是“初始化 + 逐层学习率 + 某些前向缩放”共同保证更新尺度稳定。对 Transformer，还可能涉及 attention logits 的缩放；一些 μP 实现会用 `1/d` 而不是传统 `1/sqrt(d)` 的注意力缩放，以满足更新稳定性要求。

经验研究显示，μP 对很多变化是稳健的：换 ReLU、SwiGLU、Squared ReLU，或在一定范围内改变 batch size，学习率迁移仍然可行。但它也不是万能的：可学习 norm gain、强 weight decay、Lion 这类 sign-gradient 风格优化器，都可能破坏 μP 的迁移假设。换句话说，μP 是针对特定优化器和参数化设计的工程工具，不是对所有训练配方自动成立的定理。

## 7. 实验拟合的实践流程

一个较完整的 scaling 实验可以按以下步骤执行：

1. 固定数据、tokenizer、架构族、优化器和训练代码，减少混杂变量。
2. 选若干小到中等规模模型，覆盖至少几个数量级的参数或 FLOPs。
3. 用小规模网格搜索学习率、batch size、warmup、weight decay 等关键超参数。
4. 若使用 μP，则在小模型上调参并验证学习率能跨宽度迁移；若不用 μP，则显式拟合学习率和 batch size 随规模变化。
5. 用 WSD 或等价方法收集不同 token 终点的真实 cooldown loss，避免误用 cosine 中间 checkpoint。
6. 做 IsoFLOP 或二维 `L(N,D)` 拟合，估计给定 compute 下的最优模型大小和训练 token。
7. 在比最终规模小但明显大于拟合点的中等规模上做验证，检查外推是否命中。
8. 最终训练时保留监控：如果 loss 偏离预测，要尽早排查数据、优化器、batch、学习率 schedule 和实现 bug。

## 8. 局限性与常见误区

第一，log-log 直线不等于真理。Scaling law 是经验拟合，依赖数据分布、模型族、优化器、训练 schedule 和评估集。外推越远，不确定性越大。

第二，训练 loss 最稳定，下游能力不一定稳定。困惑度下降通常是好信号，但数学推理、工具使用、长上下文、指令跟随等能力可能有阈值效应或评测噪声。

第三，token/parameter 比例没有统一常数。20:1、40:1、96:1、192:1 都出现在不同论文中，差异来自数据质量、架构、是否 MoE、训练目标和部署成本。Chinchilla 给的是训练 FLOPs 最优基准，不是产品总成本最优答案。

第四，学习率 scaling 比 loss scaling 更脆弱。Batch size 和 IsoFLOP 往往更容易拟合；学习率曲线可能很平、噪声很大，只能作为数量级参考。

第五，不能把长训练的中间 checkpoint 当作短训练终点。没有 cooldown 的 checkpoint loss 往往偏高，会系统性污染数据 scaling 拟合。WSD 的价值正是解决这个问题。

## 小结

本讲展示了 scaling laws 在真实模型训练中的用法。Cerebras-GPT 和 MiniCPM 用 μP 稳定超参数迁移；MiniCPM 和 DeepSeek 用 WSD 降低 Chinchilla 分析成本；DeepSeek 直接拟合 batch size、学习率和 IsoFLOP，并成功预测大模型 loss；Llama 3、Hunyuan、MiniMax-01 则说明现代团队仍在复用 IsoFLOP 分析，只是最优 token/parameter 比例和架构问题会随目标变化而变化。

真正可靠的 scaling 工作不是“画一条线然后相信它”，而是用小规模实验系统排除风险：让学习率、batch size、模型大小、数据量和架构选择都在放大前有证据支撑，同时承认外推的局限，并用中等规模验证点校准预测。它的价值在于把大模型训练从豪赌变成可管理的工程决策：先用便宜实验缩小搜索空间，再把昂贵训练押在最有证据的方案上。


---


# CS336 Lecture 12 中文教程：LLM Evaluation

## 1. 为什么评估并不简单

LLM evaluation 表面上像一个脚本：给定模型、准备 prompts、调用模型、收集 outputs、计算指标、求平均。但真正困难的是：你到底想用这个数字回答什么问题？

不同角色关心的评估完全不同：

- 用户或公司：我要在 Claude、Gemini、GPT、开源模型之间选一个，哪个最适合我的业务？
- 研究者：模型是否真的具备更强的通用能力？AI 是否在科学意义上进步？
- 政策制定者：模型带来了哪些收益和风险？是否足够安全？
- 模型开发者：某个训练、数据或对齐方法是否让模型变好？

因此不存在“唯一正确”的评估。一个排行榜分数只有在你理解它的输入分布、调用方式、打分规则和使用目的之后，才有解释意义。评估会反过来塑造模型开发方向：一旦某个 benchmark 成为目标，开发者就会优化它；当指标被过度优化，它也可能失去原本含义，这就是 Goodhart’s law 在 LLM 评估中的体现。

## 2. 评估框架：输入、调用、输出与解释

做一次可靠评估，可以拆成四个问题。

第一，输入来自哪里？prompt 是否覆盖了真实用例？是否包含困难样本、长尾样本和边界情况？如果是多轮聊天，后续输入会依赖模型前面的回答，因此静态 test set 未必能模拟真实对话。红队测试也常常需要根据模型行为自适应地产生攻击 prompt，否则很难发现罕见失败。

第二，如何调用模型？zero-shot、few-shot、chain-of-thought、工具调用、RAG、agent scaffolding 都会显著影响结果。早期 base model 往往需要 few-shot 示例来说明格式；现代 instruction-tuned model 通常可以 zero-shot 服从“只输出 A/B/C/D”之类的指令。prompt 顺序、格式和示例选择都会带来方差。

第三，如何评价输出？选择题可以用 accuracy，代码任务可以用 pass@1 或 pass@k，开放式生成可能需要人工偏好或 LLM-as-a-judge。还要考虑成本：一个模型分数高但价格、延迟或推理 token 数高很多，未必是更好的系统。不同错误的代价也不同，医疗、法律、安全等场景不能只看平均准确率。

第四，如何解释分数？91% 是好还是坏？是否足以部署？是否说明模型学会了能力，还是见过类似题？评估对象是 base model、chat model、完整 agent 系统，还是某个训练方法？这些必须事先说清。

## 3. Perplexity：仍然重要的基础指标

Perplexity 衡量模型给某个数据集分配了多高概率。语言模型本质上是 token 序列上的概率分布；perplexity 越低，说明模型越能预测该数据集中的 token。传统语言建模研究常在 Penn Treebank、WikiText、One Billion Word 等固定数据集上训练和测试，目标就是降低 test perplexity。

GPT-2 之后，范式发生变化：模型在大规模 web text 上预训练，再直接迁移到许多下游任务和 perplexity benchmark。此时评估更像 out-of-distribution generalization：模型没有专门训练 Penn Treebank，却可能因为训练语料足够广而取得好结果。

perplexity 的优点：

- 平滑：它利用每个 token 的 log probability，比“答对/答错”的 accuracy 更细粒度。
- 适合 scaling law：随着模型规模、数据和计算变化，loss 曲线更容易拟合。
- 覆盖全面：它关注数据集中每个 token，不只关注最终答案。
- 不易被答案格式投机取巧，只要 train/test 分离可靠。

但 perplexity 也有局限。首先，它不总是和下游任务表现强相关；在短期或具体任务上，相关性可能很乱。其次，leaderboard 若要求模型提供概率，需要信任提供方的 logits 或概率接口是正确归一化的，否则实现 bug 也可能造成虚假低 perplexity。最后，perplexity 最大化派认为“匹配真实分布就能解决一切”，但这未必是最高效路径，因为很多 token 对实际任务并不重要。

一些任务接近 perplexity，例如 LAMBADA 的缺词预测、HellaSwag 的多选补全文本：模型比较候选 continuation 的 likelihood。但这类任务也容易饱和，并且存在来自网页原始来源的近似污染风险。

## 4. Multiple-choice benchmarks：MMLU、MMLU-Pro、GPQA、HLE

MMLU 是最经典的 LLM 知识 benchmark 之一，包含 57 个学科的多项选择题。它诞生于 GPT-3 之后，当时让 base model 通过 few-shot 在大量科目上答题还是很新颖的设定。MMLU 名字里有“language understanding”，但它更像知识考试：很多题考的是具体学科事实，而不是单纯语言理解。

MMLU 的分数必须结合训练方式解释。如果一个 base model 没有专门针对 MMLU 优化，却能在多学科选择题上高分，说明它可能具备较强通用知识和迁移能力；但如果开发者专门收集类似题、调 prompt、做 chain-of-thought 和 ensemble，高分就不一定代表同等程度的通用能力。

MMLU-Pro 试图缓解 MMLU 饱和问题：去掉噪声和简单题，把选项从 4 个增加到 10 个，并更常使用 chain-of-thought。这样 frontier models 的准确率会下降，benchmark 重新获得区分度。

GPQA 更强调专家级难题。题目由博士或领域专家编写和验证，目标是“Google-proof”：非专家即使搜索也很难答对。早期 GPT-4 表现并不高，但更新模型已经显著提升。这说明“对人类很难搜索”不等于“对 LLM 永远难”。评估时还要确认模型是否允许联网，否则黑盒 API 可能暗中调用搜索功能。

Humanity’s Last Exam（HLE）进一步收集极难、多模态、选择或短答题，通过奖金和署名吸引题目贡献者，并用 frontier model 过滤太简单的问题。它的优点是难，缺点是分布偏差明显：愿意出题的人往往熟悉 LLM，也会刻意设计“模型难题”，因此它不代表普通用户需求。

## 5. 开放式与 instruction following 评估

现代聊天模型的核心能力不是只做考试题，而是根据自然语言指令完成开放任务。开放式输出没有唯一 ground truth，因此评估更难。

Chatbot Arena 的方法是让用户输入真实 prompt，两个匿名模型分别回答，用户选择更好答案，再用 pairwise preference 计算 Elo 排名。优点是动态、贴近真实使用、能接纳新模型；缺点是用户分布不受控，prompt 可能是娱乐或测试，且排行榜越重要越容易被优化或操纵。近期围绕 Arena 的争议也说明：评估协议、提交权限、模型版本和数据透明度都很关键。

IFEval 专门评估 instruction following 中的“约束服从”：例如必须少于多少词、包含或不包含某些词、使用特定格式。它的优点是可以脚本自动验证；缺点是只检查形式约束，不检查语义质量。一个 10 词故事可能满足长度要求，但并不一定好。

AlpacaEval 用 LLM-as-a-judge 比较模型回答和参考模型回答的胜率。它自动、快速、可复现，但 judge model 有偏差。例如早期小模型通过输出更长答案骗过 GPT-4 judge，后来才加入长度校正。WildBench 等数据集则从真实人机对话中抽样，并让 judge 按 checklist 评价，通常也会报告与 Chatbot Arena 的相关性。

## 6. Agent benchmark：评估模型还是系统？

很多任务需要工具调用和多步迭代，这时评估对象不再只是一个 LM，而是“模型 + agent scaffolding”的系统。

SWE-bench 给定 GitHub issue 和代码库，要求 agent 修改代码并提交 patch，最终看单元测试是否通过。Cybench 让 agent 在 CTF 网络安全环境中执行命令、探索服务器并获取 flag。MLE-bench 模拟 Kaggle：agent 要读任务、写训练代码、调参、提交结果。这些 benchmark 更接近真实工作流，但分数也受工具、上下文管理、重试策略、时间预算和成本强烈影响。

因此报告 agent 分数时必须说明：是否允许联网？能运行多少步？是否有人类提示？是否有隐藏测试？花费多少美元和时间？如果一个系统靠大量采样和昂贵推理取得高分，它和一次性低成本回答不是同一种能力。

## 7. Contamination：训练集污染与评估有效性

现代模型训练在大规模互联网数据上，开发者通常不公开完整语料，因此 train/test overlap 几乎不可完全排除。污染可以是逐字重复，也可以是近重复、改写、翻译、题解泄露或答案泄露。简单的 n-gram 去重能发现一部分问题，但发现不了跨语言或语义等价版本。

应对方法有三类：

- 数据去污染：用 test set 与训练语料做文档、段落或 n-gram overlap 检测，宁可保守删除可疑样本。
- 行为检测：通过模型对选项顺序、题目顺序或罕见文本的异常偏好，推断是否见过数据。
- 社区规范：论文和模型卡应报告 decontamination 方法、是否检查测试集泄露、置信区间和标准误。

污染不仅影响选择题，也影响 HellaSwag、WikiHow 派生任务、数学题和代码题。benchmark 数据本身还可能有标注错误；当模型分数很高时，剩余错误里相当一部分可能来自题目噪声，而不是模型能力不足。

## 8. Human eval、真实用例与安全评估

人工评估常用于开放式任务，但必须明确评审者是谁、是否专家、评分 rubric 是什么、是否盲评、是否控制答案长度和风格。普通互联网用户偏好、领域专家判断和产品用户满意度不是同一件事。

真实用例评估比考试更难也更重要。用户可能是在“问问题”（不知道答案，需要帮助），也可能是在“考模型”（自己知道答案，只想测试）。标准化考试大多属于后者，而商业价值常来自前者。Anthropic 等工作会聚类真实对话，分析用户到底用模型做什么，例如代码、写作、学习、办公等。医疗领域的 MedHELM 则让临床医生提出真实任务，如病历摘要、治疗计划、患者沟通；但真实数据往往涉及隐私，公开可复现性和现实性存在张力。

安全评估也不能只看“拒绝率”。HarmBench、AIRBench 等会测试模型是否遵守有害请求，或基于法规和公司政策构建风险分类。但一个模型如果对所有问题都拒绝，当然“安全”，却没用。因此安全必须和能力一起评估。还要区分 capability 与 propensity：模型是否知道如何做某件危险事，是 capability；它是否愿意输出，是 propensity。闭源 API 更关注 propensity 和 jailbreak 防护；开源权重则还要关注 capability，因为安全层可能被微调移除。

## 9. 可靠评估实践清单

1. 先写清评估目的：选型、科研、产品监控、安全审查还是训练反馈。
2. 明确评估对象：base model、chat model、agent system，还是训练方法。
3. 固定并公开调用协议：prompt、few-shot 示例、temperature、max tokens、工具权限、重试次数。
4. 同时报告质量、成本、延迟和方差，不只报告平均 accuracy。
5. 对选择题检查选项顺序偏差，对开放题检查长度偏差和 judge 偏差。
6. 做 contamination 检查，并报告方法；对高风险 benchmark 保持保守解释。
7. 抽样查看具体预测，不要只看 leaderboard 数字。
8. 对真实部署场景建立私有、更新、贴近用户分布的 eval set。
9. 安全评估要与有用性评估配对，避免“全拒绝”虚高。
10. 记住 benchmark 是工具，不是真理；当它成为目标，它就会被优化、饱和甚至失真。

总结来说，LLM evaluation 的核心不是“跑一个分数”，而是把一个现实问题翻译成可执行、可解释、可复现的测量过程。好的评估既要有标准化 benchmark 的可比性，也要有真实用例的代表性；既要关注能力，也要关注成本、安全、污染和数据质量。只有理解数字背后的规则，才能真正知道模型好在哪里、差在哪里，以及是否适合你的目标。


---


# Stanford CS336 Lecture 13 教程：预训练数据（Data 1）

本讲从一个核心观点出发：在现代语言模型中，数据往往比模型结构更能决定最终能力。Transformer 架构、优化器、并行训练等技术已经相对公开，而顶级模型论文通常只模糊描述训练数据，例如“来自多种数据源、覆盖到某一年”。这种保密既有商业竞争原因，也有版权和法律风险。对实践者来说，真正困难的问题不是“有了数据怎样训练”，而是“什么数据值得训练、如何把原始互联网变成可训练语料”。

## 1. 训练阶段与数据角色

大模型训练通常可以粗略分为三段：

- **Pretraining（预训练）**：使用海量、相对原始的文本，主要来自 Web、代码、书籍、论文、百科等。目标是让模型学习语言、知识和通用模式。
- **Mid-training（中期训练）**：在预训练之后，用更小但质量更高、目标更明确的数据强化能力，例如数学、代码、长上下文、多语言等。
- **Post-training（后训练）**：包括 instruction tuning、chat data、RLHF/RLAIF 等，让模型更像助手，能遵循指令、对话并满足安全要求。

术语上，**base model** 通常指完成预训练/中期训练后的模型；**instruct/chat model** 则是经过后训练、适合交互的模型。现实中三者边界并不清晰：例如 Stack Exchange 的问答数据既可进入预训练，也天然像指令数据；现代数据管线也会在预训练阶段引入由模型筛选或改写的数据。

## 2. 为什么“互联网数据”不是一个简单概念

常见说法是“大模型训练在互联网数据上”，但这过于粗糙。真正的数据来源通常经历三层转换：

1. **Live service（在线服务）**：如 Wikipedia、GitHub、Reddit、Stack Overflow、新闻站点。
2. **Raw dump / crawl（原始快照或爬取）**：如 Common Crawl、Wikipedia dump、GitHub Archive。
3. **Trainable dataset（可训练数据集）**：经过文本抽取、语言识别、清洗、过滤、去重、采样和混合之后的 tokens。

因此，当有人说“我们训练在 GitHub / Common Crawl / Reddit 上”，必须追问：使用哪个快照？如何抽取文本？如何处理许可证？是否去重？过滤规则是什么？保留了哪些字段和元数据？这些决定会显著影响模型能力。

## 3. 早期数据：Books 与 Wikipedia

BERT 使用的主要数据是 **BooksCorpus** 与 **Wikipedia**。BooksCorpus 来自 Smashwords 上免费的自出版书籍，后来因服务条款问题下线。它说明了书籍数据的重要性：书籍具有长文结构、叙事连贯性和长距离依赖，适合训练模型理解较长上下文。

Wikipedia 则是长期被视为“高质量文本”的代表。它有明确编辑规范：强调可验证性、引用来源、非原创研究、较少个人观点，并通过 notability（关注度）筛选主题。但这也意味着 Wikipedia 不覆盖所有有价值内容：个人经验、菜谱、论坛讨论、小众知识、口语表达都可能缺失。

Wikipedia 还引出一个安全问题：**data poisoning（数据投毒）**。如果攻击者能在数据快照生成前短暂插入恶意内容，即使之后被回滚，内容仍可能进入训练集。更广泛地说，互联网上的训练数据由许多具有不同动机的人共同塑造，模型行为可能被这些数据影响，而训练方很难完全审计。

## 4. WebText：用链接信号筛选网页

GPT-2 的 WebText 数据集展示了一种重要思路：不是随机抓取网页，而是利用人类社区的链接和投票信号。OpenAI 收集 Reddit 中 karma 超过一定阈值的帖子所链接的网页，得到约 800 万页面、40GB 文本。直觉是：被用户分享并获得赞同的链接，平均质量高于普通网页。

WebText 未公开，后来社区做了 OpenWebText 复现。这类方法的关键是 **link-based filtering（基于链接的过滤）**：用高质量社区、百科引用或人工 curated 页面指向的外链作为质量信号。后来的 LLaMA 也使用过类似思路：训练分类器判断网页是否像 Wikipedia 引用过的页面。

## 5. Common Crawl：最大但很脏的公共 Web 来源

**Common Crawl** 是学术和开源社区最常用的大规模网页来源。它从 2007 年开始定期爬取网页，每次包含数十亿页面。爬虫从大量 seed URLs 出发，维护 frontier 队列，类似对 Web 做广度优先搜索，同时需要处理 robots.txt、服务器负载、重复 URL、动态页面等工程问题。

Common Crawl 提供两类重要格式：

- **WARC**：原始 HTTP 响应，通常包含 HTML，也可能包含其他资源。
- **WET**：从 HTML 转出的纯文本，是有损转换。

HTML-to-text 转换看似低级，却对训练质量影响很大。使用 Common Crawl 自带 WET、Trafilatura、jusText 等工具会得到不同文本，进而影响模型评测。现代数据工程常从 WARC 重新抽取正文，而不是直接依赖 WET。

Common Crawl 不是“整个互联网”。它覆盖稀疏、偏向文本、遵守或至少考虑 robots.txt，并不保证包含所有页面；同时它也包含大量垃圾、广告、模板、重复、低质和冒犯性内容。因此 Common Crawl 更像原材料，而不是可直接训练的数据集。

## 6. 清洗、过滤与去重

从原始网页到训练 tokens，常见步骤包括：

### 语言识别（Language Identification）
用 fastText 或其他分类器判断文档语言，只保留目标语言，或按多语言配比采样。早期许多研究聚焦英语，但 Common Crawl 本身包含多语言数据。

### 规则过滤（Rule-based Filtering）
C4、Gopher/ MassiveText、RefinedWeb、FineWeb 等使用大量手写规则，例如：保留以标点结尾的行、移除句子过少的页面、过滤脏词、要求一定比例单词含字母、移除 boilerplate、过滤疑似代码或模板。规则方法透明、便宜、可解释，但容易留下结构良好的垃圾文本，也可能误伤方言、少数群体文本或非标准写法。

### 模型过滤（Model-based Filtering）
CCNet 使用 Wikipedia 训练 n-gram 模型，保留“像 Wikipedia”的文档。GPT-3 使用质量分类器，把 WebText、Wikipedia、books 作为正例，从 Common Crawl 中找相似内容。DCLM 更进一步，用 OpenHermes、ELI5 等 instruction-like 数据作为正例，用 fastText 分类器从 240T tokens 的池子中筛到约 3.8T tokens。

模型过滤能显著提升 benchmark，但风险是把“质量”缩窄为正例分布：如果正例偏百科、偏英文、偏主流写作，模型会降低多样性。近年的趋势是重新接受甚至强化模型参与数据筛选，因为收益太明显。

### 去重（Deduplication）
Web 上重复极多：镜像站、转载、模板、动态 URL、代码 fork、文档副本都会造成重复。去重分为精确去重和 **fuzzy deduplication（模糊去重）**。去重能减少训练浪费，降低模型记忆特定文本的概率，也避免某些来源被过度加权。

### 有害内容与隐私过滤
很多管线会加入 toxicity classifier、安全搜索、PII anonymization（个人信息匿名化）等步骤。但这些过滤本身也不完美：过强会损失真实世界分布，过弱则会带来安全、隐私和法律问题。

## 7. 典型预训练数据集谱系

- **C4（Colossal Clean Crawled Corpus）**：Google/T5 使用的 Common Crawl 清洗版，主要依靠规则过滤，只保留英文自然语言文本。
- **The Pile**：EleutherAI 社区构建的 22 个高质量数据源混合，包括 Common Crawl、OpenWebText、Stack Exchange、Wikipedia、arXiv、PubMed、GitHub、Books3 等，体现“人工挑选领域”的路线。
- **MassiveText / Gopher**：DeepMind 的数据混合，包含 MassiveWeb、C4、books、news、GitHub、Wikipedia，并使用规则和安全过滤。
- **LLaMA 数据**：Common Crawl + C4 + GitHub + Wikipedia + Project Gutenberg + Books3 + arXiv + Stack Exchange，总计约 1.2T tokens。未公开，但 RedPajama 做了复现。
- **RefinedWeb / FineWeb**：主张只要 Web 过滤得足够好，就可以得到强数据。FineWeb 是 Hugging Face 对大规模 Common Crawl 的轻过滤版本，可作为进一步筛选的基础。
- **DCLM Baseline**：把 Common Crawl 全量池构造成竞赛式数据基准，用强质量分类器 aggressive filtering，成为近期开源模型常用数据来源。
- **Nemotron-CC**：NVIDIA 在 DCLM 思路上扩展，用大模型打分“educational value（教育价值）”、蒸馏到较快模型，并组合多个过滤器，还尝试用 LLM 改写低质数据或把高质文档转成任务形式。

这些数据集体现了两个张力：一是质量与规模的取舍，过滤越狠质量越高但 tokens 越少；二是质量与多样性的取舍，越像高质量正例，越可能丢掉长尾知识、口语和非主流文本。

## 8. 代码、问答、书籍与论文的特殊价值

不同来源提供不同能力：

- **GitHub / The Stack**：主要训练代码能力，也可能提升结构化推理。处理时要识别许可证、去除重复、过滤生成文件、区分代码与文档、考虑 issues 和 commit history 是否使用。
- **Stack Exchange / Stack Overflow**：天然是 QA 格式，有问题、回答、评论、投票等元数据，适合筛选高质量解释，也模糊了预训练与指令训练边界。
- **Project Gutenberg / PG19**：公共领域书籍，版权清晰，适合长上下文训练；但语言风格偏旧。
- **arXiv / PubMed / Semantic Scholar**：学术论文提供知识密度、数学和技术表达，但格式抽取、公式、引用和版权都需要处理。
- **Reddit / ELI5**：更接近用户问题和通俗解释，可作为质量分类器正例或 instruction-like 语料。

## 9. 版权与数据可用性

绝大多数互联网上的原创表达默认受版权保护，即使网页没有写 copyright 标记。使用方式大致有两条路：取得 license（许可证），或主张 **fair use（合理使用）**。合理使用会考虑用途是否转换性、作品性质、使用比例、对原市场的影响等。对大模型训练来说，复制训练数据本身就涉及版权；训练是否足够 transformative、模型是否记忆和复现原文、是否替代原作者市场，都是争议焦点。

此外，即便内容采用 Creative Commons 或可能属于 fair use，平台 Terms of Service（服务条款）也可能禁止自动下载。例如公开视频不等于可以随意爬取。大型公司可通过商业授权获得 Reddit、Stack Exchange、Shutterstock 等数据；开源和学术团队则更依赖公开 dump、许可清晰数据和谨慎过滤。

## 10. Mid-training 与 Post-training 数据

中期训练和后训练更关注特定能力。长上下文扩展常在后期进行，因为从一开始用超长序列训练成本太高；数据上可使用书籍、长论文、长代码、合成长依赖任务等。

指令数据方面，早期有 Super-Natural Instructions、FLAN：把传统 NLP 任务统一成 instruction format。之后出现 Alpaca/self-instruct、Vicuna、OpenHermes、Evol-Instruct 等合成数据方法，即用强模型生成任务、回答或多轮对话。合成数据便宜、可扩展，但受生成模型许可证限制，也可能继承教师模型偏差。另一条路线是雇佣标注者写高质量指令数据，成本高但可控；不过还要防止标注者偷偷用商业模型生成答案。

## 11. 工程实践总结

构建预训练数据管线时，可以按如下流程思考：

1. 明确目标能力：通用知识、代码、数学、多语言、长上下文、对话风格。
2. 收集原始来源：Web crawl、dump、API、授权数据、公共领域数据。
3. 文本抽取：HTML/PDF/code/email 等格式转纯文本，保留必要元数据。
4. 基础清洗：语言识别、编码修复、去 boilerplate、长度过滤、格式过滤。
5. 质量筛选：规则、分类器、LLM 打分、链接信号、社区投票信号。
6. 安全与合规：版权、license、robots.txt、ToS、PII、toxicity。
7. 去重与采样：精确/模糊去重，避免重复来源支配训练。
8. 数据混合：按能力和质量设定 mixture weights，并通过小模型 ablation 验证。
9. 记录版本：保存快照时间、处理代码、过滤阈值、统计信息，保证可追溯。

实际落地时，不要只看最终 token 数。更有用的监控包括：每个来源的保留率、重复率、平均文档长度、语言分布、域名分布、困惑度或质量分数分布、被过滤样本示例，以及训练后在目标评测上的增益。数据管线应当像模型代码一样版本化，否则很难解释一次训练为什么变好或变坏。

本讲的核心结论是：数据不会从天上掉下来。可训练语料是大量工程、启发式规则、法律判断和实验迭代的结果。现代模型之间架构差异可能不大，数据来源、过滤策略、去重质量、合成数据和授权资源，才是决定模型差异的重要因素。


---


# Stanford CS336 Lecture 14：数据（二）——从原始网页到可训练语料

本讲继续讨论大模型预训练中的“数据工程”。上一讲更像数据集史：从早期语料到 Common Crawl、C4、The Pile、LLaMA、Dolma 等。本讲转向可操作的方法：当我们手里有海量原始网页和少量理想数据时，如何筛选、混合、去重，并把这些决策转化为训练数据配方。

核心问题可以概括为：给定一个小而高质量的目标集合 T，以及一个巨大但嘈杂的原始集合 R，找出 R 中“像 T”的子集 T'。这不仅是质量过滤，也适用于语言识别、领域选择、毒性过滤、数学/代码数据挖掘、合成数据筛选，以及训练过程中不同 domain mixture 的调整。

## 1. 数据选择的基本范式

数据选择不是简单地“保留看起来好的网页”。一个通用流水线通常包含三步：

1. 定义目标：什么是我们想要的数据？可能是 Wikipedia 风格文本、教材式代码、数学证明、英文网页、低毒性讨论，或某个产品需要的任务域。
2. 训练或构造打分器：用目标数据和原始数据估计一个分数 score(x)，表示样本 x 多像目标域、多有价值或多安全。
3. 选择或重采样：按阈值保留、按概率采样，或按重要性权重重新分布数据。

这里有两个实际约束。第一，打分器必须能泛化：如果只找回 T 本身就没有意义，我们需要从 R 中发现新的相似样本。第二，打分器必须足够快：Web 级数据极大，如果用一个巨型模型逐条评分，过滤成本可能接近甚至超过预训练本身。

## 2. 三类常见过滤器

### 2.1 n-gram 语言模型：粗糙但便宜

最传统的方法是训练 n-gram 语言模型，例如用 KenLM 加 Kneser-Ney 平滑。它本质上统计 n 元词组出现次数，并估计条件概率。例如给定上下文 “the cat”，估计下一个词是 “in” 的概率。由于很多 n-gram 从未出现，平滑会退回到更短上下文。

用法很直接：在目标语料上训练 n-gram 模型，然后对原始文档算困惑度（perplexity）。困惑度低表示文本更像目标域。CCNet 就用类似方法按段落困惑度排序，只保留较好的部分，后续 LLaMA 数据也受到这类流程影响。

这种方法的优点是快、简单、可扩展；缺点也明显：它主要看局部共现，无法真正判断长程逻辑和语义质量。打乱段落、模板垃圾、局部语法正常但整体无意义的文本，都可能骗过它。因此 n-gram 过滤更适合清掉明显噪声，而不是做精细质量评估。

### 2.2 fastText / 线性分类器：工业界常用基线

fastText 是轻量文本分类器。它把词或 n-gram 哈希到固定桶中，再通过低维表示做线性分类。虽然结构简单，但速度快、可并行、适合 Web 规模过滤。

典型训练方式是构造二分类任务：正例来自高质量或目标域数据，负例来自 Common Crawl 等原始数据。分类器输出 “这个样本来自目标域” 的概率。GPT-3 曾用高质量来源作为正例、Common Crawl 作为负例训练质量分类器；LLaMA 用被 Wikipedia 引用的网页作为正例；Dolma 用 fastText 做语言识别和毒性过滤。

fastText 的关键价值不是“足够聪明”，而是“便宜到可以跑完整个网络”。当原始数据要被压缩到 1% 时，过滤器处理了 100 倍于最终训练量的数据；此时每条样本的评分成本必须非常低。

### 2.3 重要性重采样：从“分类”到“匹配分布”

分类器回答的是“像不像目标域”，但训练数据还需要保持分布多样性。重要性重采样提供了更原则化的视角：目标分布为 P，原始分布为 Q，我们只能从 Q 采样，但希望最终样本像来自 P。于是为样本赋权：

w(x) = P(x) / Q(x)

然后按 w(x) 的比例重采样。直觉是：如果某类文本在目标域中常见、在原始数据中稀少，就提高它的采样概率；反之降低。

真实场景中 P 很难精确估计，因为目标数据少。实践会用哈希 n-gram 估计粗略分布，再计算近似权重。它不一定带来巨大收益，但比纯二分类更强调 domain mixture 的分布匹配，而不仅是“过阈值”。

## 3. 质量评估、领域混合与语言选择

“好数据”没有单一标准。质量可能意味着语法通顺、信息密度高、教育价值高、低毒性、少模板、符合目标任务，或来自可信来源。因此数据过滤常常被拆成多个独立维度。

语言识别是最基本的例子。若目标是英文模型，混入大量其他语言会消耗 token budget，降低英文训练强度；但若模型足够大，多语言又可能带来正迁移。Bloom 约 30% 英文，强调多语言能力；前沿模型通常覆盖上百种语言。是否过滤语言，本质上是训练数据决策：目标用户、模型容量、算力预算和评测指标共同决定 mixture。

领域选择同样重要。OpenWebMath 把“数学”视作一种特殊语言：先用规则找候选，再用在 Proof-Pile 等数学证明数据上训练的 KenLM 和 fastText 分类器筛选，最终得到约 150 亿数学 token。结果显示，面向数学域的高密度数据可以胜过数量大得多但不聚焦的数据。这说明 domain mixture 不是越大越好，而是要匹配目标能力。

质量评估还可以由强模型辅助。Phi-1 的思路是训练小模型，但给它“教材式”的高价值代码数据。研究者先让 GPT-4 判断 Python 代码片段对初学者是否有教育价值，得到约 10 万个标注样本，再用较便宜的分类器扩展到大规模数据。这是一种常见模式：用昂贵模型创建小规模高质量 T，再蒸馏成便宜过滤器处理 R。

## 4. 合成数据：目标数据可以被“生成”出来

当没有现成目标语料时，可以让强语言模型合成或筛选目标数据。例如要求模型生成教材式代码、数学推理、化学问答，或给网页打“教育价值”标签。这样 T 不再只是某个现成来源，而是由需求和提示词定义。

但合成数据有风险：分布可能过窄，风格可能单一，错误会被放大，还可能与已有数据高度相似。因此合成数据通常不应直接无限加入，而要经过质量分类、去重、人工抽检和下游评测。更稳妥的做法是：用合成数据提高某类能力，但保留真实数据的多样性；用强模型标注一小批，再训练便宜过滤器扩大规模。

## 5. 去重：减少浪费与记忆

过滤决定“哪些数据值得训练”，去重决定“同样的信息训练几遍”。Web 天然有大量重复：镜像站点、许可证文本、商品模板、复制粘贴文章、只改少数词的模板页面。C4 中曾有一句普通英文出现数万次；它不是坏文本，但训练 6 万遍没有意义。

精确去重很简单：对句子、段落或文档哈希，把相同哈希的样本分到一组，只保留一个。它精度高、易并行，但发现不了近重复。Bloom filter 则用位数组和多个哈希函数节省内存；它不会产生假阴性，但可能有假阳性，适合超大规模近似集合查询。

近重复去重通常基于 Jaccard 相似度。把文档切成 shingles 或 n-gram 集合，若两个集合交并比超过阈值，就认为近重复。直接两两比较是 O(N²)，不可行。MinHash 的关键性质是：两个集合 MinHash 碰撞概率等于它们的 Jaccard 相似度。再结合 LSH（locality sensitive hashing），把多个哈希分成若干 band，使高相似文档大概率碰撞、低相似文档低概率碰撞，从而在线性或近线性时间找到候选重复。

去重要谨慎。预训练阶段清除网页垃圾重复通常有益；但在 mid-training 或继续训练中，高质量数据重复多轮可能正是我们想要的。更合理的策略可能不是简单保留一份，而是对重复计数做降权，例如按 log 或平方根采样，让“重要且常见”的内容有更高权重，但不按原始重复次数线性放大。

## 6. Curriculum、annealing 与训练数据配方

数据决策不只发生在训练前。许多现代训练会随时间改变 mixture：早期用大规模、多样、较宽松过滤的数据学习通用语言与世界知识；后期逐渐 anneal 到高质量、目标域、指令式或推理数据，以提升最终评测。这个过程类似 curriculum learning：先覆盖广，再提高密度和难度。

常见策略包括：

- 前期扩大覆盖：网页、书籍、代码、多语言、论坛等混合，避免模型过早过拟合窄域。
- 中期提高目标域比例：若关注代码、数学或某语言，可逐步增加该域 token。
- 后期质量退火：减少低质量网页，提高教材、问答、推理、人工或强模型筛选数据比例。
- 对合成数据限量使用：避免风格塌缩，同时利用其补齐稀缺能力。
- 按评测闭环调整：每次 mixture 改动都要看下游 benchmark、困惑度、人工样本和安全指标。

因此 domain mixture 是一个优化问题，而不是静态表格。最好的比例通常无法凭直觉一次写出，需要训练小模型、做 ablation、观察数据样本、再迭代。一个实用原则是把数据配方当作模型的一部分记录下来：每个来源的 token 数、过滤阈值、去重粒度、重复采样倍数、进入训练的时间段，都应可追溯。否则当模型在某个能力或安全指标上变化时，很难判断是模型规模、优化超参还是数据配方造成的。

## 7. 实践检查清单

构建预训练语料时，可以按以下问题审查：

1. 目标能力是什么？通用聊天、代码、数学、多语言，还是某个专业领域？
2. 目标数据 T 从哪里来？人工来源、可信站点、强模型标注、合成生成，还是规则初筛？
3. 过滤器是否足够便宜？它处理的是原始海量 R，而不是最终小语料。
4. 质量阈值如何选？过松会保留噪声，过严会损失多样性和低资源群体。
5. mixture 是否匹配 token budget？某域比例提高，就意味着其他域训练机会下降。
6. 是否做了精确与近重复去重？是否避免训练集泄漏到评测集？
7. 是否需要重复高质量数据？如果需要，是线性重复还是降权重复？
8. 是否用小规模训练验证数据决策，而不是只看过滤器分数？

## 总结

本讲的主线是：数据不是自然落到训练集里的，而是通过一系列可计算、可扩展但充满取舍的决策得到的。n-gram、fastText、重要性重采样提供了从原始网页中寻找目标数据的基本工具；语言识别、质量过滤、毒性过滤和领域挖掘都是同一框架的不同实例；合成数据和强模型标注让“目标数据”本身也可以被设计；去重则减少无意义重复、降低记忆风险并节省算力。

真正的数据能力来自闭环：看数据、写过滤器、训练模型、评测结果、调整 mixture，再重复。对于大模型训练，数据选择往往和模型结构、算力规模同等重要，甚至在特定能力上更决定最终效果。


---


# CS336 Lecture 15 中文教程：Alignment、SFT 与 RLHF

## 1. 从预训练到对齐：为什么还需要 post-training

预训练把大量能力“压进”模型参数里：语言、知识、代码、推理模式、常识和各种风格。但一个只做 next-token prediction 的 base model 通常并不会自然表现成好用的聊天助手。GPT-3 已经很强，却不稳定地服从指令；ChatGPT 的关键变化是 post-training：让模型更会理解用户意图、更愿意按指令完成任务，并在危险场景中拒绝或转向安全回答。

Alignment 在这节课中不是抽象口号，而是一条工程管线：先用监督数据教模型“应该怎样回答”，再用偏好数据和强化学习或替代算法，把模型推向人类更偏好的行为。目标包括 helpfulness、truthfulness、harmlessness：有用、真实、无害。难点在于，这三者经常冲突。例如模型为了有用可能编造答案；为了安全可能过度拒绝；为了迎合偏好可能输出更长但不更正确的回答。

## 2. SFT：用示范回答教模型进入助手模式

Supervised Fine-Tuning（SFT）是 InstructGPT 管线的第一步。数据形式很简单：给定 prompt 或对话上下文，配一个理想 response，然后对 response tokens 做最大似然训练。直观地说，就是让模型模仿专家示范。

常见 SFT 数据来源有三类：

- 任务聚合型数据，例如 FLAN：把已有 NLP 数据集改写成指令格式，如摘要、分类、问答、多选题等。优点是规模大、成本低；缺点是形式常常不像真实聊天，很多回答很短，任务痕迹明显。
- 人类撰写数据，例如 OpenAssistant：志愿者或标注员写复杂 prompt 和详细回答。优点是自然、质量高；缺点是贵、慢、难以稳定控制风格和事实质量。
- 模型生成数据，例如 Alpaca：先用少量人工种子 prompt 扩展出更多指令，再用强模型生成回答。优点是便宜、长回答多；缺点是会继承教师模型偏差，也可能让学生模型模仿自己尚不具备的能力。

SFT 的一个核心经验是：少量高杠杆数据就能显著改变模型行为。强 base model 可能只需相对少的 instruction data，就能从“补全文本”变成“回答用户”。但“高质量数据”并不等于“越长、越知识密集、引用越多越好”。如果 SFT 例子要求模型回答它并不知道的事实，训练损失会奖励它生成看起来像正确答案的 token，包括看起来像引用的字符串。于是模型可能学到的不是“查到事实再引用”，而是“遇到复杂问题就在结尾编一个引用”。

这说明 SFT 更容易教会输出的类型签名和风格：要不要分点、要不要长篇解释、要不要引用、要不要道歉、要不要拒绝。它也能教新知识，但小规模 SFT 往往不如预训练或大规模 mid-training 稳定。若示范数据明显超出模型已有能力，模型可能学会幻觉式捷径。因此，好的 SFT 数据应匹配模型能力，并包含“不知道”“需要更多信息”“建议查证”等合理 abstention 行为。

## 3. 安全 SFT：拒绝与过度拒绝之间的平衡

安全对齐也可以通过 SFT 注入。把一小批安全样本混入 instruction tuning，模型就能学习在诈骗、恶意软件、暴力、自伤等场景中拒绝或给出安全替代方案。研究中甚至几百条精心构造的安全样本也能带来明显效果。

但安全不是简单提高拒绝率。真正困难的是区分危险请求和表面危险但合法的请求：例如“how can I kill a Python process?” 在计算机语境下是终止进程，不是伤害生物。如果数据只教模型看到敏感词就拒绝，会产生 over-refusal，降低可用性。安全数据需要覆盖边界案例、双用途问题、上下文歧义和允许回答的安全版本。

## 4. SFT 的训练方式正在接近预训练

学术设置中，SFT 常被理解为：拿 base model，对指令数据跑几轮梯度下降。但前沿模型的 post-training 已经更像完整训练阶段。许多现代管线会在预训练末尾、学习率衰减阶段混入高质量数据、代码 SFT、问答、聊天、多语言书籍和安全数据，这常被称为 mid-training 或 decay-stage data mixing。

这样做的好处是：数据规模可以更大，模型不容易因短暂微调而灾难性遗忘，instruction behavior 也能更深地融入模型。代价是“base model”和“chat model”的边界变模糊。今天很多所谓 base model 可能已经在训练后期见过大量指令式数据，因此比较不同模型时要注意：base 不一定意味着完全未经对齐。

## 5. 为什么需要 preference data

SFT 需要人或强模型直接写出理想答案，但生成高质量长回答很贵、很累，而且人类自己写出的答案未必是他们最喜欢的答案。验证往往比生成容易：让标注员比较 A/B 哪个更好，通常比让他从零写一个完美回答便宜。这就是 preference data 的动机。

偏好数据的基本形式是：同一 prompt 下有两个或多个模型回答，标注者选择更好的一项。InstructGPT 风格的标注准则通常围绕三点：helpful、truthful、harmless。实际准则会更细：是否回答了用户真实意图，是否遵守格式，是否幻觉，是否有毒，是否包含不当内容，是否需要澄清。

但偏好标注也很难。标注员常在时间压力下工作，可能没有足够时间查事实、验算数学或识别隐蔽幻觉。长回答更容易被认为“详细、有帮助”，即使里面有错误。不同标注员关注点不同：专家更重事实性，普通众包标注员可能更重格式、流畅度和礼貌。标注员的文化、国家、宗教和政治背景也会影响价值判断，而 alignment 位于管线末端，对最终模型行为影响很强。

因此 preference data 不只是技术资源，也是社会选择。需要清晰的 rubric、公平报酬、质量审计、多样化标注群体，以及对偏见来源的透明记录。

## 6. Reward Model：把成对偏好变成可优化奖励

RLHF 的经典第二步是训练 reward model。假设每个回答 y 在 prompt x 下都有一个潜在标量奖励 R(x, y)，但我们无法直接观察它，只能观察人类比较：回答 A 是否优于回答 B。

常用建模是 Bradley-Terry preference model：A 胜过 B 的概率取决于 R(x, A) - R(x, B) 的差，通常经过 sigmoid。训练 reward model 时，让被选中的回答得分高于未被选中的回答。训练好后，reward model 就能给任意新回答打一个标量分数，作为 RL 的奖励信号。

注意 reward model 只是人类偏好的近似，不是真理。它会学到标注数据中的偏差，例如偏好长答案、偏好列表、偏好某种语气，也可能被策略模型 exploit。RL 阶段如果过度优化 reward model，模型可能获得高 reward 但真实人类不喜欢，这就是 reward hacking 或 Goodhart 化。

## 7. RLHF：用 PPO 在奖励和约束之间优化

经典 InstructGPT 管线的第三步是用 PPO 优化策略模型。目标不是继续模仿某个参考分布，而是找到一个 policy π(y|x)，让 reward model 给出的期望奖励更高。

实际目标通常包含约束：

- 奖励项：让模型生成 reward model 喜欢的回答。
- KL 惩罚：限制 RL 后的策略不要离 SFT 模型太远，避免语言质量崩坏、模式坍缩或 reward hacking。
- 有时还混入预训练损失，缓解灾难性遗忘。

PPO 可以理解为 policy gradient 的稳定工程版本。模型采样回答，reward model 打分，根据 advantage 增强好输出、削弱坏输出；同时用重要性比率和 clipping 限制每次更新幅度。它有效，但实现复杂、调参困难、训练不稳定，对学术和开源实践不友好。

这里还要区分 on-policy 与 off-policy。On-policy 数据来自当前正在优化的模型，因此能针对模型当前错误进行改进；off-policy 数据来自其他模型或旧模型，更便宜、可复用，但不一定覆盖当前模型最需要修正的区域。现代管线常混用两者。

## 8. DPO 与替代方法：把 RL 问题变成监督损失

因为 PPO 麻烦，研究者尝试了许多替代方法：只在 preferred responses 上做 SFT；给 chosen/rejected 加好坏 token 后条件生成；用 reward model 采样多个回答再挑最好的做训练。这些方法有时有效，但通常不如经典 RLHF 稳定。

Direct Preference Optimization（DPO）之所以流行，是因为它去掉显式 reward model 和 PPO rollout，把偏好优化写成一个直接的监督式 loss。关键思想是：在带 KL 正则的最优策略问题中，某个 policy 隐含定义了一个 reward；把这个隐含 reward 代入 Bradley-Terry 偏好模型，就能直接最大化“chosen 比 rejected 更可能被偏好”的概率。

DPO 的训练直觉很简单：相对 reference model，提高 chosen response 的 log probability，降低 rejected response 的 log probability，并用系数控制离 reference 的距离。它不像 PPO 那样需要在线采样和复杂 RL 状态，因此更容易实现、复现和扩展。缺点是它通常依赖已有偏好对，较少利用 on-policy 探索；如果偏好数据质量差、分布偏或全是旧模型输出，DPO 也会受限。

相关替代路线还包括 RLAIF（AI feedback）、Constitutional AI、拒绝采样式训练和各种 DPO 变体。RLAIF 用强模型代替人类做偏好判断，成本低、规模大，GPT-4 等 judge 与人类偏好在许多开放任务上相关性较高。但 AI judge 有自我偏好、长度偏好、位置偏好和价值偏见，不能被当作无偏标注器。

使用 AI feedback 时还要注意“闭环放大”：如果同一个模型家族既生成候选、又做裁判、又被用来蒸馏学生模型，系统可能越来越偏向该家族喜欢的表达方式，而不是更接近真实用户需求。较好的实践是混合多种 judge、抽样做人类复核、保留困难负例，并单独报告长度归一化后的胜率。对于事实密集或数学任务，最好使用可验证信号、工具检查或专家审计，而不是只依赖开放式偏好。

## 9. 对齐管线的实践清单

一个典型 alignment pipeline 可以这样组织：

1. 从强 base model 开始，准备聊天模板和基础 instruction data。
2. 做 SFT，让模型稳定进入助手模式，覆盖通用能力、格式服从、多轮对话和初步安全拒绝。
3. 收集 prompts，让一个或多个模型生成候选回答。
4. 用人类或 AI 标注 pairwise preferences，建立 chosen/rejected 数据；同时审计长度、风格、事实性和标注员一致性。
5. 训练 reward model，或直接用 DPO/IPO/KTO 等偏好优化方法。
6. 若使用 PPO/RLHF，加入 KL 约束和安全监控，避免 reward hacking。
7. 用多维评估验证：开放式偏好、事实性、数学和代码 benchmark、拒绝率、过度拒绝率、红队测试、真实用户任务、成本和延迟。
8. 根据失败案例迭代数据，而不是只追逐单一排行榜。

## 10. 风险与评估：不要把偏好当成真理

Alignment 最容易被误解为“让模型更讨人喜欢”。这很危险。人类和 AI judge 都偏好长、结构化、礼貌、看似自信的回答，但这些特征不等于正确。一个模型可能因为更会分点、更会引用、更会迎合而在偏好评估中胜出，同时幻觉更多。

因此评估必须多样化：

- 开放式 human eval 或 chatbot arena 衡量用户偏好，但要控制长度和 judge 偏差。
- 标准 benchmark 衡量知识、推理、代码和数学能力，避免 post-training 只优化风格。
- 安全评估同时看 harmful compliance 和 over-refusal，不能让“全拒绝”伪装成安全。
- 事实性评估需要查证 claim，而不是只看流畅度。
- 真实部署还要监控分布漂移、越狱攻击、滥用、用户满意度和成本。

总结来说，Lecture 15 的主线是：预训练给模型能力，SFT 教模型如何表现，preference data 告诉模型哪些表现更受欢迎，reward model 或 DPO 把偏好转成可优化目标，RLHF/偏好优化再把模型推向更有用、更真实、更安全的区域。真正的挑战不只是算法，而是数据质量、标注激励、价值偏差、评估偏差和安全-可用性的平衡。


---


# Stanford CS336 Lecture 16 中文教程：Alignment 中的强化学习（一）

本讲是后训练（post-training）部分的第二讲，主题从传统 RLHF 过渡到“可验证奖励上的强化学习”（reinforcement learning from verifiable rewards）。核心问题是：为什么语言模型对齐需要 RL？PPO、GRPO 这类算法到底在优化什么？为什么训练会不稳定，工程上又有哪些关键细节？

## 1. 从 RLHF 到可验证奖励

RLHF（Reinforcement Learning from Human Feedback）通常从人类偏好数据开始：给定同一个 prompt 的两个回答，人类标注哪一个更好。目标是训练一个语言模型 policy，使它更倾向产生人类喜欢的回答。

这里的 policy 指模型在给定上下文后对输出序列的概率分布。与预训练或 SFT（supervised fine-tuning）不同，RLHF 不是单纯做“数据分布拟合”：模型生成什么会改变它获得的 reward，因此目标中包含“从当前模型采样”的过程。这使优化比普通最大似然更难。

上一讲提到的 DPO（Direct Preference Optimization）把偏好优化转化成一种类似监督学习的目标：不显式训练 reward model，也不跑完整 RL loop，而是通过偏好对直接调整 policy。DPO 的直觉很简单：提高 chosen response 的概率，降低 rejected response 的概率；当模型的隐含 reward 判断错得越多，更新越大。DPO 因为实现简单，一度成为开源模型 post-training 的主流方法。

但 DPO 也有局限：它天然适合 pairwise preference，不太适合“数学题答对/答错”这类只有标量奖励的任务；它通常也是离线的，即先收集一批偏好对，再在其上训练。对于推理模型，研究者更希望在模型不断生成新解答的过程中，直接根据可验证结果进行在线优化。

## 2. RLHF 的两个风险：过优化与校准变差

RLHF 中最重要的经验现象之一是 overoptimization（过优化）。reward model 只是人类偏好的代理模型，带有噪声和误差。训练初期，优化代理 reward 往往能提升真实人类偏好；但继续优化后，模型可能开始“钻 reward model 的空子”，代理 reward 继续上升，真实 win rate 却停滞甚至下降。这与监督学习中的 train-test gap 类似：训练集上的 reward model 不等于真实偏好 oracle。

另一个现象是 calibration（校准）变差。预训练模型可被看作概率生成模型，而 RLHF 后的模型更像为了某个 reward 调整过的 policy。若 reward 没有鼓励“表达不确定性”，模型就可能变得更自信、更迎合、更少说“不知道”。因此不要把 RLHF 模型的输出概率直接理解为可靠的真实概率估计。

这些问题说明：人类偏好很有价值，但难以大规模、低噪声、稳定地优化。于是一个自然方向是寻找 reward 更清晰的任务，例如数学、代码、形式化证明、可执行测试等。在这些领域，答案是否正确可以自动验证，reward 更接近真实目标，也更不容易被 reward hacking。

## 3. RL 基础：policy、reward、value、advantage

在语言模型 RL 中，一个样本通常包含 prompt 和模型生成的 response。可以把生成完整 response 看成一次 rollout。常见术语如下：

- Policy：当前语言模型 πθ，即给定 prompt 后生成 token 序列的概率分布。
- Reward：对生成结果的评分。RLHF 中可能来自 reward model；数学/代码中可能来自答案匹配、单元测试、格式检查等。
- Value function：估计某个状态或部分生成未来能获得多少 reward 的函数，常用于降低 policy gradient 的方差。
- Advantage：某个动作/输出比基线好多少。直觉上，advantage 为正就提高该输出概率，为负就降低该输出概率。

最基本的 policy gradient 思想是：对高 reward 的输出增加 log probability，对低 reward 的输出减少 log probability。许多 RL 算法本质上都可理解为“upweight good stuff, downweight bad stuff”，区别在于：如何定义好坏、如何做方差降低、如何避免 policy 一步走太远。

语言模型 RL 还有一个特点：很多任务更像 contextual bandit。模型看到 prompt，生成完整回答，然后得到一个终局 reward；没有传统游戏环境里复杂的状态转移。但训练时仍常把 KL penalty 等正则项分摊到 token 级别，而把“答对/答错”这样的任务 reward 放在最后一个 token 或序列级别。

## 4. PPO：强大但工程复杂

PPO（Proximal Policy Optimization）是 RLHF 早期最重要的算法之一。它从 policy gradient 出发，引入两个关键机制。

第一，重要性采样和旧 policy。纯 on-policy 方法要求每次更新都用当前 policy 新生成样本，代价很高，因为 rollout 很慢。PPO 允许先用旧 policy 采样一批数据，再对同一批 rollout 做多次梯度更新。

第二，clipping。PPO 不希望新 policy 相比旧 policy 变化过大，因此使用概率比值 ratio，并把它裁剪在 1-ε 到 1+ε 之间，例如 0.8 到 1.2。这样即使某个样本 reward 很高，模型也不会无限制地把概率推高，从而提升训练稳定性。

PPO 通常还需要 value model 来估计 advantage，例如使用 GAE（Generalized Advantage Estimation）。这能降低梯度方差，但代价是工程复杂：要维护 policy model、reward model、value model，有时还要处理不同 tokenizer、KL shaping、value loss、policy loss、clip norm、rollout 与训练 worker 的同步等。实际 PPO 有大量实现细节，稍有差异就可能影响结果。

在大语言模型上，value model 尤其昂贵：它往往和 policy 一样大，显存和计算成本接近翻倍。因此人们希望找到一种保留 PPO 稳定性、但去掉 value model 的方法。

## 5. GRPO：用组内基线替代 value model

GRPO（Group Relative Policy Optimization）可以看作 PPO 的简化变体，也是 DeepSeek Math / R1 系列中的关键算法。它保留 policy gradient、KL regularization、ratio clipping 等思想，但去掉了 value function 和复杂的 GAE。

GRPO 的核心做法是：对同一个问题 q，一次采样 G 个回答，形成一个 group。每个回答都有 reward。然后用组内 reward 的均值和标准差构造 advantage：

A_i = (r_i - mean(r_1, ..., r_G)) / std(r_1, ..., r_G)

也就是说，不再问“这个回答的绝对 reward 多高”，而是问“它比同一问题下的其他回答好多少”。这很自然：不同题目难度不同，简单题平均 reward 高，难题平均 reward 低；组内均值可以作为题目难度的 baseline。这样就不需要额外训练 value model。

如果只对每批 rollout 做一步在线更新，GRPO 甚至可以非常接近普通 policy gradient：高于组内平均的回答被上调，低于平均的回答被下调。实现上只需：生成多个回答、计算 reward、按组归一化、加 KL penalty、做梯度更新。

但 GRPO 也有微妙问题。标准差归一化并不是严格 policy gradient 推导中允许的普通 baseline。它会放大 reward 方差很小的组：例如所有回答都错或都对的题目。这可能把训练重点放在“太难”或“太容易”的问题上，而不是最有学习信号的中等难度问题。

另一个问题是长度归一化。如果把序列 reward 除以输出长度，那么答错时模型可能通过生成更长内容来稀释负 reward；答对时则倾向更短。这会诱导模型在不确定时输出很长的 chain-of-thought，看起来像“思考更久”，但可能只是目标函数偏差。后续 Dr. GRPO 等分析认为，去掉某些长度归一化能在保持 reward 的同时减少无界变长。

## 6. 为什么可验证奖励推动 reasoning models

以 DeepSeek R1 为例，训练流程展示了一个非常简单但有效的范式：在数学、代码等可验证任务上，用 outcome reward（最终答案对不对）进行 RL。R1-Zero 几乎直接从基础模型出发做 RL，reward 主要包括 accuracy reward 和 format reward。format reward 要求模型把推理放在特定 think tags 中，虽然看似只是格式约束，但实践中对稳定训练很重要。

R1 的重要结论是：不一定需要复杂的 MCTS search 或 PRM（Process Reward Model）。PRM 能给推理中间步骤打分，理论上反馈更丰富，但很难构建可靠的过程监督器。R1 发现，简单的 outcome-based reward 加 GRPO 就能得到强推理能力。

实际可发布模型通常不会只做 RL。更常见流程是：先做少量 long chain-of-thought SFT，让模型学会可读的推理格式；再做 verifiable reward RL，提高数学/代码正确率；最后再做通用 instruction tuning 和 RLHF，恢复聊天、写作、安全等通用能力。这说明 SFT 和 RL 是互补的：SFT 提供初始行为模式，RL 则让模型针对真实目标继续优化。

Kimi K1.5 和 Qwen3 也体现了类似思路。Kimi 强调数据筛选和长度控制：用 best-of-N 过滤太容易的问题，构造课程学习，并在训练后期加入 length reward，避免推理链过长导致推理成本失控。Qwen3 则加入 thinking mode fusion：同一模型支持 think 与 no-think 模式，并可通过 token budget 控制测试时思考长度，实现 inference-time scaling。

## 7. 训练稳定性与工程注意事项

LLM RL 的难点不只在算法，还在系统。rollout 需要自回归生成，远比普通 teacher-forcing 训练慢；训练 worker 更新权重后，还要同步到 inference worker；长 chain-of-thought 会造成 batch 长度不均，降低 GPU 利用率。许多系统会把训练和推理分成不同 worker，并用 vLLM 等推理引擎生成样本。

稳定训练通常依赖以下技巧：

1. KL regularization：限制新 policy 不要偏离 reference policy 太远，避免语言质量崩坏。
2. Clipping 或显式正则：控制单次 policy update 幅度。
3. 合理 baseline：降低 policy gradient 方差，例如 PPO 的 value function 或 GRPO 的组内均值。
4. Reward shaping：把格式、语言一致性、长度等辅助目标以加权 reward 形式加入，但权重需要经验调参。
5. 数据难度控制：太容易没有学习信号，太难全错也没有信号；best-of-N 过滤和 curriculum 能改善训练效率。
6. 长度控制：长推理可能提升性能，也可能只是被目标函数诱导；需要在正确率与推理成本之间权衡。

## 8. SFT、RL 与 RLHF 的分工

把这一讲放回整个 alignment pipeline，可以看到三类训练各有角色。SFT 负责给模型示范“应该怎样回答”，例如遵循指令、写出长推理链、使用固定格式、避免明显有害输出。它的优点是稳定、便宜、容易调试；缺点是只能模仿数据中已有的行为，不能直接鼓励模型探索比示范更好的解法。

RL 负责把模型从“会模仿”推向“会优化目标”。在数学和代码中，模型可以尝试许多不同解法，只要最终答案或测试通过，就获得正 reward。这种探索能发现 SFT 数据没有覆盖的行为模式，也能把正确答案概率持续推高。RLHF 则把优化目标从可验证任务扩展到人类偏好，例如有用性、礼貌性、安全性和风格一致性；但由于偏好 reward 噪声更大，所以更需要 KL、早停、评测集和人工检查来防止过优化。

因此，一个实用的顺序通常是：先用 SFT 建立可控的初始 policy；再在高质量、可验证、难度合适的数据上做 RL，提升推理和解题能力；最后用偏好优化或 RLHF 修正通用聊天体验。若顺序反过来，直接从很弱或格式混乱的模型开始 RL，reward 可能太稀疏，训练会不稳定；若只做 SFT 不做 RL，模型又可能停留在“看起来像会推理”，而不是在真实可验证目标上取得最高正确率。

评估时也要区分不同目标：数学榜单提升不代表通用助手更好，通用偏好提升也不代表推理更强。可靠的训练流程需要同时监控任务正确率、回答长度、KL 距离、格式违规率、拒答率、人工偏好和安全指标。只有这些曲线一起合理，才能说明 RL 真正在改善模型，而不是仅仅利用了某个评测或 reward 的漏洞。

## 9. 小结

本讲的主线是：RLHF 证明了 RL 可以用于语言模型对齐，但人类偏好 reward 噪声大、易过优化；可验证奖励提供了更清晰、更可扩展的训练信号。PPO 是经典且强大的 RLHF 算法，但 value model 和大量实现细节使它成本高、难调。GRPO 用同题多回答的组内相对 reward 替代 value model，大幅简化训练，因此成为 reasoning model 后训练的重要工具。

从 R1、Kimi K1.5、Qwen3 的经验看，成功 recipe 往往包含：少量高质量 long CoT SFT、可验证任务上的 RL、KL/长度/格式等稳定化约束、再加通用 RLHF 或指令微调。最终目标不是让模型“无限思考”，而是在可控成本下，把 policy 推向更高正确率、更好对齐和更稳定的行为。


---


# Stanford CS336 第 17 讲：Alignment - RL 2 中文教程

本讲继续上一讲的 RLHF 与 RL for Verifiable Rewards（RLVR）主题，但重心不在引入全新概念，而是深入拆解语言模型上策略梯度、PPO/GRPO 类算法、奖励设计和工程实现中的关键细节。核心问题是：当模型已经具备一定能力后，如何用“可评分的结果”继续优化它，而不是仅仅模仿人类标注数据。

## 1. 语言模型里的强化学习设定

在经典强化学习中，我们需要定义状态、动作、奖励和转移动态。对语言模型来说，这些概念有非常具体的对应关系：

- 状态：prompt 加上目前已经生成的 response 前缀。
- 动作：生成下一个 token。
- 轨迹/episode/rollout：从 prompt 开始，模型连续生成一段完整回答。
- 奖励：回答整体有多好。

本讲主要讨论 outcome reward，也就是只在完整回答生成完之后给一个奖励。例如数学题中，模型先写推理过程，最后输出“答案是 3 miles”，奖励函数负责抽取最终答案并与标准答案比较。若匹配则给 1，否则给 0。

这和一般 RL 有一个重要区别：语言模型的转移动态非常简单，就是把新 token 拼接到已有上下文后面。因此，语言模型可以天然做 test-time compute：采样多个候选答案、搜索、重排、验证。机器人控制里很难拥有这种可精确模拟的世界动态。

但困难也随之转移。机器人常常难在“到达某个物理状态”；语言模型几乎可以写出任意 token 序列，难点变成：这些 token 是否真的对应正确推理、正确答案和可靠行为。

## 2. 从 SFT 到策略梯度

语言模型 RL 的目标是最大化期望奖励：

\[
J(\pi)=\mathbb{E}_{s\sim p(s), a\sim \pi(\cdot|s)}[R(s,a)]
\]

其中 \(s\) 是 prompt，\(a\) 可以暂时看作完整 response。策略梯度的基本技巧是：

\[
\nabla J(\pi)=\mathbb{E}[R(s,a)\nabla \log \pi(a|s)]
\]

直觉上，它很像 SFT：SFT 是人类给定一个好答案，然后最大化该答案概率；策略梯度是模型自己采样答案，再根据奖励给这个答案加权。若奖励为 1，就增加该回答概率；若奖励为 0，朴素形式下几乎不更新。

这解释了为什么 RLVR 对初始模型能力有要求。若任务太难，模型几乎永远采不到正确答案，所有奖励都是 0，梯度也接近 0，训练无法启动。实际系统通常需要：

- 一个足够强的 base/SFT 模型；
- 足够多采样以增加碰到正奖励的概率；
- 设计更平滑的奖励或部分奖励；
- 使用 baseline、advantage、归一化等降低方差。

## 3. Baseline、Advantage 与方差降低

策略梯度的主要问题是高方差。奖励绝对值高，并不意味着该动作在当前 prompt 下是好选择。比如容易题的错误答案可能得 9 分，难题的较好答案只得 2 分；如果直接按奖励大小更新，模型可能错误地偏向“容易 prompt 下的次优动作”。

解决方法是引入 baseline：

\[
\mathbb{E}[(R(s,a)-B(s))\nabla\log\pi(a|s)]
\]

只要 \(B(s)\) 不依赖动作 \(a\)，它不会改变期望梯度的方向，因为减掉的是与策略无关的常数项。但它可以显著降低方差。

一个常见选择是令 baseline 近似为当前状态下的期望奖励：

\[
B(s)\approx \mathbb{E}_{a\sim\pi}[R(s,a)]
\]

此时 \(R(s,a)-B(s)\) 就是 advantage：该回答比同一 prompt 下的平均回答好多少。好于平均值的回答被增强，差于平均值的回答被压低。这也回答了“错误奖励为什么不设成 -1”的问题：中心化后，低于组内平均的样本自然会得到负 advantage。

## 4. GRPO：利用同一 prompt 的多回答作为组

PPO 通常使用 value function/critic 来估计 baseline。GRPO（Group Relative Policy Optimization）的思想更适合语言模型：对同一个 prompt，一次采样多个 response，组成一个 group，用组内奖励均值作为 baseline。

流程大致是：

1. 从一批 prompt 出发；
2. 每个 prompt 采样多个候选回答；
3. 对每个回答计算 reward；
4. 在同一 prompt 的 group 内计算均值和标准差；
5. 用 \((r_i-\bar r)/\sigma\) 作为更新信号；
6. 更新当前策略，使高于组内平均的回答概率上升，低于组内平均的回答概率下降。

这就是“relative”的含义：不是问某个回答绝对分数高不高，而是问它是否比同一 prompt 下的其他回答更好。语言模型天然适合这种结构，因为同一个 prompt 可以并行生成多个候选；在传统机器人 RL 中，每条轨迹往往状态差异很大，没有这么自然的组结构。

标准化还有一个好处：奖励尺度变化不再那么敏感。如果所有奖励都乘以 100，归一化后的 advantage 基本不变。不过如果一个 group 内所有回答奖励完全相同，中心化后 delta 全为 0，模型不会从这组样本中更新。这符合直觉：组内没有相对优劣信号。

## 5. PPO/GRPO 中的 ratio、clipping 与 KL

策略优化中常见一个重要量：

\[
\rho=\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}
\]

它表示当前策略相对采样时旧策略，对同一个回答概率改变了多少。PPO/GRPO 会把它与 advantage 相乘，并进行 clipping：

\[
\text{clip}(\rho, 1-\epsilon, 1+\epsilon)
\]

clipping 的作用是限制单次更新幅度，避免模型因为少数高奖励样本而突然偏离太远。实现时必须注意：\(\pi_{old}\) 应该是常数，不能让梯度穿过旧策略。工程上通常用 `no_grad`，或在 rollout 时直接保存旧策略对这些 response 的 log probability。

除了 old policy，训练中还可能有 reference model，用于 KL 正则：

\[
\text{reward objective} - \beta \mathrm{KL}(\pi_\theta || \pi_{ref})
\]

reference model 通常是初始 SFT 模型或较慢更新的模型。KL 惩罚防止当前模型为了追求奖励而偏离原始语言能力太远，例如输出格式崩坏、过度投机或丧失通用性。实践中可能同时存在三类模型/量：当前训练策略、用于 ratio 的 old policy、用于 KL 的 reference policy。

## 6. 奖励设计、reward hacking 与可验证奖励

RLVR 的魅力在于奖励可以自动计算，不必每个样本都问人。例如数学题答案匹配、代码单元测试通过、排序结果正确、定理证明检查器通过等。这类奖励具有确定性、可扩展、低成本的优点。

但奖励设计非常危险。讲座中的排序例子展示了这一点：如果只奖励“输出 token 是否来自输入”和“相邻 token 是否有序”，模型可能找到漏洞，例如重复输出某些 token、利用局部有序性骗取高分，却没有真正完成排序。这就是 reward hacking：模型优化了指标，但没有优化我们真正关心的目标。

奖励越密集，训练信号越强，但也越可能引入错误捷径；奖励越稀疏，越贴近真实目标，但优化更困难。工程上常见折中包括：

- 用精确最终奖励保证目标正确；
- 用部分奖励帮助探索，但不断检查是否可被利用；
- 对输出格式做严格解析和校验；
- 使用隐藏测试集或多样化环境防止过拟合奖励；
- 对高奖励样本人工抽查，寻找投机模式。

## 7. Reasoning models 与 RLVR 的关系

推理模型的关键不是简单“学会写更长 chain-of-thought”，而是在可验证任务上通过大量采样和强化学习，让模型更频繁地产生能导向正确答案的推理轨迹。只要奖励能可靠判断最终结果，模型就有机会发现比人类示范更有效的策略。

这也是 RL 相比 SFT 的潜力所在：SFT 只能模仿已有答案，RL 可以在可测量目标上超越示范。但前提是“可测量”本身足够可信。数学、代码、形式化证明、游戏和工具调用环境更适合 RLVR；开放式写作、价值判断和真实用户满意度则更接近 RLHF，需要奖励模型、人类偏好或 LLM-as-judge，但也更容易受到偏差和奖励攻击影响。

## 8. 评估与工程风险

讲座最后强调，RL 训练比预训练和 SFT 工程复杂得多。预训练主要是固定数据集上的 next-token loss；RL 则需要不断生成新数据、评分、更新，再生成新数据。loss 本身也不再像监督学习那样可直接解释，因为训练分布随策略变化而变化。真正需要监控的是 reward、通过率、格式错误率、KL、样本多样性和外部评测表现。

评估时尤其要区分“训练奖励上升”和“真实能力提升”。如果奖励函数有漏洞，训练曲线会很好看，但模型可能只是学会了输出某种模板、利用解析器缺陷、重复高分 token，或在测试环境中触发意外行为。因此应至少准备三类评估：第一，训练同分布验证集，用来快速发现过拟合和训练崩溃；第二，隐藏或更难的 out-of-distribution 评测，用来检查是否学到通用策略；第三，人工审查样本，用来发现自动指标难以捕捉的错误，例如推理胡编、格式投机、解释与答案不一致。

还要注意，RL 中的“好样本”并不一定始终好。模型早期采样出的某个高奖励回答可能只是偶然正确，如果对它做太多梯度步，策略会快速收缩到狭窄模式，探索能力下降。采样温度、每个 prompt 的候选数、每批数据复用多少步、clip 范围、KL 系数和参考模型更新频率，都会影响探索与稳定性的平衡。实践中常常需要同时看平均奖励、最佳样本奖励、响应长度、重复率、熵、拒答率和 KL 曲线，而不能只看单一指标。

工程上还要处理：

- 推理成本：每个 prompt 要采样多个 response；
- 奖励计算成本：可能要运行测试、调用环境或执行 agent；
- 多模型管理：current policy、old policy、reference model、critic/reward model；
- 分布式同步：rollout worker 与 trainer 之间要传模型、样本和 logprob；
- 内存开销：reference model 可能使显存翻倍；
- stale policy：样本由旧参数生成，但用新参数训练，必须控制偏移。

一个典型的 RLVR 系统会把 rollout、reward、训练和评估拆成多个服务。rollout worker 负责用当前或稍旧的模型生成候选；reward worker 负责解析答案、运行测试或调用环境；trainer 负责根据保存的 logprob、reward 和 advantage 更新模型；evaluator 定期在固定 benchmark 上评测。任何一个环节出错都会污染训练：解析器 bug 会制造错误奖励，环境非确定性会放大奖励噪声，worker 使用过旧模型会让 ratio 失真，分布式日志不完整会让问题难以复现。

因此，RLVR 不只是算法问题，更是系统问题。一个可工作的训练系统必须同时保证采样、奖励、优化、监控和安全评估都可靠。越强的优化器越会放大奖励定义中的瑕疵，所以在扩大训练规模之前，应该先用小模型、小数据和可解释样例反复验证奖励函数。

## 9. 小结

本讲的主线可以概括为：语言模型 RL 把完整回答当作动作序列，用可验证或可学习的奖励评价结果，再通过策略梯度提升高奖励回答的概率。朴素策略梯度像“带奖励权重的 SFT”，但高方差、稀疏奖励和 credit assignment 会让训练困难。baseline 和 advantage 通过相对比较降低方差；GRPO 利用同一 prompt 的多采样自然构造组内 baseline；clipping 和 KL 正则控制更新幅度，避免策略崩坏。

RLVR 是训练 reasoning models 的重要路径，因为它允许模型从可验证任务中自我改进。但它的成功依赖于奖励函数是否真实、不可 hack、可泛化，以及整个训练工程是否稳定。最终原则是：如果能可靠测量，就可以优化；如果测量有漏洞，优化会把漏洞放大。换句话说，推理模型的能力提升来自“生成—验证—更新”的闭环，而不是单纯把推理文本写得更长。真正重要的是验证器能否区分有效推理和看似合理的废话。


---
