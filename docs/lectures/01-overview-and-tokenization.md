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
