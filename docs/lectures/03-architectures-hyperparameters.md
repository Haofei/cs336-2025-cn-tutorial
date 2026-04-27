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
