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
