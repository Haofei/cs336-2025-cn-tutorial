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
