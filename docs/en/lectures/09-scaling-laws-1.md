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
