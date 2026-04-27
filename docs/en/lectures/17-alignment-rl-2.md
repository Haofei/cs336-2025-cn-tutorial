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
