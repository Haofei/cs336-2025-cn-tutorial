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
