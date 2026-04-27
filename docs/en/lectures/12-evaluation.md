# CS336 Lecture 12 Tutorial: LLM Evaluation

> Adaptation note: This English tutorial is adapted from the Chinese lecture notes. It preserves the original structure, technical terminology, formulas, and code-style notation, while presenting the material in clear tutorial English.

## 1. Why evaluation is not simple

On the surface, LLM evaluation looks like a script: choose a model, prepare prompts, call the model, collect outputs, compute metrics, and average the results. The real difficulty is deciding what question you want the number to answer.

Different roles care about very different evaluations:

- Users or companies: I need to choose among Claude, Gemini, GPT, and open-source models. Which is best for my business?
- Researchers: Does the model really have stronger general capability? Is AI progressing in a scientifically meaningful sense?
- Policymakers: What benefits and risks does the model bring? Is it safe enough?
- Model developers: Did a training, data, or alignment method actually improve the model?

Therefore there is no single “correct” evaluation. A leaderboard score is meaningful only after you understand its input distribution, calling protocol, scoring rules, and intended use. Evaluation also shapes model development: once a benchmark becomes a target, developers optimize for it. When a metric is over-optimized, it may lose its original meaning. This is Goodhart's law in LLM evaluation.

## 2. Evaluation framework: input, invocation, output, and interpretation

A reliable evaluation can be decomposed into four questions.

First, where do the inputs come from? Do the prompts cover real use cases? Do they include difficult examples, long-tail examples, and edge cases? If the task is multi-turn chat, later inputs depend on the model's earlier answers, so a static test set may not simulate real conversation. Red-team testing often also needs to adaptively generate attack prompts based on model behavior; otherwise rare failures are hard to find.

Second, how is the model called? Zero-shot, few-shot, chain-of-thought, tool use, RAG, and agent scaffolding can all significantly affect results. Early base models often needed few-shot examples to specify the format. Modern instruction-tuned models can usually follow zero-shot instructions such as “output only A/B/C/D.” Prompt order, formatting, and example choice all introduce variance.

Third, how is the output judged? Multiple-choice tasks can use accuracy, code tasks can use pass@1 or pass@k, and open-ended generation may need human preferences or LLM-as-a-judge. Cost must also be considered: a model with a higher score but much higher price, latency, or inference token count may not be the better system. Different errors also have different costs; medical, legal, and safety-critical settings cannot be evaluated only by average accuracy.

Fourth, how should the score be interpreted? Is 91% good or bad? Is it enough for deployment? Does it show that the model learned a capability, or merely that it has seen similar questions? Is the evaluated object a base model, a chat model, a complete agent system, or a training method? These questions must be made explicit in advance.

## 3. Perplexity: still an important foundational metric

Perplexity measures how much probability a model assigns to a dataset. A language model is fundamentally a probability distribution over token sequences; lower perplexity means the model can better predict the tokens in that dataset. Traditional language-modeling research trained and tested on fixed datasets such as Penn Treebank, WikiText, and One Billion Word, with the goal of reducing test perplexity.

After GPT-2, the paradigm changed: models are pretrained on large-scale web text and then directly transferred to many downstream tasks and perplexity benchmarks. Evaluation then becomes closer to out-of-distribution generalization: the model was not specifically trained on Penn Treebank, but may perform well because its training corpus is broad enough.

Advantages of perplexity:

- Smoothness: it uses the log probability of every token, giving more fine-grained information than binary right/wrong accuracy.
- Suitability for scaling laws: as model size, data, and compute change, loss curves are easier to fit.
- Comprehensive coverage: it cares about every token in the dataset, not only the final answer.
- Less vulnerable to answer-format gaming, as long as train/test separation is reliable.

But perplexity also has limitations. First, it is not always strongly correlated with downstream task performance; over short periods or on specific tasks, the relationship can be messy. Second, if a leaderboard requires models to provide probabilities, it must trust that the provider's logits or probability API is correctly normalized; implementation bugs can otherwise create falsely low perplexity. Finally, the perplexity-maximization view says that “matching the true distribution solves everything,” but this may not be the most efficient path, because many tokens are not important for practical tasks.

Some tasks are close to perplexity, such as LAMBADA's missing-word prediction and HellaSwag's multiple-choice continuation task: the model compares the likelihood of candidate continuations. But these tasks can also saturate and may suffer from approximate contamination from original web sources.

## 4. Multiple-choice benchmarks: MMLU, MMLU-Pro, GPQA, HLE

MMLU is one of the classic LLM knowledge benchmarks, containing multiple-choice questions across 57 subjects. It appeared after GPT-3, when making a base model answer questions across many subjects via few-shot prompting was still a novel setup. The name MMLU contains “language understanding,” but it is more like a knowledge exam: many questions test specific disciplinary facts rather than pure language understanding.

MMLU scores must be interpreted together with the training and evaluation setup. If a base model is not specifically optimized for MMLU but scores highly on multi-subject multiple-choice questions, this suggests strong general knowledge and transfer ability. But if developers collect similar questions, tune prompts, use chain-of-thought, and ensemble results specifically for MMLU, then a high score does not necessarily represent the same degree of general capability.

MMLU-Pro tries to address MMLU saturation: it removes noisy and easy questions, increases the number of options from 4 to 10, and more often uses chain-of-thought. This lowers frontier-model accuracy and gives the benchmark discriminative power again.

GPQA emphasizes expert-level difficult questions. The questions are written and verified by PhDs or domain experts, with the goal of being “Google-proof”: non-experts should have difficulty answering them even with search. Early GPT-4 did not perform very well, but newer models have improved substantially. This shows that “hard for humans to search” does not mean “hard for LLMs forever.” During evaluation, one must also confirm whether the model is allowed to use the internet, because black-box APIs may secretly call search tools.

Humanity's Last Exam (HLE) further collects extremely difficult, multimodal, multiple-choice or short-answer questions. It uses prizes and attribution to attract contributors and filters out questions that frontier models find too easy. Its advantage is difficulty; its disadvantage is clear distributional bias. People willing to write questions are often familiar with LLMs and may deliberately design “model-hard” questions, so HLE does not represent ordinary user needs.

## 5. Open-ended and instruction-following evaluation

The core ability of modern chat models is not merely taking exams, but completing open-ended tasks from natural-language instructions. Open-ended outputs have no unique ground truth, making evaluation harder.

Chatbot Arena asks users to submit real prompts, shows answers from two anonymous models, asks the user to choose the better answer, and computes Elo rankings from pairwise preferences. Its advantages are that it is dynamic, close to real usage, and can include new models. Its disadvantages are that the user distribution is uncontrolled, prompts may be for entertainment or testing, and the more important the leaderboard becomes, the easier it is to optimize for or manipulate. Recent debates around Arena also show that evaluation protocol, submission access, model versions, and data transparency all matter.

IFEval specifically evaluates “constraint following” in instruction following: for example, the output must be shorter than a certain number of words, must include or exclude certain words, or must use a specific format. Its advantage is that scripts can automatically verify the constraints. Its disadvantage is that it checks only formal constraints, not semantic quality. A 10-word story may satisfy the length requirement without being good.

AlpacaEval uses LLM-as-a-judge to compare a model answer against a reference model answer and compute win rate. It is automatic, fast, and reproducible, but the judge model has biases. For example, early small models could fool the GPT-4 judge by producing longer answers; length correction was added later. Datasets such as WildBench sample from real human-model conversations and ask judges to evaluate with a checklist, often also reporting correlation with Chatbot Arena.

## 6. Agent benchmarks: evaluating the model or the system?

Many tasks require tool use and multi-step iteration. In this case, the evaluation target is no longer just an LM, but a “model + agent scaffolding” system.

SWE-bench gives an agent a GitHub issue and codebase, asks it to modify the code and submit a patch, and finally checks whether unit tests pass. Cybench puts the agent in a CTF cybersecurity environment where it must execute commands, explore servers, and obtain flags. MLE-bench simulates Kaggle: the agent must read the task, write training code, tune hyperparameters, and submit results. These benchmarks are closer to real workflows, but their scores are strongly affected by tools, context management, retry strategy, time budget, and cost.

Therefore, when reporting agent scores, one must state: Is internet access allowed? How many steps can it run? Are human hints allowed? Are there hidden tests? How many dollars and how much time did it spend? If a system obtains a high score through massive sampling and expensive inference, it is not the same capability as a single low-cost answer.

## 7. Contamination: training-set contamination and evaluation validity

Modern models are trained on large-scale internet data, and developers usually do not release the full corpus, so train/test overlap is almost impossible to rule out completely. Contamination may be exact repetition, near duplication, paraphrase, translation, leaked solutions, or leaked answers. Simple n-gram deduplication can find some problems, but cannot detect cross-lingual or semantically equivalent versions.

There are three broad responses:

- Data decontamination: check document, paragraph, or n-gram overlap between the test set and training corpus, conservatively removing suspicious samples.
- Behavioral detection: infer whether a model has seen data by observing abnormal preferences for option order, question order, or rare text.
- Community norms: papers and model cards should report decontamination methods, whether test-set leakage was checked, confidence intervals, and standard errors.

Contamination affects not only multiple-choice questions but also HellaSwag, WikiHow-derived tasks, math problems, and code tasks. Benchmark data itself may also contain annotation errors. When model scores are very high, a substantial fraction of remaining errors may come from question noise rather than insufficient model capability.

## 8. Human evaluation, real use cases, and safety evaluation

Human evaluation is often used for open-ended tasks, but it must specify who the reviewers are, whether they are experts, what the scoring rubric is, whether evaluation is blind, and whether answer length and style are controlled. Preferences of ordinary internet users, judgments of domain experts, and product user satisfaction are not the same thing.

Real-use evaluation is harder than exams and also more important. Users may be “asking questions” because they do not know the answer and need help, or “testing the model” because they already know the answer. Standardized exams mostly belong to the latter category, while business value often comes from the former. Work from Anthropic and others clusters real conversations to analyze what people actually use models for, such as coding, writing, learning, and office work. In medicine, MedHELM asks clinicians to propose real tasks such as clinical-note summarization, treatment planning, and patient communication. But real data often involves privacy, creating tension between public reproducibility and realism.

Safety evaluation also cannot look only at “refusal rate.” HarmBench, AIRBench, and related benchmarks test whether a model complies with harmful requests or build risk categories from laws and company policies. But a model that refuses every question is of course “safe” and also useless. Therefore safety must be evaluated together with capability. It is also important to distinguish **capability** from **propensity**: whether the model knows how to do something dangerous is capability; whether it is willing to output it is propensity. Closed-source APIs focus more on propensity and jailbreak defenses; open-weight models also require attention to capability, because safety layers may be removed by fine-tuning.

## 9. Checklist for reliable evaluation practice

1. First state the purpose of evaluation: model selection, research, product monitoring, safety review, or training feedback.
2. Specify the evaluation target: base model, chat model, agent system, or training method.
3. Fix and disclose the calling protocol: prompt, few-shot examples, temperature, max tokens, tool permissions, and retry count.
4. Report quality, cost, latency, and variance together, not only average accuracy.
5. For multiple-choice tasks, check option-order bias; for open-ended tasks, check length bias and judge bias.
6. Perform contamination checks and report the method; interpret high-risk benchmarks conservatively.
7. Inspect sampled predictions; do not look only at leaderboard numbers.
8. For real deployment scenarios, build private, updated eval sets close to the user distribution.
9. Pair safety evaluation with usefulness evaluation to avoid artificially high “refuse everything” scores.
10. Remember that benchmarks are tools, not truth. Once a benchmark becomes a target, it will be optimized, saturated, and possibly distorted.

In summary, the core of LLM evaluation is not “running a score,” but translating a real-world question into an executable, interpretable, and reproducible measurement process. Good evaluation needs both the comparability of standardized benchmarks and the representativeness of real use cases. It must care about capability, but also about cost, safety, contamination, and data quality. Only by understanding the rules behind the number can we really know where a model is good, where it is weak, and whether it fits our goal.
