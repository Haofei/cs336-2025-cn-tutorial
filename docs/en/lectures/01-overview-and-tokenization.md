# CS336 2025 Lecture 1 Tutorial: Course Overview and Tokenization

> This is an English tutorial adaptation of the Chinese CS336 2025 study guide.

## Learning goals

After this lecture, you should be able to:

1. Explain that CS336, “Language Models from Scratch,” is not about merely calling APIs, but about understanding and implementing the language-model building pipeline.
2. Describe the central question of the course: given a compute budget and a data budget, how do we train the best possible model?
3. Recognize the full workflow: tokenizer, Transformer architecture, training, systems optimization, scaling laws, data, evaluation, and alignment.
4. Understand tokenization: converting Unicode strings into integer sequences that a model can process.
5. Explain the basic idea, training procedure, and encode/decode logic of BPE (Byte Pair Encoding).

## Prerequisites

Recommended background:

- Basic Python and PyTorch.
- Basic machine-learning concepts: loss, optimizer, batch size, overfitting.
- Some familiarity with Transformer and attention; the course rebuilds these from the bottom up.
- You do not need to be a GPU expert, but you should be willing to reason about performance from an engineering perspective.

## Lecture map

This lecture has two main parts:

1. Course overview: why build language models from scratch, what modules the course covers, and how engineering and research perspectives fit together.
2. Tokenization basics: why tokenizers are needed, what goes wrong with character-, byte-, and word-level schemes, and how BPE provides a practical compromise.

The course storyline is:

```text
raw data → cleaning/filtering → tokenizer → integer sequences → Transformer → training → evaluation → systems optimization → alignment/fine-tuning → usable model
```

## 1. Why build language models from scratch?

The lecture begins with a trend: researchers are increasingly far from the underlying technology. In earlier NLP work, researchers often implemented and trained models themselves. Later, many projects downloaded models such as BERT and fine-tuned them. Today, many applications only prompt a closed model through an API.

This abstraction is useful, but it leaks. A language-model API may look like “string in, string out,” but without understanding data, model architecture, systems, and training, it is hard to do foundational research. CS336’s motto is:

```text
To understand it, you have to build it.
```

The course is also realistic. Frontier models require enormous capital, GPU clusters, and unpublished engineering details. A class cannot have every student train a GPT-4-scale model. Instead, the course trains small models while emphasizing what small-scale experiments can and cannot teach us.

The course separates three kinds of knowledge:

- mechanics: how Transformer is implemented and how GPU parallelism works. This can be taught concretely.
- mindset: always think in terms of efficiency and scaling; make hardware do useful work.
- intuitions: which data and architecture choices work at scale. These can only be partially learned from small experiments.

## 2. Core perspective: efficiency, not blind scale

The instructor warns against misreading the bitter lesson as “only scale matters and algorithms do not.” A better summary is:

```text
Algorithms at scale matter.
```

Model quality is roughly the result of resources multiplied by efficiency. The more expensive the resources, the more important efficiency becomes. If one training run costs a large amount of money, you cannot waste attempts the way you might in a local toy experiment. Algorithmic efficiency, hardware utilization, data quality, and architecture all affect the final result.

The course repeatedly asks:

```text
Given a compute budget and a data budget, what is the best model we can train?
```

This is the engineering core: do not only ask “can the model be larger?” Ask whether each FLOP, GPU, and token is being used effectively.

## 3. Brief background of language models

Language models are not new. Shannon used language models to estimate the entropy of English. In traditional NLP, language models were components in machine translation, speech recognition, and other systems. Deep learning then accumulated several key ingredients:

- neural language models
- seq2seq models
- Adam optimizer
- attention mechanism
- Transformer
- model parallelism
- foundation models such as ELMo, BERT, and T5

After GPT-2 and GPT-3, scaling laws and engineered training became central. Model openness also split into layers:

- closed models: accessible only through APIs.
- open-weight models: weights are available, but data and training details may be incomplete.
- open-source models: weights, data, and implementation are made as open as possible, though papers still cannot replace building the system yourself.

## 4. The five course modules

### 4.1 Basics

The goal is to implement a minimal but complete language-model training pipeline:

- tokenizer: maps between strings and integer sequences.
- model architecture: mostly Transformer.
- training: loss, optimizer, learning-rate schedule, and training loop.

Assignments implement a BPE tokenizer, Transformer, cross-entropy loss, AdamW optimizer, and training loop. PyTorch is allowed, but prebuilt Transformer implementations are not the point.

### 4.2 Systems

Training is not only formulas; it is hardware. GPU compute is on-chip, while GPU memory is often off-chip, so data movement can dominate. The course discusses:

- kernels: for example, how matrix multiplication uses tiling and fusion to reduce data movement.
- Triton: a tool for writing high-performance GPU kernels.
- parallelism: data parallelism, tensor/model parallelism, and related strategies.
- inference: the process of generating tokens with a trained model.

Inference has two phases:

- prefill: process the prompt; all input tokens are known and can be parallelized, similar to training.
- decode: autoregressively generate one token at a time; this often underuses GPUs and becomes memory-bound.

The course also mentions speculative decoding: a cheaper small model proposes candidate tokens, then the large model verifies them in parallel to accelerate inference.

### 4.3 Scaling laws

Central question: given a FLOPs budget, how should we balance model parameters and training tokens? A larger model can see fewer tokens; a smaller model can see more. Where is the optimum?

The lecture discusses Chinchilla optimal ideas: fit scaling relationships on small experiments, then predict the best parameter count and loss for large runs. The value is that cheap experiments can guide expensive training decisions.

A caution: rules of thumb about the ratio between model parameters and training tokens have assumptions and usually ignore inference cost. They should not be applied mechanically.

### 4.4 Data and evaluation

Model capabilities are largely determined by data: multilingual data creates multilingual ability; code data creates coding ability. Common sources include Web/Common Crawl, Wikipedia, GitHub, StackExchange, books, and papers.

But “feed the internet to the model” is misleading. Raw web data contains HTML, PDFs, code repositories, spam, boilerplate, duplicates, and legal/safety issues. It needs:

- extraction: convert HTML/PDF and other formats into text.
- filtering: remove low-quality, harmful, or irrelevant content.
- deduplication: delete repeated data so training budget is not wasted.
- legal considerations: discuss what data can be used for training.

Evaluation includes:

- perplexity: how well the model predicts the next token.
- standardized benchmarks such as MMLU.
- instruction-following evaluation.
- evaluation of agentic systems that include language models.

### 4.5 Alignment

A pretrained base model mainly predicts the next token. It has raw capability, but may not follow instructions. Alignment makes a model more useful, safer, and better suited for interaction.

Alignment goals include:

- instruction following
- style control, such as length, bullet formatting, and tone
- safety, including refusing harmful requests

Common stages include:

- SFT, supervised fine-tuning: supervised learning on user/assistant prompt-response pairs.
- learning from feedback: improve the model using preference data or a verifier.
- PPO, DPO, GRPO: reinforcement-learning or preference-optimization methods. DPO is designed for preference data; GRPO is a simplified PPO-like method used by DeepSeek-style training.

## 5. Tokenization: why do we need a tokenizer?

Language models process numerical tensors, while raw text is a Unicode string. Tokenization converts a string into an integer sequence and, ideally, can decode it back into the original string.

A tokenizer has two directions:

```text
encode: string → list[int]
decode: list[int] → string
```

Vocabulary size is the number of possible token IDs. A larger vocabulary often lets one token represent longer text fragments, but it also makes the input/output layers larger. A smaller vocabulary creates longer sequences, increasing attention cost.

An important detail: modern tokenizers are usually reversible and encode spaces. For example, “hello” and “ hello” may be different tokens. This differs from traditional whitespace word splitting.

## 6. Simple tokenization schemes and their problems

### 6.1 Character-based tokenization

Map every Unicode character to its code point. English letters and emoji all have corresponding integers.

Problems:

- Unicode has a very large code-point range.
- Many characters are extremely rare but still occupy vocabulary space.
- Compression is poor.

### 6.2 Byte-based tokenization

Encode the string as UTF-8 bytes. Each byte ranges from 0 to 255, so the vocabulary is tiny and any text can be represented.

Advantages: simple, elegant, and no unknown-character problem.

Problem: sequences are too long. Each token represents only one byte, so the compression ratio is low. Since standard attention is quadratic in sequence length, byte-level sequences are inefficient for today’s Transformer architectures.

### 6.3 Word-based tokenization

Split text into words or fragments using spaces, regular expressions, or pre-tokenization rules, then assign each word an integer.

Advantage: common words can be represented by one token, making sequences shorter.

Problems: the vocabulary may become huge, and there will always be unseen words, spellings, names, code fragments, and other rare strings. Handling them with an UNK token loses information and complicates evaluation.

## 7. BPE: Byte Pair Encoding

BPE is an old compression algorithm later adopted for neural machine translation and then language models such as GPT-2. Its key idea is: do not manually decide what counts as a word; learn tokens from corpus statistics.

Intuition:

- Frequent consecutive fragments should be merged into one token to improve compression.
- Rare fragments can remain split; they need not consume vocabulary entries.
- Starting from bytes guarantees that every string can be represented.

BPE training:

```text
Input: training corpus, target vocabulary size or number of merges
Initialize: convert text into byte sequences; initial vocabulary is 0..255
Repeat:
  1. Count every adjacent token pair in the current sequences
  2. Find the most frequent pair, for example (116, 104)
  3. Assign a new token id to that pair, for example 256
  4. Replace all occurrences of that pair in the training sequences with the new token
Output: merge rules and vocabulary
```

Encoding new text:

```text
1. Convert the string into bytes
2. Apply learned merge rules in training order
3. Return the integer token sequence
```

Decoding:

```text
1. Map each token id back to its byte sequence
2. Concatenate the bytes
3. Decode the bytes as UTF-8
```

GPT-2-style tokenizers also use pre-tokenization: a regular expression splits text into pieces, and BPE runs inside each piece. This is a practical compromise for efficiency and behavior control.

## 8. Common misconceptions

1. “Training from scratch is the first step for every problem.”
   No. If prompting or fine-tuning solves the problem, use that first. Training from scratch is for learning foundations or for cases that truly need a new base model.

2. “Small-model conclusions always transfer to large models.”
   Not always. FLOP ratios between attention and MLP, emergent behavior, and stability can change with scale.

3. “Tokenization is just a minor preprocessing detail.”
   No. Tokenizers directly affect sequence length, training efficiency, vocabulary size, reversibility, and multilingual/code performance.

4. “Byte-level tokenization is clean, so it must be best.”
   It is elegant, but with current Transformer architectures it usually creates sequences that are too long.

5. “Internet data can be trained on directly.”
   No. Raw Common Crawl-style data contains spam, duplicates, HTML/PDF structure, and legal/safety issues. It must be carefully processed.

## 9. Practice exercises

1. Open a tokenizer visualization tool. Try English, Japanese or Chinese, numbers, code, and emoji. Observe token boundaries.
2. Implement a minimal byte tokenizer with `encode(str) -> list[int]` and `decode(list[int]) -> str`.
3. On a tiny corpus, manually perform three BPE merges. Record the highest-frequency pair and new token id each time.
4. Compare token counts for the same text under character-, byte-, and BPE-level tokenization. Think about how sequence length affects attention cost.
5. Randomly sample web text and judge what is high quality, what should be filtered, and what should be deduplicated.

## 10. Summary

This lecture establishes the CS336 frame: a language model is not an isolated Transformer, but an end-to-end engineering pipeline. The course builds the tokenizer, model, training loop, systems optimizations, data pipeline, evaluation, and alignment methods from the bottom up, always asking how to get the best model under limited compute and data.

Tokenization is the entrance to the pipeline. Character-, byte-, and word-level schemes each have serious drawbacks. BPE starts from bytes and repeatedly merges frequent adjacent tokens, giving a practical balance between universal representability and compression. Tokenizer-free architectures may become mature in the future, but BPE and its variants remain foundational in current frontier-model practice.

## Further reading and next lecture

- Andrej Karpathy’s videos on tokenization and building models from scratch.
- The original Transformer paper: “Attention Is All You Need.”
- GPT-2 tokenizer and byte-level BPE implementations.
- Papers on Chinchilla scaling laws.

Next lecture moves into PyTorch details and resource accounting: writing code that runs is not enough; we also need to track FLOPs, memory, and data movement to understand where compute resources go.
