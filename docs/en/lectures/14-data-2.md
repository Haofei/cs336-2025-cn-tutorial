# Stanford CS336 Lecture 14: Data (Part 2) — From Raw Web Pages to Trainable Corpora

> Adaptation note: This English tutorial is adapted from the Chinese lecture notes. It preserves the original structure, technical terminology, formulas, and code-style notation, while presenting the material in clear tutorial English.

This lecture continues the discussion of “data engineering” in large-model pretraining. The previous lecture was more like a history of datasets: from early corpora to Common Crawl, C4, The Pile, LLaMA, Dolma, and so on. This lecture turns to actionable methods: when we have massive raw Web pages and a small amount of ideal data, how do we filter, mix, deduplicate, and turn these decisions into a training-data recipe?

The core problem can be summarized as follows: given a small, high-quality target set T and a huge but noisy raw set R, find the subset T' of R that “looks like T.” This is not only quality filtering; the same idea also applies to language identification, domain selection, toxicity filtering, mining math/code data, filtering synthetic data, and adjusting different domain mixtures during training.

## 1. The basic paradigm of data selection

Data selection is not simply “keep Web pages that look good.” A general pipeline usually contains three steps:

1. Define the target: what data do we want? It may be Wikipedia-style text, textbook-style code, mathematical proofs, English Web pages, low-toxicity discussions, or a task domain needed by a product.
2. Train or construct a scorer: use target data and raw data to estimate a score `score(x)`, indicating how much sample x resembles the target domain, how valuable it is, or how safe it is.
3. Select or resample: keep samples above a threshold, sample according to probabilities, or redistribute data using importance weights.

There are two practical constraints. First, the scorer must generalize: it is meaningless if it only retrieves T itself; we need it to discover new similar samples from R. Second, the scorer must be fast enough: Web-scale data is enormous, and if a giant model scores every item, filtering may cost nearly as much as, or even more than, pretraining itself.

## 2. Three common kinds of filters

### 2.1 n-gram language models: crude but cheap

The most traditional method is to train an n-gram language model, for example using KenLM with Kneser-Ney smoothing. It essentially counts occurrences of n-word sequences and estimates conditional probabilities. For example, given the context “the cat,” it estimates the probability that the next word is “in.” Because many n-grams have never appeared, smoothing backs off to shorter contexts.

The usage is straightforward: train an n-gram model on the target corpus, then compute perplexity for raw documents. Lower perplexity means the text is more similar to the target domain. CCNet used a similar method to sort paragraphs by perplexity and keep the better parts; later LLaMA data was also influenced by this kind of pipeline.

The advantages of this method are speed, simplicity, and scalability. Its disadvantages are also obvious: it mainly looks at local co-occurrence and cannot truly judge long-range logic or semantic quality. Shuffled paragraphs, template spam, and text with locally normal grammar but globally meaningless content may all fool it. Therefore n-gram filtering is better suited for removing obvious noise than for fine-grained quality assessment.

### 2.2 fastText / linear classifiers: a common industrial baseline

fastText is a lightweight text classifier. It hashes words or n-grams into fixed buckets and then performs linear classification through low-dimensional representations. Although the structure is simple, it is fast, parallelizable, and suitable for Web-scale filtering.

A typical training setup is to construct a binary classification task: positive examples come from high-quality or target-domain data, and negative examples come from raw data such as Common Crawl. The classifier outputs the probability that “this sample comes from the target domain.” GPT-3 used high-quality sources as positives and Common Crawl as negatives to train a quality classifier; LLaMA used Web pages cited by Wikipedia as positives; Dolma used fastText for language identification and toxicity filtering.

The key value of fastText is not that it is “smart enough,” but that it is “cheap enough to run over the whole Web.” If raw data must be compressed down to 1%, the filter processes 100 times the final training volume; in that setting, scoring cost per sample must be extremely low.

### 2.3 Importance resampling: from “classification” to “distribution matching”

A classifier answers “does this look like the target domain,” but training data also needs to preserve distributional diversity. Importance resampling provides a more principled view: suppose the target distribution is P and the raw distribution is Q. We can only sample from Q, but we want the final samples to look as if they came from P. Therefore each sample receives a weight:

w(x) = P(x) / Q(x)

Then we resample in proportion to w(x). The intuition is: if some kind of text is common in the target domain but rare in the raw data, raise its sampling probability; if the opposite is true, lower it.

In real settings, P is hard to estimate accurately because target data is small. In practice, people use hashed n-grams to estimate a rough distribution, then compute approximate weights. This does not always bring huge gains, but compared with pure binary classification, it emphasizes distribution matching in the domain mixture rather than simply “passing a threshold.”

## 3. Quality assessment, domain mixture, and language selection

There is no single definition of “good data.” Quality may mean grammatical fluency, high information density, high educational value, low toxicity, little templating, fit to a target task, or provenance from trustworthy sources. Therefore data filtering is often decomposed into multiple independent dimensions.

Language identification is the most basic example. If the goal is an English model, mixing in many other languages consumes the token budget and reduces English training intensity. But if the model is large enough, multilingual data may also bring positive transfer. Bloom is about 30% English and emphasizes multilingual capability; frontier models usually cover hundreds of languages. Whether to filter by language is fundamentally a training-data decision: target users, model capacity, compute budget, and evaluation metrics jointly determine the mixture.

Domain selection is equally important. OpenWebMath treats “mathematics” as a special language: first use rules to find candidates, then use KenLM and fastText classifiers trained on mathematical-proof data such as Proof-Pile to filter them, ultimately obtaining about 15 billion math tokens. The results show that high-density data targeted at the math domain can outperform much larger but unfocused data. This illustrates that domain mixture is not “the bigger the better”; it must match the target capability.

Quality assessment can also be assisted by strong models. Phi-1’s idea was to train a small model, but give it high-value “textbook-style” code data. Researchers first asked GPT-4 to judge whether Python code snippets had educational value for beginners, obtaining about 100,000 labeled examples, then used a cheaper classifier to scale up to large data. This is a common pattern: use an expensive model to create a small high-quality T, then distill it into a cheap filter that can process R.

## 4. Synthetic data: target data can be “generated”

When no ready-made target corpus exists, a strong language model can synthesize or select target data. For example, it can be asked to generate textbook-style code, mathematical reasoning, chemistry QA, or to label Web pages with an “educational value” score. In this case, T is no longer just some existing source; it is defined by the requirement and the prompt.

But synthetic data has risks: the distribution may be too narrow, style may be uniform, errors may be amplified, and it may be highly similar to existing data. Therefore synthetic data usually should not be added without limit. It should go through quality classification, deduplication, human spot checks, and downstream evaluation. A more robust approach is to use synthetic data to improve a specific capability while preserving the diversity of real data; use a strong model to label a small batch, then train a cheap filter to scale it up.

## 5. Deduplication: reducing waste and memorization

Filtering decides “which data is worth training on”; deduplication decides “how many times the same information is trained on.” The Web naturally contains massive duplication: mirror sites, license text, product templates, copy-pasted articles, and template pages with only a few words changed. C4 once contained an ordinary English sentence that appeared tens of thousands of times. It is not bad text, but training on it 60,000 times is meaningless.

Exact deduplication is simple: hash sentences, paragraphs, or documents, group samples with identical hashes, and keep only one. It has high precision and is easy to parallelize, but it cannot find near duplicates. A Bloom filter uses a bit array and multiple hash functions to save memory; it has no false negatives, but may have false positives, making it suitable for approximate set queries at huge scale.

Near-duplicate deduplication is usually based on Jaccard similarity. Split documents into shingles or n-gram sets; if the intersection-over-union of two sets exceeds a threshold, treat them as near duplicates. Direct pairwise comparison is O(N²), which is infeasible. The key property of MinHash is that the collision probability of two sets under MinHash equals their Jaccard similarity. Combined with LSH (locality sensitive hashing), multiple hashes are divided into several bands, so highly similar documents collide with high probability and dissimilar documents collide with low probability. This finds duplicate candidates in linear or near-linear time.

Deduplication requires care. Removing junk Web duplicates during pretraining is usually beneficial. But in mid-training or continued training, repeating high-quality data for multiple epochs may be exactly what we want. A more reasonable strategy may be not to keep only one copy, but to downweight duplicate counts, for example sampling according to log or square root counts, so that “important and common” content receives higher weight without being amplified linearly by its raw repetition count.

## 6. Curriculum, annealing, and training-data recipes

Data decisions do not only happen before training. Many modern training runs change the mixture over time: early on, they use large-scale, diverse, loosely filtered data to learn general language and world knowledge; later, they gradually anneal toward high-quality, target-domain, instruction-style, or reasoning data to improve final evaluations. This is similar to curriculum learning: start broad, then increase density and difficulty.

Common strategies include:

- Expand coverage early: mix Web pages, books, code, multilingual data, forums, and so on, avoiding early overfitting to a narrow domain.
- Raise target-domain proportion in the middle: if code, math, or a particular language matters, gradually increase that domain’s tokens.
- Quality annealing late: reduce low-quality Web data and increase textbooks, QA, reasoning, human-written data, or data selected by strong models.
- Use synthetic data in limited amounts: avoid style collapse while using it to fill gaps in scarce capabilities.
- Adjust through evaluation loops: every mixture change should be checked against downstream benchmarks, perplexity, human samples, and safety metrics.

Therefore domain mixture is an optimization problem, not a static table. The best proportions usually cannot be written down by intuition in one attempt. They require training small models, doing ablations, inspecting data samples, and iterating. A practical principle is to record the data recipe as part of the model: token counts from each source, filtering thresholds, deduplication granularity, repetition sampling multipliers, and the time periods during which each source enters training should all be traceable. Otherwise, when a model changes on some capability or safety metric, it is hard to know whether the cause was model scale, optimization hyperparameters, or the data recipe.

## 7. Practical checklist

When building a pretraining corpus, you can audit it with the following questions:

1. What is the target capability? General chat, code, mathematics, multilinguality, or a specialized domain?
2. Where does the target data T come from? Human sources, trusted sites, strong-model labels, synthetic generation, or rule-based preselection?
3. Is the filter cheap enough? It processes the massive raw R, not the final small corpus.
4. How is the quality threshold chosen? Too loose keeps noise; too strict loses diversity and low-resource groups.
5. Does the mixture match the token budget? Increasing one domain’s proportion means reducing training opportunities for other domains.
6. Has exact and near-duplicate deduplication been done? Has training-set leakage into evaluation sets been avoided?
7. Is repeated high-quality data needed? If so, is repetition linear or downweighted?
8. Are data decisions validated through small-scale training rather than only by filter scores?

## Summary

The main thread of this lecture is: data does not naturally fall into the training set. It is produced through a series of computable and scalable decisions that are nevertheless full of trade-offs. n-gram models, fastText, and importance resampling provide basic tools for finding target data within raw Web pages; language identification, quality filtering, toxicity filtering, and domain mining are all instances of the same framework; synthetic data and strong-model labeling mean that “target data” itself can be designed; deduplication reduces meaningless repetition, lowers memorization risk, and saves compute.

Real data capability comes from a loop: inspect data, write filters, train models, evaluate results, adjust the mixture, and repeat. For large-model training, data selection is often as important as model architecture and compute scale, and in specific capabilities it may determine final performance even more strongly.
