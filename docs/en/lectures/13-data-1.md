# Stanford CS336 Lecture 13 Tutorial: Pretraining Data (Data 1)

> Adaptation note: This English tutorial is adapted from the Chinese lecture notes. It preserves the original structure, technical terminology, formulas, and code-style notation, while presenting the material in clear tutorial English.

This lecture starts from a central point: in modern language models, data often determines final capability more than model architecture does. Transformer architectures, optimizers, and parallel training techniques are now relatively public, while papers on top models usually describe training data only vaguely, for example as “from multiple data sources, up to a certain year.” This secrecy has commercial reasons, but also copyright and legal-risk reasons. For practitioners, the truly difficult question is not “how do we train once we have data,” but “which data is worth training on, and how do we turn the raw Internet into trainable corpora?”

## 1. Training stages and the role of data

Large-model training can roughly be divided into three stages:

- **Pretraining**: uses massive, relatively raw text, mainly from the Web, code, books, papers, encyclopedias, and similar sources. The goal is for the model to learn language, knowledge, and general patterns.
- **Mid-training**: after pretraining, uses smaller but higher-quality and more targeted data to strengthen capabilities such as mathematics, code, long context, and multilinguality.
- **Post-training**: includes instruction tuning, chat data, RLHF/RLAIF, and related methods, making the model more assistant-like: able to follow instructions, hold conversations, and satisfy safety requirements.

Terminologically, a **base model** usually refers to a model after pretraining and/or mid-training; an **instruct/chat model** is a model that has undergone post-training and is suitable for interaction. In practice, the boundaries among the three stages are blurry. For example, Stack Exchange question-answer data can be used during pretraining, but it also naturally resembles instruction data; modern data pipelines may also introduce model-filtered or model-rewritten data already during pretraining.

## 2. Why “Internet data” is not a simple concept

A common statement is that “large models are trained on Internet data,” but this is far too crude. Real data sources usually pass through three layers of transformation:

1. **Live service**: Wikipedia, GitHub, Reddit, Stack Overflow, news sites, and so on.
2. **Raw dump / crawl**: Common Crawl, Wikipedia dumps, GitHub Archive, and similar snapshots.
3. **Trainable dataset**: tokens after text extraction, language identification, cleaning, filtering, deduplication, sampling, and mixing.

Therefore, when someone says “we trained on GitHub / Common Crawl / Reddit,” you must ask: which snapshot was used? How was text extracted? How were licenses handled? Was deduplication performed? What filtering rules were used? Which fields and metadata were retained? These decisions significantly affect model capability.

## 3. Early data: Books and Wikipedia

The main data used by BERT was **BooksCorpus** and **Wikipedia**. BooksCorpus came from free self-published books on Smashwords and was later taken down because of Terms-of-Service issues. It illustrates the importance of book data: books have long-form structure, narrative coherence, and long-range dependencies, making them suitable for training models to understand longer contexts.

Wikipedia has long been regarded as a representative source of “high-quality text.” It has explicit editing norms: verifiability, cited sources, no original research, relatively few personal opinions, and topic selection through notability. But this also means Wikipedia does not cover all valuable content: personal experience, recipes, forum discussions, niche knowledge, and colloquial expression may all be missing.

Wikipedia also raises a safety issue: **data poisoning**. If an attacker can briefly insert malicious content before a data snapshot is produced, the content may enter the training set even if it is later reverted. More broadly, training data on the Internet is jointly shaped by many people with different motivations. Model behavior may be affected by that data, while the training organization may find it hard to fully audit.

## 4. WebText: using link signals to select Web pages

GPT-2’s WebText dataset demonstrated an important idea: do not crawl Web pages randomly; instead, use link and voting signals from human communities. OpenAI collected Web pages linked from Reddit posts whose karma exceeded a threshold, obtaining about 8 million pages and 40 GB of text. The intuition is that links shared by users and endorsed by the community are, on average, higher quality than ordinary Web pages.

WebText was not released, and the community later produced OpenWebText as a reproduction. The key idea in this family of methods is **link-based filtering**: use outgoing links from high-quality communities, encyclopedia citations, or manually curated pages as quality signals. Later LLaMA data used a similar idea: train a classifier to determine whether a Web page resembles pages cited by Wikipedia.

## 5. Common Crawl: the largest but very noisy public Web source

**Common Crawl** is the large-scale Web source most commonly used by academic and open-source communities. It has crawled the Web periodically since 2007, with each crawl containing billions of pages. The crawler starts from many seed URLs and maintains a frontier queue, roughly performing breadth-first search over the Web, while also handling engineering issues such as robots.txt, server load, duplicate URLs, and dynamic pages.

Common Crawl provides two important formats:

- **WARC**: raw HTTP responses, usually containing HTML but sometimes containing other resources.
- **WET**: plain text converted from HTML; this is a lossy conversion.

HTML-to-text conversion may look low-level, but it greatly affects training quality. Using Common Crawl’s own WET files, Trafilatura, jusText, or other tools produces different text, which in turn affects model evaluation. Modern data engineering often re-extracts article text from WARC instead of relying directly on WET.

Common Crawl is not “the whole Internet.” Its coverage is sparse, biased toward text, follows or at least considers robots.txt, and does not guarantee inclusion of all pages. At the same time, it contains large amounts of spam, advertising, templates, duplicates, low-quality text, and offensive content. Common Crawl is therefore more like raw material than a dataset that can be directly used for training.

## 6. Cleaning, filtering, and deduplication

Common steps from raw Web pages to training tokens include:

### Language Identification

Use fastText or another classifier to identify document language, keeping only target languages or sampling according to a multilingual mixture. Many early studies focused on English, but Common Crawl itself contains multilingual data.

### Rule-based Filtering

C4, Gopher/MassiveText, RefinedWeb, FineWeb, and others use many hand-written rules, such as keeping lines that end with punctuation, removing pages with too few sentences, filtering profanity, requiring a certain fraction of words to contain letters, removing boilerplate, and filtering likely code or templates. Rule-based methods are transparent, cheap, and interpretable, but they can leave behind well-structured junk text and may mistakenly remove dialects, minority-group text, or non-standard writing.

### Model-based Filtering

CCNet trained an n-gram model on Wikipedia and kept documents that “look like Wikipedia.” GPT-3 trained a quality classifier using WebText, Wikipedia, and books as positive examples, then searched for similar content in Common Crawl. DCLM went further: it used instruction-like data such as OpenHermes and ELI5 as positives, and used a fastText classifier to filter a pool of 240T tokens down to about 3.8T tokens.

Model-based filtering can significantly improve benchmarks, but the risk is that it narrows “quality” to the positive-example distribution. If the positives are biased toward encyclopedic text, English, or mainstream writing, the model may lose diversity. A recent trend is to accept, and even strengthen, model involvement in data selection again because the gains are so large.

### Deduplication

The Web contains enormous duplication: mirror sites, reposts, templates, dynamic URLs, code forks, and documentation copies all create duplicates. Deduplication is divided into exact deduplication and **fuzzy deduplication**. Deduplication reduces wasted training, lowers the probability that the model memorizes specific text, and prevents some sources from being overweighted.

### Harmful-content and privacy filtering

Many pipelines add toxicity classifiers, safe search, PII anonymization, and similar steps. But these filters are themselves imperfect: if too strong, they damage the real-world distribution; if too weak, they introduce safety, privacy, and legal problems.

## 7. A genealogy of typical pretraining datasets

- **C4 (Colossal Clean Crawled Corpus)**: Google/T5’s cleaned version of Common Crawl, relying mainly on rule-based filtering and keeping only English natural-language text.
- **The Pile**: a mixture of 22 high-quality data sources built by the EleutherAI community, including Common Crawl, OpenWebText, Stack Exchange, Wikipedia, arXiv, PubMed, GitHub, Books3, and others. It represents the “manually selected domains” approach.
- **MassiveText / Gopher**: DeepMind’s data mixture, including MassiveWeb, C4, books, news, GitHub, and Wikipedia, with rule-based and safety filtering.
- **LLaMA data**: Common Crawl + C4 + GitHub + Wikipedia + Project Gutenberg + Books3 + arXiv + Stack Exchange, totaling about 1.2T tokens. It was not released, but RedPajama produced a reproduction.
- **RefinedWeb / FineWeb**: argues that if Web filtering is good enough, strong data can be obtained from the Web alone. FineWeb is Hugging Face’s lightly filtered large-scale Common Crawl dataset and can serve as a base for further selection.
- **DCLM Baseline**: turns the full Common Crawl pool into a competition-style data benchmark and uses a strong quality classifier for aggressive filtering. It has become a common data source for recent open-source models.
- **Nemotron-CC**: NVIDIA’s extension of the DCLM idea. It uses large models to score “educational value,” distills this into faster models, combines multiple filters, and also tries to use LLMs to rewrite low-quality data or convert high-quality documents into task form.

These datasets reveal two tensions. The first is the trade-off between quality and scale: stronger filtering gives higher quality but fewer tokens. The second is the trade-off between quality and diversity: the more data resembles high-quality positives, the more likely it is to lose long-tail knowledge, colloquial language, and non-mainstream text.

## 8. The special value of code, question answering, books, and papers

Different sources provide different capabilities:

- **GitHub / The Stack**: mainly trains coding ability and may also improve structured reasoning. Processing requires license detection, duplicate removal, filtering generated files, distinguishing code from documentation, and deciding whether to use issues and commit history.
- **Stack Exchange / Stack Overflow**: naturally has a QA format, with questions, answers, comments, votes, and other metadata. It is suitable for selecting high-quality explanations and also blurs the boundary between pretraining and instruction training.
- **Project Gutenberg / PG19**: public-domain books with clear copyright status, suitable for long-context training, but with an older language style.
- **arXiv / PubMed / Semantic Scholar**: academic papers provide dense knowledge, mathematics, and technical expression, but format extraction, formulas, citations, and copyright all need handling.
- **Reddit / ELI5**: closer to user questions and plain-language explanations; can be used as positive examples for quality classifiers or as instruction-like corpora.

## 9. Copyright and data availability

Most original expression on the Internet is protected by copyright by default, even if a Web page does not display a copyright notice. There are roughly two routes for use: obtain a license, or claim **fair use**. Fair use considers factors such as whether the use is transformative, the nature of the work, the proportion used, and the effect on the original market. For large-model training, copying the training data itself implicates copyright; whether training is sufficiently transformative, whether the model memorizes and reproduces original text, and whether it substitutes for the original author’s market are all disputed questions.

In addition, even if content uses Creative Commons or might fall under fair use, a platform’s Terms of Service may prohibit automated downloading. Publicly viewable videos, for example, are not automatically free to crawl. Large companies can obtain data from Reddit, Stack Exchange, Shutterstock, and similar sources through commercial licensing; open-source and academic teams rely more on public dumps, clearly licensed data, and careful filtering.

## 10. Mid-training and post-training data

Mid-training and post-training focus more on specific capabilities. Long-context extension is often done late because training with very long sequences from the start is too expensive; data may include books, long papers, long code, synthetic long-dependency tasks, and so on.

For instruction data, early examples include Super-Natural Instructions and FLAN, which unified traditional NLP tasks into an instruction format. Later came synthetic-data methods such as Alpaca/self-instruct, Vicuna, OpenHermes, and Evol-Instruct, where strong models generate tasks, answers, or multi-turn conversations. Synthetic data is cheap and scalable, but it is constrained by the generating model’s license and may inherit the teacher model’s biases. Another route is to hire annotators to write high-quality instruction data, which is expensive but controllable; even then, one must prevent annotators from secretly using commercial models to generate answers.

## 11. Engineering practice summary

When building a pretraining data pipeline, you can reason through the following process:

1. Define target capabilities: general knowledge, code, mathematics, multilinguality, long context, dialogue style.
2. Collect raw sources: Web crawls, dumps, APIs, licensed data, public-domain data.
3. Extract text: convert HTML/PDF/code/email and other formats into plain text while retaining necessary metadata.
4. Perform basic cleaning: language identification, encoding repair, boilerplate removal, length filtering, format filtering.
5. Select for quality: rules, classifiers, LLM scoring, link signals, community voting signals.
6. Handle safety and compliance: copyright, license, robots.txt, ToS, PII, toxicity.
7. Deduplicate and sample: exact/fuzzy deduplication, avoiding repeated sources dominating training.
8. Mix data: set mixture weights according to capability and quality, and validate through small-model ablations.
9. Record versions: save snapshot time, processing code, filtering thresholds, and statistics to ensure traceability.

In practice, do not look only at the final token count. More useful monitoring includes retention rate by source, duplicate rate, average document length, language distribution, domain distribution, perplexity or quality-score distributions, examples of filtered samples, and gains on target evaluations after training. The data pipeline should be versioned like model code; otherwise it is hard to explain why a training run improved or worsened.

The core conclusion of this lecture is: data does not fall from the sky. Trainable corpora are the result of extensive engineering, heuristics, legal judgment, and experimental iteration. Modern models may not differ much in architecture; data sources, filtering strategies, deduplication quality, synthetic data, and licensed resources are often the important factors that determine model differences.
