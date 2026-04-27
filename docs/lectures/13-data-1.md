# Stanford CS336 Lecture 13 教程：预训练数据（Data 1）

本讲从一个核心观点出发：在现代语言模型中，数据往往比模型结构更能决定最终能力。Transformer 架构、优化器、并行训练等技术已经相对公开，而顶级模型论文通常只模糊描述训练数据，例如“来自多种数据源、覆盖到某一年”。这种保密既有商业竞争原因，也有版权和法律风险。对实践者来说，真正困难的问题不是“有了数据怎样训练”，而是“什么数据值得训练、如何把原始互联网变成可训练语料”。

## 1. 训练阶段与数据角色

大模型训练通常可以粗略分为三段：

- **Pretraining（预训练）**：使用海量、相对原始的文本，主要来自 Web、代码、书籍、论文、百科等。目标是让模型学习语言、知识和通用模式。
- **Mid-training（中期训练）**：在预训练之后，用更小但质量更高、目标更明确的数据强化能力，例如数学、代码、长上下文、多语言等。
- **Post-training（后训练）**：包括 instruction tuning、chat data、RLHF/RLAIF 等，让模型更像助手，能遵循指令、对话并满足安全要求。

术语上，**base model** 通常指完成预训练/中期训练后的模型；**instruct/chat model** 则是经过后训练、适合交互的模型。现实中三者边界并不清晰：例如 Stack Exchange 的问答数据既可进入预训练，也天然像指令数据；现代数据管线也会在预训练阶段引入由模型筛选或改写的数据。

## 2. 为什么“互联网数据”不是一个简单概念

常见说法是“大模型训练在互联网数据上”，但这过于粗糙。真正的数据来源通常经历三层转换：

1. **Live service（在线服务）**：如 Wikipedia、GitHub、Reddit、Stack Overflow、新闻站点。
2. **Raw dump / crawl（原始快照或爬取）**：如 Common Crawl、Wikipedia dump、GitHub Archive。
3. **Trainable dataset（可训练数据集）**：经过文本抽取、语言识别、清洗、过滤、去重、采样和混合之后的 tokens。

因此，当有人说“我们训练在 GitHub / Common Crawl / Reddit 上”，必须追问：使用哪个快照？如何抽取文本？如何处理许可证？是否去重？过滤规则是什么？保留了哪些字段和元数据？这些决定会显著影响模型能力。

## 3. 早期数据：Books 与 Wikipedia

BERT 使用的主要数据是 **BooksCorpus** 与 **Wikipedia**。BooksCorpus 来自 Smashwords 上免费的自出版书籍，后来因服务条款问题下线。它说明了书籍数据的重要性：书籍具有长文结构、叙事连贯性和长距离依赖，适合训练模型理解较长上下文。

Wikipedia 则是长期被视为“高质量文本”的代表。它有明确编辑规范：强调可验证性、引用来源、非原创研究、较少个人观点，并通过 notability（关注度）筛选主题。但这也意味着 Wikipedia 不覆盖所有有价值内容：个人经验、菜谱、论坛讨论、小众知识、口语表达都可能缺失。

Wikipedia 还引出一个安全问题：**data poisoning（数据投毒）**。如果攻击者能在数据快照生成前短暂插入恶意内容，即使之后被回滚，内容仍可能进入训练集。更广泛地说，互联网上的训练数据由许多具有不同动机的人共同塑造，模型行为可能被这些数据影响，而训练方很难完全审计。

## 4. WebText：用链接信号筛选网页

GPT-2 的 WebText 数据集展示了一种重要思路：不是随机抓取网页，而是利用人类社区的链接和投票信号。OpenAI 收集 Reddit 中 karma 超过一定阈值的帖子所链接的网页，得到约 800 万页面、40GB 文本。直觉是：被用户分享并获得赞同的链接，平均质量高于普通网页。

WebText 未公开，后来社区做了 OpenWebText 复现。这类方法的关键是 **link-based filtering（基于链接的过滤）**：用高质量社区、百科引用或人工 curated 页面指向的外链作为质量信号。后来的 LLaMA 也使用过类似思路：训练分类器判断网页是否像 Wikipedia 引用过的页面。

## 5. Common Crawl：最大但很脏的公共 Web 来源

**Common Crawl** 是学术和开源社区最常用的大规模网页来源。它从 2007 年开始定期爬取网页，每次包含数十亿页面。爬虫从大量 seed URLs 出发，维护 frontier 队列，类似对 Web 做广度优先搜索，同时需要处理 robots.txt、服务器负载、重复 URL、动态页面等工程问题。

Common Crawl 提供两类重要格式：

- **WARC**：原始 HTTP 响应，通常包含 HTML，也可能包含其他资源。
- **WET**：从 HTML 转出的纯文本，是有损转换。

HTML-to-text 转换看似低级，却对训练质量影响很大。使用 Common Crawl 自带 WET、Trafilatura、jusText 等工具会得到不同文本，进而影响模型评测。现代数据工程常从 WARC 重新抽取正文，而不是直接依赖 WET。

Common Crawl 不是“整个互联网”。它覆盖稀疏、偏向文本、遵守或至少考虑 robots.txt，并不保证包含所有页面；同时它也包含大量垃圾、广告、模板、重复、低质和冒犯性内容。因此 Common Crawl 更像原材料，而不是可直接训练的数据集。

## 6. 清洗、过滤与去重

从原始网页到训练 tokens，常见步骤包括：

### 语言识别（Language Identification）
用 fastText 或其他分类器判断文档语言，只保留目标语言，或按多语言配比采样。早期许多研究聚焦英语，但 Common Crawl 本身包含多语言数据。

### 规则过滤（Rule-based Filtering）
C4、Gopher/ MassiveText、RefinedWeb、FineWeb 等使用大量手写规则，例如：保留以标点结尾的行、移除句子过少的页面、过滤脏词、要求一定比例单词含字母、移除 boilerplate、过滤疑似代码或模板。规则方法透明、便宜、可解释，但容易留下结构良好的垃圾文本，也可能误伤方言、少数群体文本或非标准写法。

### 模型过滤（Model-based Filtering）
CCNet 使用 Wikipedia 训练 n-gram 模型，保留“像 Wikipedia”的文档。GPT-3 使用质量分类器，把 WebText、Wikipedia、books 作为正例，从 Common Crawl 中找相似内容。DCLM 更进一步，用 OpenHermes、ELI5 等 instruction-like 数据作为正例，用 fastText 分类器从 240T tokens 的池子中筛到约 3.8T tokens。

模型过滤能显著提升 benchmark，但风险是把“质量”缩窄为正例分布：如果正例偏百科、偏英文、偏主流写作，模型会降低多样性。近年的趋势是重新接受甚至强化模型参与数据筛选，因为收益太明显。

### 去重（Deduplication）
Web 上重复极多：镜像站、转载、模板、动态 URL、代码 fork、文档副本都会造成重复。去重分为精确去重和 **fuzzy deduplication（模糊去重）**。去重能减少训练浪费，降低模型记忆特定文本的概率，也避免某些来源被过度加权。

### 有害内容与隐私过滤
很多管线会加入 toxicity classifier、安全搜索、PII anonymization（个人信息匿名化）等步骤。但这些过滤本身也不完美：过强会损失真实世界分布，过弱则会带来安全、隐私和法律问题。

## 7. 典型预训练数据集谱系

- **C4（Colossal Clean Crawled Corpus）**：Google/T5 使用的 Common Crawl 清洗版，主要依靠规则过滤，只保留英文自然语言文本。
- **The Pile**：EleutherAI 社区构建的 22 个高质量数据源混合，包括 Common Crawl、OpenWebText、Stack Exchange、Wikipedia、arXiv、PubMed、GitHub、Books3 等，体现“人工挑选领域”的路线。
- **MassiveText / Gopher**：DeepMind 的数据混合，包含 MassiveWeb、C4、books、news、GitHub、Wikipedia，并使用规则和安全过滤。
- **LLaMA 数据**：Common Crawl + C4 + GitHub + Wikipedia + Project Gutenberg + Books3 + arXiv + Stack Exchange，总计约 1.2T tokens。未公开，但 RedPajama 做了复现。
- **RefinedWeb / FineWeb**：主张只要 Web 过滤得足够好，就可以得到强数据。FineWeb 是 Hugging Face 对大规模 Common Crawl 的轻过滤版本，可作为进一步筛选的基础。
- **DCLM Baseline**：把 Common Crawl 全量池构造成竞赛式数据基准，用强质量分类器 aggressive filtering，成为近期开源模型常用数据来源。
- **Nemotron-CC**：NVIDIA 在 DCLM 思路上扩展，用大模型打分“educational value（教育价值）”、蒸馏到较快模型，并组合多个过滤器，还尝试用 LLM 改写低质数据或把高质文档转成任务形式。

这些数据集体现了两个张力：一是质量与规模的取舍，过滤越狠质量越高但 tokens 越少；二是质量与多样性的取舍，越像高质量正例，越可能丢掉长尾知识、口语和非主流文本。

## 8. 代码、问答、书籍与论文的特殊价值

不同来源提供不同能力：

- **GitHub / The Stack**：主要训练代码能力，也可能提升结构化推理。处理时要识别许可证、去除重复、过滤生成文件、区分代码与文档、考虑 issues 和 commit history 是否使用。
- **Stack Exchange / Stack Overflow**：天然是 QA 格式，有问题、回答、评论、投票等元数据，适合筛选高质量解释，也模糊了预训练与指令训练边界。
- **Project Gutenberg / PG19**：公共领域书籍，版权清晰，适合长上下文训练；但语言风格偏旧。
- **arXiv / PubMed / Semantic Scholar**：学术论文提供知识密度、数学和技术表达，但格式抽取、公式、引用和版权都需要处理。
- **Reddit / ELI5**：更接近用户问题和通俗解释，可作为质量分类器正例或 instruction-like 语料。

## 9. 版权与数据可用性

绝大多数互联网上的原创表达默认受版权保护，即使网页没有写 copyright 标记。使用方式大致有两条路：取得 license（许可证），或主张 **fair use（合理使用）**。合理使用会考虑用途是否转换性、作品性质、使用比例、对原市场的影响等。对大模型训练来说，复制训练数据本身就涉及版权；训练是否足够 transformative、模型是否记忆和复现原文、是否替代原作者市场，都是争议焦点。

此外，即便内容采用 Creative Commons 或可能属于 fair use，平台 Terms of Service（服务条款）也可能禁止自动下载。例如公开视频不等于可以随意爬取。大型公司可通过商业授权获得 Reddit、Stack Exchange、Shutterstock 等数据；开源和学术团队则更依赖公开 dump、许可清晰数据和谨慎过滤。

## 10. Mid-training 与 Post-training 数据

中期训练和后训练更关注特定能力。长上下文扩展常在后期进行，因为从一开始用超长序列训练成本太高；数据上可使用书籍、长论文、长代码、合成长依赖任务等。

指令数据方面，早期有 Super-Natural Instructions、FLAN：把传统 NLP 任务统一成 instruction format。之后出现 Alpaca/self-instruct、Vicuna、OpenHermes、Evol-Instruct 等合成数据方法，即用强模型生成任务、回答或多轮对话。合成数据便宜、可扩展，但受生成模型许可证限制，也可能继承教师模型偏差。另一条路线是雇佣标注者写高质量指令数据，成本高但可控；不过还要防止标注者偷偷用商业模型生成答案。

## 11. 工程实践总结

构建预训练数据管线时，可以按如下流程思考：

1. 明确目标能力：通用知识、代码、数学、多语言、长上下文、对话风格。
2. 收集原始来源：Web crawl、dump、API、授权数据、公共领域数据。
3. 文本抽取：HTML/PDF/code/email 等格式转纯文本，保留必要元数据。
4. 基础清洗：语言识别、编码修复、去 boilerplate、长度过滤、格式过滤。
5. 质量筛选：规则、分类器、LLM 打分、链接信号、社区投票信号。
6. 安全与合规：版权、license、robots.txt、ToS、PII、toxicity。
7. 去重与采样：精确/模糊去重，避免重复来源支配训练。
8. 数据混合：按能力和质量设定 mixture weights，并通过小模型 ablation 验证。
9. 记录版本：保存快照时间、处理代码、过滤阈值、统计信息，保证可追溯。

实际落地时，不要只看最终 token 数。更有用的监控包括：每个来源的保留率、重复率、平均文档长度、语言分布、域名分布、困惑度或质量分数分布、被过滤样本示例，以及训练后在目标评测上的增益。数据管线应当像模型代码一样版本化，否则很难解释一次训练为什么变好或变坏。

本讲的核心结论是：数据不会从天上掉下来。可训练语料是大量工程、启发式规则、法律判断和实验迭代的结果。现代模型之间架构差异可能不大，数据来源、过滤策略、去重质量、合成数据和授权资源，才是决定模型差异的重要因素。