# CS336 2025 中文教程版：Language Modeling from Scratch

这是 Stanford CS336 “Language Modeling from Scratch” 2025 课程的中文教程化学习资料。

原始视频课很精彩，但课堂 transcript 往往有口语、跳跃、重复、临时问答。这个仓库把 17 讲 transcript 重新组织成更适合自学的中文教程：按概念脉络、工程直觉、关键公式、实践练习和常见误区来整理，而不是逐字翻译。

> Source course: https://stanford-cs336.github.io/  
> YouTube playlist: https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_

## Languages / 语言 / 言語

- 中文: [README](README.md), [完整教程](docs/CS336-2025-complete-tutorial.md)
- English: [README](docs/en/README.md), [Complete tutorial](docs/en/CS336-2025-complete-tutorial-en.md)
- 日本語: [README](docs/ja/README.md), [完全チュートリアル](docs/ja/CS336-2025-complete-tutorial-ja.md)

## 适合谁

- 想系统学习 LLM pretraining / inference / evaluation / data / alignment 的同学
- 想补 AI Infra 基础：GPU、Triton、parallelism、serving 的工程师
- 已经会 Python/PyTorch，但想理解大模型训练成本和系统瓶颈的人
- 想快速浏览课程主线，再决定哪些视频需要精看的学习者

## 如何使用

推荐顺序：

1. 先读这份 README，建立课程地图。
2. 如果想快速通读，打开：[完整合并版教程](docs/CS336-2025-complete-tutorial.md)。
3. 如果想精学，按 `docs/lectures/` 中单讲逐篇阅读。
4. 遇到不清楚的地方，回到对应 YouTube 视频片段精看。

## 学习路线

### 第一阶段：从零构建语言模型

- Lecture 1: Overview and Tokenization
- Lecture 2: PyTorch and Resource Accounting
- Lecture 3: Architectures and Hyperparameters

目标：理解从文本到 token、从 Transformer 到训练成本的基本闭环。

### 第二阶段：训练系统与高性能计算

- Lecture 4: Mixture of Experts
- Lecture 5: GPUs
- Lecture 6: Kernels and Triton
- Lecture 7-8: Parallelism

目标：理解为什么大模型训练是系统工程问题：显存、带宽、通信、并行策略和 kernel 实现都决定训练效率。

### 第三阶段：扩展规律、推理、评估与数据

- Lecture 9 & 11: Scaling Laws
- Lecture 10: Inference
- Lecture 12: Evaluation
- Lecture 13-14: Data

目标：理解如何决定模型规模、数据规模、训练预算；如何部署模型；如何评估模型；如何构建高质量训练数据。

### 第四阶段：对齐与强化学习

- Lecture 15: SFT / RLHF
- Lecture 16-17: Alignment RL

目标：理解 post-training、preference data、reward model、PPO/GRPO/RLVR、reasoning model 和 reward hacking。

## 章节目录

| # | 教程 | 原视频 | 时长 | 中文字符数 |
|---|------|--------|------|------------|
| 01 | [CS336 2025 第 1 讲教程：课程概览与 Tokenization（分词/标记化）](docs/lectures/01-overview-and-tokenization.md) | [YouTube](https://www.youtube.com/watch?v=SQ3fZ1sAqXI) | 1:18:55 | ~3401 |
| 02 | [Stanford CS336 2025 第 2 讲教程：PyTorch 与资源核算](docs/lectures/02-pytorch-resource-accounting.md) | [YouTube](https://www.youtube.com/watch?v=msHyYioAyNE) | 1:19:18 | ~3003 |
| 03 | [CS336 2025 第 3 讲中文教程：Transformer 架构与超参数](docs/lectures/03-architectures-hyperparameters.md) | [YouTube](https://www.youtube.com/watch?v=ptFiH_bHnJw) | 1:26:59 | ~3213 |
| 04 | [Stanford CS336 第 4 讲教程：Mixture of Experts（MoE）](docs/lectures/04-mixture-of-experts.md) | [YouTube](https://www.youtube.com/watch?v=LPv1KfUXLCo) | 1:22:00 | ~3026 |
| 05 | [Stanford CS336 Lecture 5：GPU 中文教程](docs/lectures/05-gpus.md) | [YouTube](https://www.youtube.com/watch?v=6OBtO9niT00) | 1:14:17 | ~3038 |
| 06 | [Stanford CS336 2025 第 6 讲教程：Kernel、Triton 与 LLM 算子优化](docs/lectures/06-kernels-triton.md) | [YouTube](https://www.youtube.com/watch?v=E8Mju53VB00) | 1:20:18 | ~3001 |
| 07 | [Stanford CS336 Lecture 7 中文教程：Parallelism 1](docs/lectures/07-parallelism-1.md) | [YouTube](https://www.youtube.com/watch?v=l1RJcDjzK8M) | 1:24:39 | ~3005 |
| 08 | [Stanford CS336 Lecture 8：并行训练（二）中文教程](docs/lectures/08-parallelism-2.md) | [YouTube](https://www.youtube.com/watch?v=LHpr5ytssLo) | 1:15:06 | ~3001 |
| 09 | [CS336 第 9 讲教程：Scaling Laws（一）](docs/lectures/09-scaling-laws-1.md) | [YouTube](https://www.youtube.com/watch?v=6Q-ESEmDf4Q) | 1:05:15 | ~3001 |
| 10 | [Stanford CS336 Lecture 10：LLM Inference 中文教程](docs/lectures/10-inference.md) | [YouTube](https://www.youtube.com/watch?v=fcgPYo3OtV0) | 1:22:48 | ~3058 |
| 11 | [CS336 第 11 讲教程：Scaling Laws（二）](docs/lectures/11-scaling-laws-2.md) | [YouTube](https://www.youtube.com/watch?v=OSYuUqGBQxw) | 1:18:10 | ~3024 |
| 12 | [CS336 Lecture 12 中文教程：LLM Evaluation](docs/lectures/12-evaluation.md) | [YouTube](https://www.youtube.com/watch?v=x-R5l2HsXqM) | 1:20:45 | ~3004 |
| 13 | [Stanford CS336 Lecture 13 教程：预训练数据（Data 1）](docs/lectures/13-data-1.md) | [YouTube](https://www.youtube.com/watch?v=WePxmeXU1xg) | 1:19:02 | ~3046 |
| 14 | [Stanford CS336 Lecture 14：数据（二）——从原始网页到可训练语料](docs/lectures/14-data-2.md) | [YouTube](https://www.youtube.com/watch?v=9Cd0THLS1t0) | 1:19:08 | ~3038 |
| 15 | [CS336 Lecture 15 中文教程：Alignment、SFT 与 RLHF](docs/lectures/15-alignment-sft-rlhf.md) | [YouTube](https://www.youtube.com/watch?v=Dfu7vC9jo4w) | 1:14:47 | ~3076 |
| 16 | [Stanford CS336 Lecture 16 中文教程：Alignment 中的强化学习（一）](docs/lectures/16-alignment-rl-1.md) | [YouTube](https://www.youtube.com/watch?v=46f2QTDB08Q) | 1:20:29 | ~3035 |
| 17 | [Stanford CS336 第 17 讲：Alignment - RL 2 中文教程](docs/lectures/17-alignment-rl-2.md) | [YouTube](https://www.youtube.com/watch?v=JdGFdViaOJk) | 1:16:04 | ~3010 |

## 和直接看视频相比有什么不同

这份教程不是视频替代品，而是学习加速器：

- 去掉课堂口语、重复、跳跃和临时问答
- 把概念按学习顺序重排
- 对关键英文术语保留原文并加中文解释
- 补充“常见误区”和“实践练习”
- 适合先建立知识骨架，再按需回看视频

如果目标是“了解课程全貌”，阅读教程会比看完整视频快很多；如果目标是“做出 assignment / 复现系统”，仍建议结合原视频、课程作业和代码实践。

## 仓库结构

```text
.
├── README.md
├── LICENSE
├── NOTICE.md
├── CONTRIBUTING.md
├── docs/
│   ├── CS336-2025-complete-tutorial.md
│   ├── study-plan.md
│   ├── glossary.md
│   └── lectures/
│       ├── 01-overview-and-tokenization.md
│       └── ...
└── tutorial-manifest.json
```

## 版权与声明

- 原课程、视频、讲义和作业版权归 Stanford CS336 课程团队及相关权利人所有。
- 本仓库是基于公开课程内容整理的中文学习笔记/教程，目的为非商业学习分享。
- 本仓库不是 Stanford 官方资料，也不代表 Stanford 或课程教师观点。
- 如果你是权利人并认为内容需要调整或移除，请开 issue 联系。

## 贡献

欢迎提交 PR：

- 修正术语翻译
- 增补公式推导
- 增加图示、代码练习、阅读材料
- 对照原视频补充遗漏点
- 改善中文表达和结构

详见 [CONTRIBUTING.md](CONTRIBUTING.md)。
