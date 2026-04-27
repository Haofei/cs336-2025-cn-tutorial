# Stanford CS336 2025 Tutorial: Language Modeling from Scratch

This repository is an unofficial, tutorial-style study guide for Stanford CS336 2025, **Language Modeling from Scratch**.

The original videos are excellent, but classroom transcripts can be repetitive, conversational, and jump between topics. This project reorganizes the material into more self-study-friendly notes: conceptual structure, engineering intuition, formulas, pitfalls, and practice suggestions.

> Source course: https://stanford-cs336.github.io/  
> YouTube playlist: https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_

## Languages

English is the default language for this repository.

- English: [README](docs/en/README.md), [complete tutorial](docs/en/CS336-2025-complete-tutorial-en.md), [study plan](docs/en/study-plan.md), [glossary](docs/en/glossary.md)
- 中文: [完整教程](docs/CS336-2025-complete-tutorial.md), [学习计划](docs/study-plan.md), [术语表](docs/glossary.md), [单讲目录](docs/lectures/)
- 日本語: [README](docs/ja/README.md), [完全チュートリアル](docs/ja/CS336-2025-complete-tutorial-ja.md), [学習計画](docs/ja/study-plan.md), [用語集](docs/ja/glossary.md)

## Start here

- [Complete English tutorial](docs/en/CS336-2025-complete-tutorial-en.md)
- [English study plan](docs/en/study-plan.md)
- [English glossary](docs/en/glossary.md)
- [English lecture notes](docs/en/lectures/)

## Who this is for

- Students who want a structured path through LLM pretraining, inference, evaluation, data, and alignment.
- Engineers who want to strengthen AI infrastructure fundamentals: GPUs, Triton kernels, distributed parallelism, and serving.
- Readers who know Python/PyTorch and want a clearer mental model of model-training cost, bottlenecks, and scaling decisions.
- Anyone who wants to scan the course quickly before deciding which videos to study in depth.

## Suggested learning path

1. **Foundations:** lectures 1-3 — tokenization, PyTorch resource accounting, Transformer architecture.
2. **Training systems:** lectures 4-8 — MoE, GPUs, Triton kernels, distributed parallelism.
3. **Training decisions and deployment:** lectures 9-14 — scaling laws, inference, evaluation, data.
4. **Alignment and RL:** lectures 15-17 — SFT, RLHF, PPO/GRPO, RLVR, reasoning models.

## Lecture index

| # | English tutorial | Source video | Duration |
|---|------------------|--------------|----------|
| 01 | [CS336 2025 Lecture 1 Tutorial: Course Overview and Tokenization](docs/en/lectures/01-overview-and-tokenization.md) | [YouTube](https://www.youtube.com/watch?v=SQ3fZ1sAqXI) | 1:18:55 |
| 02 | [Stanford CS336 2025 Lecture 2 Tutorial: PyTorch and Resource Accounting](docs/en/lectures/02-pytorch-resource-accounting.md) | [YouTube](https://www.youtube.com/watch?v=msHyYioAyNE) | 1:19:18 |
| 03 | [CS336 2025 Lecture 3 Tutorial: Transformer Architectures and Hyperparameters](docs/en/lectures/03-architectures-hyperparameters.md) | [YouTube](https://www.youtube.com/watch?v=ptFiH_bHnJw) | 1:26:59 |
| 04 | [Stanford CS336 Lecture 4 Tutorial: Mixture of Experts (MoE)](docs/en/lectures/04-mixture-of-experts.md) | [YouTube](https://www.youtube.com/watch?v=LPv1KfUXLCo) | 1:22:00 |
| 05 | [Stanford CS336 Lecture 5 Tutorial: GPUs](docs/en/lectures/05-gpus.md) | [YouTube](https://www.youtube.com/watch?v=6OBtO9niT00) | 1:14:17 |
| 06 | [Stanford CS336 2025 Lecture 6 Tutorial: Kernels, Triton, and LLM Operator Optimization](docs/en/lectures/06-kernels-triton.md) | [YouTube](https://www.youtube.com/watch?v=E8Mju53VB00) | 1:20:18 |
| 07 | [Stanford CS336 Lecture 7 Tutorial: Parallelism 1](docs/en/lectures/07-parallelism-1.md) | [YouTube](https://www.youtube.com/watch?v=l1RJcDjzK8M) | 1:24:39 |
| 08 | [Stanford CS336 Lecture 8 Tutorial: Parallel Training (Part 2)](docs/en/lectures/08-parallelism-2.md) | [YouTube](https://www.youtube.com/watch?v=LHpr5ytssLo) | 1:15:06 |
| 09 | [CS336 Lecture 9 Tutorial: Scaling Laws (Part 1)](docs/en/lectures/09-scaling-laws-1.md) | [YouTube](https://www.youtube.com/watch?v=6Q-ESEmDf4Q) | 1:05:15 |
| 10 | [Stanford CS336 Lecture 10 Tutorial: LLM Inference](docs/en/lectures/10-inference.md) | [YouTube](https://www.youtube.com/watch?v=fcgPYo3OtV0) | 1:22:48 |
| 11 | [CS336 Lecture 11 Tutorial: Scaling Laws (Part 2)](docs/en/lectures/11-scaling-laws-2.md) | [YouTube](https://www.youtube.com/watch?v=OSYuUqGBQxw) | 1:18:10 |
| 12 | [CS336 Lecture 12 Tutorial: LLM Evaluation](docs/en/lectures/12-evaluation.md) | [YouTube](https://www.youtube.com/watch?v=x-R5l2HsXqM) | 1:20:45 |
| 13 | [Stanford CS336 Lecture 13 Tutorial: Pretraining Data (Data 1)](docs/en/lectures/13-data-1.md) | [YouTube](https://www.youtube.com/watch?v=WePxmeXU1xg) | 1:19:02 |
| 14 | [Stanford CS336 Lecture 14: Data (Part 2) — From Raw Web Pages to Trainable Corpora](docs/en/lectures/14-data-2.md) | [YouTube](https://www.youtube.com/watch?v=9Cd0THLS1t0) | 1:19:08 |
| 15 | [CS336 Lecture 15 Tutorial: Alignment, SFT, and RLHF](docs/en/lectures/15-alignment-sft-rlhf.md) | [YouTube](https://www.youtube.com/watch?v=Dfu7vC9jo4w) | 1:14:47 |
| 16 | [CS336 Lecture 16 Tutorial: Alignment with Reinforcement Learning (Part 1)](docs/en/lectures/16-alignment-rl-1.md) | [YouTube](https://www.youtube.com/watch?v=46f2QTDB08Q) | 1:20:29 |
| 17 | [CS336 Lecture 17 Tutorial: Alignment with Reinforcement Learning (Part 2)](docs/en/lectures/17-alignment-rl-2.md) | [YouTube](https://www.youtube.com/watch?v=JdGFdViaOJk) | 1:16:04 |

## Repository structure

```text
.
├── README.md                         # English default entry point
├── LICENSE
├── NOTICE.md
├── CONTRIBUTING.md
├── tutorial-manifest.json
└── docs/
    ├── en/                           # English edition
    │   ├── README.md
    │   ├── CS336-2025-complete-tutorial-en.md
    │   ├── study-plan.md
    │   ├── glossary.md
    │   └── lectures/
    ├── ja/                           # Japanese edition
    │   ├── README.md
    │   ├── CS336-2025-complete-tutorial-ja.md
    │   ├── study-plan.md
    │   ├── glossary.md
    │   └── lectures/
    ├── CS336-2025-complete-tutorial.md  # Chinese complete tutorial
    ├── study-plan.md                    # Chinese study plan
    ├── glossary.md                      # Chinese glossary
    └── lectures/                        # Chinese lecture notes
```

## How this differs from watching the videos

This repository is not a replacement for the original course. It is a learning accelerator:

- Removes classroom repetition, conversational filler, and transcript noise.
- Reorders ideas into a more tutorial-like learning sequence.
- Keeps key technical terms, formulas, and system concepts explicit.
- Adds common pitfalls and practice-oriented explanations.
- Helps you build the conceptual map first, then return to selected video segments when needed.

## Disclaimer

- The original course, videos, slides, assignments, and other Stanford CS336 materials belong to the Stanford CS336 course staff and their respective rights holders.
- This repository is an unofficial educational adaptation based on publicly available course content.
- This repository is not affiliated with, sponsored by, or endorsed by Stanford University or the course instructors.
- If you are a rights holder and believe content should be adjusted or removed, please open an issue.

## Contributing

Contributions are welcome: fix terminology, improve explanations, add diagrams, add exercises, or cross-check details against the original videos.

See [CONTRIBUTING.md](CONTRIBUTING.md).
