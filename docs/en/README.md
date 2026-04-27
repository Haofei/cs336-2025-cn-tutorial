# CS336 2025 English Tutorial Adaptation

This is the English edition of the community study guide for Stanford CS336 2025, **Language Modeling from Scratch**.

The original course videos are excellent, but classroom transcripts can be repetitive and jump between topics. This edition reorganizes the material into tutorial-style notes: concepts, engineering intuition, formulas, common pitfalls, and practice suggestions.

> Original course: https://stanford-cs336.github.io/  
> YouTube playlist: https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_

## Start here

- [Complete English tutorial](CS336-2025-complete-tutorial-en.md)
- [Study plan](study-plan.md)
- [Glossary](glossary.md)

## Lecture index

| # | Tutorial | Source video | Duration |
|---|----------|--------------|----------|
| 01 | [CS336 2025 Lecture 1 Tutorial: Course Overview and Tokenization](lectures/01-overview-and-tokenization.md) | [YouTube](https://www.youtube.com/watch?v=SQ3fZ1sAqXI) | 1:18:55 |
| 02 | [Stanford CS336 2025 Lecture 2 Tutorial: PyTorch and Resource Accounting](lectures/02-pytorch-resource-accounting.md) | [YouTube](https://www.youtube.com/watch?v=msHyYioAyNE) | 1:19:18 |
| 03 | [CS336 2025 Lecture 3 Tutorial: Transformer Architectures and Hyperparameters](lectures/03-architectures-hyperparameters.md) | [YouTube](https://www.youtube.com/watch?v=ptFiH_bHnJw) | 1:26:59 |
| 04 | [Stanford CS336 Lecture 4 Tutorial: Mixture of Experts (MoE)](lectures/04-mixture-of-experts.md) | [YouTube](https://www.youtube.com/watch?v=LPv1KfUXLCo) | 1:22:00 |
| 05 | [Stanford CS336 Lecture 5 Tutorial: GPUs](lectures/05-gpus.md) | [YouTube](https://www.youtube.com/watch?v=6OBtO9niT00) | 1:14:17 |
| 06 | [Stanford CS336 2025 Lecture 6 Tutorial: Kernels, Triton, and LLM Operator Optimization](lectures/06-kernels-triton.md) | [YouTube](https://www.youtube.com/watch?v=E8Mju53VB00) | 1:20:18 |
| 07 | [Stanford CS336 Lecture 7 Tutorial: Parallelism 1](lectures/07-parallelism-1.md) | [YouTube](https://www.youtube.com/watch?v=l1RJcDjzK8M) | 1:24:39 |
| 08 | [Stanford CS336 Lecture 8 Tutorial: Parallel Training (Part 2)](lectures/08-parallelism-2.md) | [YouTube](https://www.youtube.com/watch?v=LHpr5ytssLo) | 1:15:06 |
| 09 | [CS336 Lecture 9 Tutorial: Scaling Laws (Part 1)](lectures/09-scaling-laws-1.md) | [YouTube](https://www.youtube.com/watch?v=6Q-ESEmDf4Q) | 1:05:15 |
| 10 | [Stanford CS336 Lecture 10 Tutorial: LLM Inference](lectures/10-inference.md) | [YouTube](https://www.youtube.com/watch?v=fcgPYo3OtV0) | 1:22:48 |
| 11 | [CS336 Lecture 11 Tutorial: Scaling Laws (Part 2)](lectures/11-scaling-laws-2.md) | [YouTube](https://www.youtube.com/watch?v=OSYuUqGBQxw) | 1:18:10 |
| 12 | [CS336 Lecture 12 Tutorial: LLM Evaluation](lectures/12-evaluation.md) | [YouTube](https://www.youtube.com/watch?v=x-R5l2HsXqM) | 1:20:45 |
| 13 | [Stanford CS336 Lecture 13 Tutorial: Pretraining Data (Data 1)](lectures/13-data-1.md) | [YouTube](https://www.youtube.com/watch?v=WePxmeXU1xg) | 1:19:02 |
| 14 | [Stanford CS336 Lecture 14: Data (Part 2) — From Raw Web Pages to Trainable Corpora](lectures/14-data-2.md) | [YouTube](https://www.youtube.com/watch?v=9Cd0THLS1t0) | 1:19:08 |
| 15 | [CS336 Lecture 15 Tutorial: Alignment, SFT, and RLHF](lectures/15-alignment-sft-rlhf.md) | [YouTube](https://www.youtube.com/watch?v=Dfu7vC9jo4w) | 1:14:47 |
| 16 | [CS336 Lecture 16 Tutorial: Alignment with Reinforcement Learning (Part 1)](lectures/16-alignment-rl-1.md) | [YouTube](https://www.youtube.com/watch?v=46f2QTDB08Q) | 1:20:29 |
| 17 | [CS336 Lecture 17 Tutorial: Alignment with Reinforcement Learning (Part 2)](lectures/17-alignment-rl-2.md) | [YouTube](https://www.youtube.com/watch?v=JdGFdViaOJk) | 1:16:04 |

## Suggested path

1. Foundations: lectures 1-3 — tokenization, PyTorch resource accounting, Transformer architecture.
2. Training systems: lectures 4-8 — MoE, GPUs, Triton kernels, distributed parallelism.
3. Training decisions and deployment: lectures 9-14 — scaling laws, inference, evaluation, data.
4. Alignment and RL: lectures 15-17 — SFT, RLHF, PPO/GRPO, RLVR, reasoning models.

## Disclaimer

This is an unofficial educational adaptation. Stanford CS336 and the original course staff own the original course materials. This repository is not affiliated with or endorsed by Stanford University.
