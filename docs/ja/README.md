# CS336 2025 日本語チュートリアル版

これは Stanford CS336 2025 **Language Modeling from Scratch** のコミュニティ学習ガイド日本語版です。

元の講義動画は非常に優れていますが、授業 transcript には口語的な繰り返しや話題のジャンプがあります。この版では、概念、工学的直感、数式、よくある誤解、実践課題を中心に、独学しやすいチュートリアル形式へ再構成しています。

> 元コース: https://stanford-cs336.github.io/  
> YouTube playlist: https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_

## まず読むもの

- [日本語完全チュートリアル](CS336-2025-complete-tutorial-ja.md)
- [学習計画](study-plan.md)
- [用語集](glossary.md)

## 講義一覧

| # | チュートリアル | 元動画 | 時間 |
|---|----------------|--------|------|
| 01 | [CS336 2025 第1講チュートリアル：コース概観と Tokenization](lectures/01-overview-and-tokenization.md) | [YouTube](https://www.youtube.com/watch?v=SQ3fZ1sAqXI) | 1:18:55 |
| 02 | [Stanford CS336 2025 第2講チュートリアル：PyTorch と Resource Accounting](lectures/02-pytorch-resource-accounting.md) | [YouTube](https://www.youtube.com/watch?v=msHyYioAyNE) | 1:19:18 |
| 03 | [CS336 2025 第3講チュートリアル：Transformer Architecture と Hyperparameters](lectures/03-architectures-hyperparameters.md) | [YouTube](https://www.youtube.com/watch?v=ptFiH_bHnJw) | 1:26:59 |
| 04 | [Stanford CS336 第4回チュートリアル：Mixture of Experts（MoE）](lectures/04-mixture-of-experts.md) | [YouTube](https://www.youtube.com/watch?v=LPv1KfUXLCo) | 1:22:00 |
| 05 | [Stanford CS336 Lecture 5：GPU 日本語チュートリアル](lectures/05-gpus.md) | [YouTube](https://www.youtube.com/watch?v=6OBtO9niT00) | 1:14:17 |
| 06 | [Stanford CS336 2025 第6回チュートリアル：Kernel、Triton、LLM 演算子最適化](lectures/06-kernels-triton.md) | [YouTube](https://www.youtube.com/watch?v=E8Mju53VB00) | 1:20:18 |
| 07 | [Stanford CS336 第7回チュートリアル：Parallelism 1](lectures/07-parallelism-1.md) | [YouTube](https://www.youtube.com/watch?v=l1RJcDjzK8M) | 1:24:39 |
| 08 | [Stanford CS336 第8回チュートリアル：並列学習（二）](lectures/08-parallelism-2.md) | [YouTube](https://www.youtube.com/watch?v=LHpr5ytssLo) | 1:15:06 |
| 09 | [CS336 第9回チュートリアル：Scaling Laws（一）](lectures/09-scaling-laws-1.md) | [YouTube](https://www.youtube.com/watch?v=6Q-ESEmDf4Q) | 1:05:15 |
| 10 | [Stanford CS336 第10回チュートリアル：LLM Inference](lectures/10-inference.md) | [YouTube](https://www.youtube.com/watch?v=fcgPYo3OtV0) | 1:22:48 |
| 11 | [CS336 第11回チュートリアル：Scaling Laws（二）](lectures/11-scaling-laws-2.md) | [YouTube](https://www.youtube.com/watch?v=OSYuUqGBQxw) | 1:18:10 |
| 12 | [CS336 第12回チュートリアル：LLM Evaluation](lectures/12-evaluation.md) | [YouTube](https://www.youtube.com/watch?v=x-R5l2HsXqM) | 1:20:45 |
| 13 | [Stanford CS336 第13回チュートリアル：事前学習データ（Data 1）](lectures/13-data-1.md) | [YouTube](https://www.youtube.com/watch?v=WePxmeXU1xg) | 1:19:02 |
| 14 | [Stanford CS336 第14回：データ（二）——生の Web ページから学習可能コーパスへ](lectures/14-data-2.md) | [YouTube](https://www.youtube.com/watch?v=9Cd0THLS1t0) | 1:19:08 |
| 15 | [CS336 第15回 日本語チュートリアル：Alignment、SFT、RLHF](lectures/15-alignment-sft-rlhf.md) | [YouTube](https://www.youtube.com/watch?v=Dfu7vC9jo4w) | 1:14:47 |
| 16 | [CS336 第16回チュートリアル：Alignment における強化学習（一）](lectures/16-alignment-rl-1.md) | [YouTube](https://www.youtube.com/watch?v=46f2QTDB08Q) | 1:20:29 |
| 17 | [CS336 第17回チュートリアル：Alignment における強化学習（二）](lectures/17-alignment-rl-2.md) | [YouTube](https://www.youtube.com/watch?v=JdGFdViaOJk) | 1:16:04 |

## 推奨ルート

1. 基礎: Lecture 1-3 — tokenization、PyTorch resource accounting、Transformer architecture。
2. 学習システム: Lecture 4-8 — MoE、GPU、Triton kernel、distributed parallelism。
3. 学習判断とデプロイ: Lecture 9-14 — scaling laws、inference、evaluation、data。
4. Alignment と RL: Lecture 15-17 — SFT、RLHF、PPO/GRPO、RLVR、reasoning models。

## 免責

本資料は非公式の教育用 adaptation です。元の講義資料の権利は Stanford CS336 と講義スタッフおよび各権利者に帰属します。本リポジトリは Stanford University と提携・承認されたものではありません。
