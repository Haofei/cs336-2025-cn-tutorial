# 学習計画

## 7日間の高速ルート

- Day 1: Lecture 1-3 — tokenization、Transformer architecture、resource accounting。
- Day 2: Lecture 4-6 — MoE、GPU architecture、kernel、Triton。
- Day 3: Lecture 7-8 — data parallelism、tensor parallelism、pipeline parallelism、FSDP、通信コスト。
- Day 4: Lecture 9 と 11 — scaling laws、compute-optimal training、外挿。
- Day 5: Lecture 10 と 12 — inference serving、KV cache、decoding、evaluation。
- Day 6: Lecture 13-14 — データソース、filtering、deduplication、data mixture。
- Day 7: Lecture 15-17 — SFT、RLHF、PPO/GRPO、RLVR、reward hacking。

## 4週間の深掘りルート

### Week 1: モデルと学習の基礎

Lecture 1-3 を読みます。小さな tokenizer、Transformer block、FLOPs/メモリ見積もり器を実装してみましょう。

### Week 2: GPU と分散学習

Lecture 4-8 を読みます。PyTorch code を profiling し、`torch.compile` を試し、小さな Triton kernel を書き、DDP/FSDP/tensor/pipeline parallelism を比較します。

### Week 3: Scaling、推論、評価、データ

Lecture 9-14 を読みます。簡単な scaling curve を fitting し、decoding strategy を比較し、小さな evaluation suite と data-cleaning pipeline を作ります。

### Week 4: Alignment と RL

Lecture 15-17 を読みます。SFT → preference data → reward model → RLHF/RLVR の流れを追い、小さな Bradley-Terry preference model を実装し、reward hacking 例を分析します。

## 学習原則

1. まず全体地図を作り、不明点だけ動画に戻る。
2. 各講義で少なくとも 5 個の takeaway を書く。
3. 数式では、各変数の意味と近似が成り立つ条件を理解する。
4. システムの話では常に、ボトルネックが compute、memory、communication、latency のどれかを考える。
5. AI Infra を重視するなら Lecture 2、5、6、7、8、10 を優先する。
