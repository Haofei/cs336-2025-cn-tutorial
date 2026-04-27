# 学习计划建议

这套资料有两种典型用法。

## 7 天快速路线

适合已经有 ML/PyTorch 基础、想快速建立 CS336 全局地图的人。

- Day 1: Lecture 1-3，重点是 tokenization、Transformer、resource accounting。
- Day 2: Lecture 4-6，重点是 MoE、GPU、Triton/kernel。
- Day 3: Lecture 7-8，重点是 data/tensor/pipeline parallelism、FSDP、通信开销。
- Day 4: Lecture 9-11，重点是 scaling laws、compute optimality、实验外推。
- Day 5: Lecture 10 & 12，重点是 inference serving、KV cache、evaluation。
- Day 6: Lecture 13-14，重点是 data pipeline、filtering、dedup、mixture。
- Day 7: Lecture 15-17，重点是 SFT/RLHF/RLVR、PPO/GRPO、reward hacking。

## 4 周深入路线

适合想真正转向 LLM Infra / AI Engineer / Research Engineer 的学习者。

### Week 1: 模型与训练基本功

阅读 Lecture 1-3。实践建议：

- 自己实现一个最小 BPE tokenizer。
- 用 PyTorch 写一个 tiny Transformer block。
- 估算一次 forward/backward 的参数量、activation、optimizer state 和 FLOPs。

### Week 2: GPU 与分布式训练

阅读 Lecture 4-8。实践建议：

- 用 profiler 比较 naive PyTorch、torch.compile、fused op 的差异。
- 写一个简单 Triton elementwise kernel。
- 理解 DDP、FSDP、tensor parallel、pipeline parallel 的通信模式。

### Week 3: 训练决策、推理和评估

阅读 Lecture 9-14。实践建议：

- 用小实验拟合 loss vs compute 的趋势。
- 比较 greedy、temperature、top-p 等 decoding 策略。
- 设计一个小型 evaluation suite，并检查 contamination 风险。
- 搭一个数据清洗 pipeline：HTML/text extraction、language filter、dedup、quality filter。

### Week 4: Alignment 与 RL

阅读 Lecture 15-17。实践建议：

- 梳理 SFT、DPO、RLHF、RLVR 的数据需求和优化目标。
- 实现一个 Bradley-Terry preference model 的小例子。
- 分析 reward hacking 案例：奖励函数哪里被钻空子？

## 学习原则

1. 先读教程建立结构，再回看视频解决疑点。
2. 每章至少写 5 条自己的 takeaway。
3. 能用公式解释的地方不要只背结论。
4. 能用小代码验证的地方尽量动手。
5. 对 AI Infra 方向，优先精学 Lecture 2、5、6、7、8、10。
