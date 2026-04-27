# Study Plan

## 7-day fast track

- Day 1: Lectures 1-3 — tokenization, Transformer architecture, and resource accounting.
- Day 2: Lectures 4-6 — MoE, GPU architecture, kernels, and Triton.
- Day 3: Lectures 7-8 — data parallelism, tensor parallelism, pipeline parallelism, FSDP, and communication cost.
- Day 4: Lectures 9 and 11 — scaling laws, compute-optimal training, and extrapolation.
- Day 5: Lectures 10 and 12 — inference serving, KV cache, decoding, and evaluation.
- Day 6: Lectures 13-14 — data sources, filtering, deduplication, and data mixture decisions.
- Day 7: Lectures 15-17 — SFT, RLHF, PPO/GRPO, RLVR, and reward hacking.

## 4-week deep track

### Week 1: Model and training fundamentals

Read lectures 1-3. Implement a tiny tokenizer, a small Transformer block, and a simple FLOPs/memory estimator.

### Week 2: GPU and distributed training

Read lectures 4-8. Profile PyTorch code, try `torch.compile`, write a small Triton kernel, and compare DDP/FSDP/tensor/pipeline parallelism.

### Week 3: Scaling, inference, evaluation, and data

Read lectures 9-14. Fit a toy scaling curve, compare decoding strategies, design a small evaluation suite, and build a minimal data-cleaning pipeline.

### Week 4: Alignment and RL

Read lectures 15-17. Trace the SFT → preference data → reward model → RLHF/RLVR pipeline. Implement a small Bradley-Terry preference model and analyze reward hacking examples.

## Learning principles

1. Build the conceptual map first; return to videos for unclear parts.
2. Write down at least five takeaways per lecture.
3. For formulas, understand what each variable means and when the approximation applies.
4. For systems topics, always ask: what is the bottleneck — compute, memory, communication, or latency?
5. For AI Infra, prioritize lectures 2, 5, 6, 7, 8, and 10.
