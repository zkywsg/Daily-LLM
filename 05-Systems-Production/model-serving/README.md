# 模型服务 (Model Serving)

[English](README_EN.md) | [中文](README.md)

## 这个问题从哪来

> 2023年，vLLM 与 PagedAttention 的出现将 LLM 推理吞吐量提升了数十倍，模型服务从此成为独立且关键的工程领域——它决定了用户感受到的延迟、成本与可用性。

## 学习目标

完成后你应能回答：
1. vLLM 的 PagedAttention 相比传统 KV Cache 管理有哪些核心优势？
2. AWQ、GPTQ 等量化方法在精度与速度之间如何权衡？
3. Speculative Decoding 为什么能显著降低推理延迟？

## 本模块内容

- [服务架构](architecture/README.md)
- [模型压缩与量化](compression/README.md)
- [模型注册与版本管理](registry/README.md)

## 演进笔记

> 这一技术的遗产：模型服务技术让大模型从"能跑"到"跑得又快又省"，但长上下文下的显存爆炸、批处理调度公平性、以及多模型混布，仍是活跃的研究方向。
→ 详见 [监控与可观测性](../monitoring/README.md)

---

**上一章**: [训练基础设施](../training-infrastructure/README.md) | **下一章**: [监控与可观测性](../monitoring/README.md)
