# 监控与可观测性 (Monitoring & Observability)

[English](README_EN.md) | [中文](README.md)

## 这个问题从哪来

> 大模型系统上线后，"能跑"只是开始。幻觉、数据漂移、回答质量衰退、用户满意度下降，这些问题往往不会在部署当天暴露。监控与可观测性成为持续保障系统健康的生命线。

## 学习目标

完成后你应能回答：
1. RAG 系统的可观测性需要关注哪些独特指标？
2. 如何检测和应对数据漂移与模型退化？
3. LLM 应用的评估体系应包含哪些维度和基准？

## 本模块内容

- [评估方法论](methodology/README.md)
- [RAG 可观测性](rag-observability/README.md)
- [漂移检测](drift/README.md)
- [基准测试](benchmarks/README.md)
- [测试策略](testing/README.md)
- [数据治理](data-governance/README.md)

## 演进笔记

> 这一技术的遗产：监控与评估让 LLM 系统从「一次性部署」走向「持续运营」，但自动化评估与人类偏好之间的鸿沟、以及多维度指标的综合权衡，仍是长期挑战。
→ 详见 [部署基础设施](../deployment/README.md)

---

**上一章**: [模型服务](../model-serving/README.md) | **下一章**: [部署基础设施](../deployment/README.md)
