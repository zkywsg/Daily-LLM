# 部署基础设施 (Deployment Infrastructure)

[English](README_EN.md) | [中文](README.md)

## 这个问题从哪来

> 当 LLM 应用从原型走向生产，部署不再只是"推代码"，而是涉及 CI/CD for ML、成本优化、弹性扩缩容、安全合规的复杂系统工程。传统 DevOps 工具链需要为 AI 场景做针对性演进。

## 学习目标

完成后你应能回答：
1. ML 系统的 CI/CD 与传统软件有哪些本质差异？
2. 如何在 Kubernetes 上高效部署和扩展 LLM 推理服务？
3. 生产级 LLM 应用的成本优化有哪些关键杠杆？

## 本模块内容

- [MLOps 流水线](pipeline/README.md)
- [CI/CD for ML](cicd/README.md)
- [成本优化](cost-optimization/README.md)
- [安全与合规](security/README.md)

## 演进笔记

> 这一技术的遗产：部署基础设施让 LLM 应用具备了工业化交付能力，但模型版本管理、A/B 测试、回滚策略、以及安全边界的持续验证，仍是 rapidly evolving 的领域。
→ 详见 [实战项目](../../../projects/README.md)

---

**上一章**: [监控与可观测性](../monitoring/README.md) | **下一章**: [实战项目](../../../projects/README.md)
