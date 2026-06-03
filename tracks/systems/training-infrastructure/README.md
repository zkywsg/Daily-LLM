# 训练基础设施 (Training Infrastructure)

[English](README_EN.md) | [中文](README.md)

## 这个问题从哪来

> 随着模型规模突破百亿、千亿，训练不再是单卡跑跑脚本就能完成的事情。数据流水线、分布式策略、训练稳定性、可复现性，构成了大模型工程化的基础设施层。

## 学习目标

完成后你应能回答：
1. FSDP 与 DeepSpeed 在不同规模模型下如何选型？
2. 如何设计高吞吐、低失败率的数据流水线？
3. 大模型训练中常见的损失尖峰与数值不稳定如何应对？

## 本模块内容

- [分布式训练](distributed/README.md)
- [数据流水线](data-pipeline/README.md)
- [参数高效微调基础设施](peft/README.md)
- [超参数优化](hpo/README.md)
- [训练稳定性](stability/README.md)
- [可复现性](reproducibility/README.md)

## 演进笔记

> 这一技术的遗产：训练基础设施让「堆卡」变成了系统工程，但集群调度、故障恢复、以及跨框架兼容性，仍是工业界持续投入的重点。
→ 详见 [模型服务](../model-serving/README.md)

---

**上一章**: [生产系统](../production/README.md) | **下一章**: [模型服务](../model-serving/README.md)
