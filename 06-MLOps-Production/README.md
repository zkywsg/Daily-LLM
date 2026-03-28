# Phase 6: MLOps 与生产工程

**[English](README_EN.md) | [中文](README.md)**

## 概览

本阶段涵盖将大语言模型部署到生产环境并进行维护的工程挑战。重点关注 AI 系统的可扩展性、可靠性和可观测性。

## 架构

### 1. [训练基础设施 (Training Infrastructure)](training-infrastructure/distributed/README.md)
构建可扩展的训练流水线。
- **分布式训练**: FSDP, DeepSpeed, 3D 并行。
- **数据流水线**: 高吞吐量数据处理。
- **训练稳定性**: 处理 Loss 尖峰和硬件故障。
- **超参数调优**: 高效搜索策略。

### 2. [模型服务 (Model Serving)](model-serving/architecture/README.md)
高效推理与部署。
- **服务架构**: vLLM, TGI, TensorRT-LLM。
- **模型压缩**: 量化 (AWQ, GPTQ), 剪枝。
- **模型仓库**: 版本控制与制品管理。

### 3. [监控与可观测性 (Monitoring & Observability)](monitoring/rag-observability/README.md)
确保系统健康与质量。
- **RAG 可观测性**: 追踪检索与生成。
- **漂移检测**: 监控数据漂移和概念漂移。
- **评估**: 自动化测试与基准测试。
- **数据治理**: 隐私与合规。

### 4. [部署基础设施 (Deployment Infrastructure)](deployment/pipeline/README.md)
面向 AI 的 DevOps。
- **CI/CD for ML**: 自动化测试与部署。
- **成本优化**: 管理 GPU 资源与成本。
- **安全性**: 可靠性与威胁防护。

---

## 时间线节点

本模块对应时间线中的以下关键工作：

| 年份 | 工作 | 核心意义 |
|------|------|---------|
| 2019 | Megatron-LM | 模型并行训练框架，大规模分布式训练基础设施确立 |
| 2021 | Megatron-LM v2 | 支持千亿参数训练，大规模并行训练成熟 |
| 2021 | GitHub Copilot | Codex 驱动的 AI 工具，开发者工具 AI 化起点 |
| 2022 | Flash Attention | IO 感知注意力，训练速度 2-4×，显存节省 5-20× |
| 2023 | vLLM / PagedAttention | LLM 推理吞吐量提升 24×，成为服务基础设施标配 |
| 2024 | Flash Attention 2/3 | 进一步提升效率，支持更长上下文训练 |
| 2025 | Speculative Decoding 普及 | 草稿模型加速推理 2-3×，进入工业标配 |
| 2025 | Benchmark 失效危机 | 传统评测集饱和，新一代评估体系重建 |

→ 完整时间线见 [00-Timeline](../00-Timeline/)

**下一阶段**: [实战项目](../07-Capstone-Projects/README.md)
