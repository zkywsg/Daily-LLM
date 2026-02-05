# Phase 6: MLOps 与生产工程

**[English](README.md) | [中文](README_CN.md)**

## 概览

本阶段涵盖将大语言模型部署到生产环境并进行维护的工程挑战。重点关注 AI 系统的可扩展性、可靠性和可观测性。

## 架构

### 1. [训练基础设施 (Training Infrastructure)](training-infrastructure/distributed/README_CN.md)
构建可扩展的训练流水线。
- **分布式训练**: FSDP, DeepSpeed, 3D 并行。
- **数据流水线**: 高吞吐量数据处理。
- **训练稳定性**: 处理 Loss 尖峰和硬件故障。
- **超参数调优**: 高效搜索策略。

### 2. [模型服务 (Model Serving)](model-serving/architecture/README_CN.md)
高效推理与部署。
- **服务架构**: vLLM, TGI, TensorRT-LLM。
- **模型压缩**: 量化 (AWQ, GPTQ), 剪枝。
- **模型仓库**: 版本控制与制品管理。

### 3. [监控与可观测性 (Monitoring & Observability)](monitoring/rag-observability/README_CN.md)
确保系统健康与质量。
- **RAG 可观测性**: 追踪检索与生成。
- **漂移检测**: 监控数据漂移和概念漂移。
- **评估**: 自动化测试与基准测试。
- **数据治理**: 隐私与合规。

### 4. [部署基础设施 (Deployment Infrastructure)](deployment/pipeline/README_CN.md)
面向 AI 的 DevOps。
- **CI/CD for ML**: 自动化测试与部署。
- **成本优化**: 管理 GPU 资源与成本。
- **安全性**: 可靠性与威胁防护。

---

**下一阶段**: [实战项目](../07-Capstone-Projects/README_CN.md)
