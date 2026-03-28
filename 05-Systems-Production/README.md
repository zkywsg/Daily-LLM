# Phase 5: RAG 与 Agent 系统

**[English](README_EN.md) | [中文](README.md)**

## 概览

本阶段专注于构建高级检索系统和自主智能体。我们超越简单的提示词工程，致力于构建能够推理、检索外部知识并与世界互动的系统。

## 架构

### 1. [RAG 基础 (RAG Foundations)](rag-foundations/README.md)
检索增强生成系统的核心组件。
- **分块策略 (Chunking Strategies)**: 如何有效地切分文本。
- **Embedding 模型**: 将文本转换为向量。
- **重排序 (Rerank)**: 提高检索精度。
- **检索评估 (Retrieval Evaluation)**: MRR 和 NDCG 等指标。

### 2. [向量数据库 (Vector Databases)](vector-databases/README.md)
语义搜索的存储引擎。
- **核心概念**: HNSW, IVF, 标量量化。
- **工具**: Milvus, Pinecone, Weaviate, Chroma。

### 3. [智能体 (Agents)](agents/README.md)
构建自主系统。
- **架构**: 规划、记忆、工具。
- **模式**: ReAct, Plan-and-Solve。
- **多智能体系统**: 协作模式。
- **工具使用**: 函数调用与 API 集成。

### 4. [生产系统 (Production Systems)](production/README.md)
真实世界的应用模式。
- **企业搜索**: 安全、可扩展的搜索。
- **代码助手**: 类 Copilot 系统。
- **对话式 AI**: 上下文感知的聊天机器人。
- **质量与安全**: 护栏与治理。

---

## 时间线节点

本模块对应时间线中的以下关键工作：

| 年份 | 工作 | 核心意义 |
|------|------|---------|
| 2014 | Neural Turing Machine | 神经网络外挂可读写记忆，早期 Agent 思想雏形 |
| 2020 | RAG 论文（Lewis et al.） | 检索 + 生成结合正式提出，解决幻觉和知识截止 |
| 2021 | WebGPT | LLM 使用搜索引擎，早期 Tool Use 实践 |
| 2023 | ReAct / Chain-of-Thought | 推理 + 行动循环，Agent 工程化的理论基础 |
| 2023 | AutoGPT | LLM 自主多步任务执行，Agent 进入大众视野 |
| 2023 | LangChain / LlamaIndex | RAG 与 Agent 工程化框架，应用开发基础设施成型 |
| 2024 | 长上下文（1M tokens） | 部分替代 RAG，改变检索策略设计 |
| 2025 | 多模态 Agent 成熟化 | 视觉 + 语言 + 工具调用统一，真实任务可用阶段 |

→ 完整时间线见 [00-Timeline](../00-Timeline/)

**下一阶段**: [MLOps 与生产工程](../06-MLOps-Production/README.md)
