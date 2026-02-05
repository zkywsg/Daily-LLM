# Phase 5: RAG 与 Agent 系统

**[English](README.md) | [中文](README_CN.md)**

## 概览

本阶段专注于构建高级检索系统和自主智能体。我们超越简单的提示词工程，致力于构建能够推理、检索外部知识并与世界互动的系统。

## 架构

### 1. [RAG 基础 (RAG Foundations)](rag-foundations/README_CN.md)
检索增强生成系统的核心组件。
- **分块策略 (Chunking Strategies)**: 如何有效地切分文本。
- **Embedding 模型**: 将文本转换为向量。
- **重排序 (Rerank)**: 提高检索精度。
- **检索评估 (Retrieval Evaluation)**: MRR 和 NDCG 等指标。

### 2. [向量数据库 (Vector Databases)](vector-databases/README_CN.md)
语义搜索的存储引擎。
- **核心概念**: HNSW, IVF, 标量量化。
- **工具**: Milvus, Pinecone, Weaviate, Chroma。

### 3. [智能体 (Agents)](agents/README_CN.md)
构建自主系统。
- **架构**: 规划、记忆、工具。
- **模式**: ReAct, Plan-and-Solve。
- **多智能体系统**: 协作模式。
- **工具使用**: 函数调用与 API 集成。

### 4. [生产系统 (Production Systems)](production/README_CN.md)
真实世界的应用模式。
- **企业搜索**: 安全、可扩展的搜索。
- **代码助手**: 类 Copilot 系统。
- **对话式 AI**: 上下文感知的聊天机器人。
- **质量与安全**: 护栏与治理。

---

**下一阶段**: [MLOps 与生产工程](../06-MLOps-Production/README_CN.md)
