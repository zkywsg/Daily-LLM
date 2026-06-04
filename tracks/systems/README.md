# Phase 05 · 系统与生产（2023–2025）

从模型能力到真实系统落地。
RAG、Agent、推理优化、生产监控，这一阶段把前四个 Phase 的知识变成可交付的系统。

## 本阶段内容

### [RAG 基础](rag-foundations/README.md)
检索增强生成系统的核心组件
- Chunking 策略、Embedding 模型选型
- Rerank、Query Rewriting、HyDE
- 检索评估：MRR、NDCG

### [向量数据库](vector-databases/README.md)
语义搜索的存储引擎
- HNSW、IVF、标量量化
- Milvus、Pinecone、Weaviate、Chroma

### [智能体](agents/README.md)
构建自主系统
- ReAct、Plan-and-Solve 框架
- 工具调用、记忆、多智能体协作
- LangChain / LlamaIndex 工程化

### [生产系统](production/README.md)
真实世界应用模式
- 企业搜索、代码助手、对话系统

### [训练基础设施](training-infrastructure/README.md)
分布式训练流水线
- FSDP、DeepSpeed、3D 并行
- 数据流水线、训练稳定性

### [模型服务](model-serving/README.md)
高效推理与部署
- vLLM / PagedAttention、TGI、TensorRT-LLM
- 量化：AWQ、GPTQ
- Speculative Decoding

### [监控与可观测性](monitoring/README.md)
确保系统健康与质量
- RAG 可观测性、漂移检测
- 基准测试、评估体系

### [部署基础设施](deployment/README.md)
面向 AI 的 DevOps
- CI/CD for ML、成本优化、Kubernetes

## 时间线节点

| 年份 | 工作 | 核心意义 |
|------|------|---------|
| 2023 | RAG 生产化 / LangChain | 检索增强系统工程化框架成型 |
| 2023 | AutoGPT / ReAct | Agent 破圈，多步任务执行 |
| 2023 | vLLM / PagedAttention | LLM 推理吞吐量提升 24× |
| 2024 | o1 / 推理模型 | 推理时慢思考，复杂任务准确率飞跃 |
| 2024 | MoE 主流化 | 激活参数比例下降，推理成本显著降低 |
| 2024 | Flash Attention 2/3 | 支持更长上下文训练 |
| 2025 | DeepSeek R1 | 纯 RL 训练推理能力，开源追平闭源 |
| 2025 | Speculative Decoding 普及 | 草稿模型加速推理 2-3× |

→ 完整时间线见 [../../timeline](../../timeline/)

**上一阶段**: [对齐与开源 ←](../alignment/) | **下一阶段**: [实战项目 →](../../projects/)

---

<!-- BEGIN: timeline-references (auto-generated) -->

## 涉及的时间线节点

> 本 track 在主时间线里被以下年份引用为「主题深挖」入口。

| 年份 | 节点 | 引用入口 |
|---|---|---|
| [2016](../../timeline/2016/) | AlphaGo：强化学习登台 | [系统生产总览](./) |
| [2023](../../timeline/2023/) | GPT-4 与 LLaMA：开源的反击 | [模型服务](model-serving/) |
| [2024](../../timeline/2024/) | MoE、长上下文、o1：推理时慢思考 | [模型服务架构](model-serving/architecture/) |
| [2025](../../timeline/2025/) | DeepSeek R1 与 Test-Time Compute：开源追平 | [模型服务](model-serving/) |

<!-- END: timeline-references -->
