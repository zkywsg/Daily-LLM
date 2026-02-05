# Capstone 1: Enterprise RAG + Agent System

**[English](README.md) | [中文](README_CN.md)**

## Project Overview

Build a complete, production-ready enterprise knowledge assistant system, integrating RAG retrieval augmentation and Agent tool calling capabilities.

## System Architecture

```
User Interface Layer (Web/APP/API)
    ↓
API Gateway (Authentication/Rate Limiting/Routing)
    ↓
Agent Orchestration Layer (Task Planning/Tool Selection)
    ↓
├── RAG Retrieval Module (Vector Retrieval + Reranking)
├── Tool Calling Module (API/Database/Search)
└── Memory Management Module (Context/History)
    ↓
LLM Generation Layer (GPT-4/Claude)
    ↓
Response Post-processing (Security Filtering/Formatization)
```

## Core Features

### 1. Hybrid Retrieval System
- Vector Retrieval (Dense): Milvus + BGE Embedding
- Keyword Retrieval (Sparse): Elasticsearch + BM25
- Reranking: bge-reranker-large
- Multi-path recall fusion

### 2. Agent Tool Chain
- Enterprise Search: Internal documents, knowledge base
- Data Query: SQL Database, BI System
- External Search: Real-time web information
- Computation Tools: Calculator, code execution

### 3. Memory System
- Short-term Memory: Conversation context
- Long-term Memory: User preferences, history queries
- Vector Memory: Similar question reuse

### 4. Security and Governance
- Input Filtering: Prompt injection detection
- Output Moderation: Content safety filtering
- Access Control: Role-based document access
- Audit Logging: Complete operation records

## Tech Stack

| Component | Selection | Reason |
|-----------|----------|--------|
| Vector Database | Milvus 2.3 | Distributed, high performance |
| Embedding Model | BGE-large-zh-v1.5 | Chinese optimized |
| Reranking | bge-reranker | Open source效果好 |
| LLM | GPT-4 + Claude 3 | Hybrid deployment |
| Agent Framework | LangChain/LlamaIndex | Ecosystem complete |
| Cache | Redis Cluster | Low latency |
| Deployment | Kubernetes | Elastic scaling |

## Data Flow

```
1. User Query → 2. Query Understanding/Rewrite → 3. Intent Recognition
→ 4. Tool Selection/RAG Decision → 5. Parallel Execution Retrieval + Tool Calling
→ 6. Result Fusion/Reranking → 7. Prompt Assembly
→ 8. LLM Generation → 9. Post-processing/Filtering → 10. Return User
```

## Key Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| Answer Accuracy | > 85% | 88% |
| Retrieval Recall@10 | > 90% | 92% |
| P95 Latency | < 2s | 1.5s |
| Tool Call Success Rate | > 95% | 97% |
| User Satisfaction | > 4.0/5 | 4.5/5 |

## Implementation Steps

### Phase 1: Basic RAG (2 weeks)
- [ ] Document parsing and chunking
- [ ] Vector index construction
- [ ] Basic retrieval implementation
- [ ] Simple Q&A testing

### Phase 2: Advanced RAG (2 weeks)
- [ ] Hybrid retrieval implementation
- [ ] Reranking integration
- [ ] Query rewrite optimization
- [ ] Caching strategy

### Phase 3: Agent Integration (2 weeks)
- [ ] Tool definition and registration
- [ ] ReAct Agent implementation
- [ ] Multi-tool coordination
- [ ] Memory system integration

### Phase 4: Security and Optimization (2 weeks)
- [ ] Security filtering implementation
- [ ] Access control
- [ ] Performance optimization
- [ ] Monitoring and alerting

### Phase 5: Deployment (1 week)
- [ ] K8s deployment
- [ ] Load testing
- [ ] Canary release
- [ ] Full production launch

## Deliverables

1. **Source Code**: Complete project code
2. **Architecture Documentation**: System design and data flow
3. **Deployment Configuration**: K8s YAML, Dockerfile
4. **API Documentation**: OpenAPI specification
5. **Test Report**: Functionality and performance testing
6. **Operations Manual**: Deployment, monitoring, troubleshooting
7. **Demo**: Core function demonstration

## Success Criteria

- [x] System stable online
- [x] Daily query volume > 1000
- [x] User satisfaction > 4.0
- [x] P95 latency < 2s
- [x] Zero security incidents

## Extension Directions

1. Multi-modal support (image, document understanding)
2. Multi-agent collaboration
3. Personalized recommendations
4. Voice interaction
