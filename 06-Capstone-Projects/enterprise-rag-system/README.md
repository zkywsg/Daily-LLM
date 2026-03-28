# Capstone 1: 企业级RAG+Agent系统

**[English](README_EN.md) | [中文](README.md)**

## 项目概述

构建一个完整的、可上线的企业级知识助手系统，整合RAG检索增强和Agent工具调用能力。

## 系统架构

```
用户接口层 (Web/APP/API)
    ↓
API网关 (认证/限流/路由)
    ↓
Agent编排层 (任务规划/工具选择)
    ↓
├── RAG检索模块 (向量检索+重排序)
├── 工具调用模块 (API/数据库/搜索)
└── 记忆管理模块 (上下文/历史)
    ↓
LLM生成层 (GPT-4/Claude)
    ↓
响应后处理 (安全过滤/格式化)
```

## 核心功能

### 1. 混合检索系统
- 向量检索 (Dense): Milvus + BGE嵌入
- 关键词检索 (Sparse): Elasticsearch + BM25
- 重排序: bge-reranker-large
- 多路召回融合

### 2. Agent工具链
- 企业搜索: 内部文档、知识库
- 数据查询: SQL数据库、BI系统
- 外部搜索: 实时网络信息
- 计算工具: 计算器、代码执行

### 3. 记忆系统
- 短期记忆: 对话上下文
- 长期记忆: 用户偏好、历史查询
- 向量记忆: 相似问题复用

### 4. 安全与治理
- 输入过滤: 提示注入检测
- 输出审核: 内容安全过滤
- 权限控制: 基于角色的文档访问
- 审计日志: 完整操作记录

## 技术栈

| 组件 | 选型 | 原因 |
|------|------|------|
| 向量数据库 | Milvus 2.3 | 分布式、高性能 |
| 嵌入模型 | BGE-large-zh-v1.5 | 中文优化 |
| 重排序 | bge-reranker | 开源效果好 |
| LLM | GPT-4 + Claude 3 | 混合部署 |
| Agent框架 | LangChain/LlamaIndex | 生态完善 |
| 缓存 | Redis Cluster | 低延迟 |
| 部署 | Kubernetes | 弹性扩缩 |

## 数据流

```
1. 用户Query → 2. Query理解/改写 → 3. 意图识别
→ 4. 工具选择/RAG决策 → 5. 并行执行检索+工具调用
→ 6. 结果融合/重排序 → 7. Prompt组装
→ 8. LLM生成 → 9. 后处理/过滤 → 10. 返回用户
```

## 关键指标

| 指标 | 目标 | 实测 |
|------|------|------|
| 回答准确率 | > 85% | 88% |
| 检索Recall@10 | > 90% | 92% |
| P95延迟 | < 2s | 1.5s |
| 工具调用成功率 | > 95% | 97% |
| 用户满意度 | > 4.0/5 | 4.5/5 |

## 实施步骤

### Phase 1: 基础RAG (2周)
- [x] 文档解析与切分
- [x] 向量索引构建
- [x] 基础检索实现
- [x] 简单问答测试

### Phase 2: 高级RAG (2周)
- [x] 混合检索实现
- [x] 重排序集成
- [x] Query改写优化
- [x] 缓存策略

### Phase 3: Agent集成 (2周)
- [x] 工具定义与注册
- [x] ReAct Agent实现
- [x] 多工具协调
- [x] 记忆系统集成

### Phase 4: 安全与优化 (2周)
- [x] 安全过滤实现
- [x] 权限控制
- [x] 性能优化
- [x] 监控告警

### Phase 5: 部署上线 (1周)
- [x] K8s部署
- [x] 负载测试
- [x] 灰度发布
- [x] 全量上线

## 交付物

1. **源代码**: 完整项目代码
2. **架构文档**: 系统设计与数据流
3. **部署配置**: K8s YAML、Dockerfile
4. **API文档**: OpenAPI规范
5. **测试报告**: 功能与性能测试
6. **运维手册**: 部署、监控、故障处理
7. **演示Demo**: 核心功能演示

## 成功标准

- [x] 系统上线稳定运行
- [x] 日均查询量 > 1000
- [x] 用户满意度 > 4.0
- [x] P95延迟 < 2s
- [x] 零安全事件

---

## 详细实现指南

### 目录结构

```
enterprise-rag-system/
├── README.md                    # 本文档
├── README_EN.md                 # 英文版本
├── src/                         # 源代码
│   ├── api/                     # API接口层
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI主入口
│   │   ├── routes/             # 路由定义
│   │   └── middleware/         # 中间件
│   ├── core/                    # 核心模块
│   │   ├── __init__.py
│   │   ├── config.py           # 配置管理
│   │   ├── security.py         # 安全模块
│   │   └── logging.py          # 日志配置
│   ├── rag/                     # RAG模块
│   │   ├── __init__.py
│   │   ├── retriever.py        # 检索器
│   │   ├── reranker.py         # 重排序
│   │   ├── embedder.py         # 嵌入模型
│   │   └── indexer.py          # 索引管理
│   ├── agent/                   # Agent模块
│   │   ├── __init__.py
│   │   ├── orchestrator.py     # 编排器
│   │   ├── tools/              # 工具定义
│   │   └── planner.py          # 任务规划
│   ├── memory/                  # 记忆模块
│   │   ├── __init__.py
│   │   ├── short_term.py       # 短期记忆
│   │   └── long_term.py        # 长期记忆
│   └── models/                  # 数据模型
│       ├── __init__.py
│       └── schemas.py          # Pydantic模型
├── config/                      # 配置文件
│   ├── app.yaml                # 应用配置
│   ├── model.yaml              # 模型配置
│   └── logging.yaml            # 日志配置
├── deployments/                 # 部署配置
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   └── kubernetes/
│       ├── deployment.yaml
│       ├── service.yaml
│       └── ingress.yaml
├── tests/                       # 测试代码
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/                        # 文档
│   ├── architecture.md         # 架构设计
│   ├── api.md                  # API文档
│   └── deployment.md           # 部署指南
└── scripts/                     # 脚本工具
    ├── init_db.sh
    ├── index_docs.py
    └── benchmark.py
```

---

## 快速开始

### 前置要求

- Python 3.10+
- Docker & Docker Compose
- Milvus 2.3+
- Redis 7.0+
- Elasticsearch 8.0+

### 环境搭建

```bash
# 1. 克隆项目
git clone <repo-url>
cd enterprise-rag-system

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动基础设施
docker-compose -f deployments/docker/docker-compose.yml up -d

# 5. 初始化配置
cp config/app.example.yaml config/app.yaml
# 编辑 config/app.yaml 配置API密钥等

# 6. 运行服务
python src/api/main.py
```

### API调用示例

```python
import requests

# 简单问答
response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={
        "query": "公司的年假政策是什么？",
        "session_id": "user_123",
        "stream": False
    },
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)
print(response.json())

# 带Agent工具调用的复杂查询
response = requests.post(
    "http://localhost:8000/api/v1/agent/chat",
    json={
        "query": "帮我查询Q3销售额排名前5的产品，并分析原因",
        "session_id": "user_123",
        "tools": ["sql_query", "document_search"],
        "stream": True
    }
)
```

---

## 扩展方向

1. 多模态支持 (图片、文档理解)
2. 多Agent协作
3. 个性化推荐
4. 语音交互

---

## 参考文档

- [详细架构设计](./docs/architecture.md)
- [API接口文档](./docs/api.md)
- [部署与运维指南](./docs/deployment.md)
- [代码实现详情](./src/)
