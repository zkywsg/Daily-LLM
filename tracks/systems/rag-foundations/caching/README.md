# 语义缓存与复用 (Semantic Caching and Reuse)

## 目录

1. [背景 (Why Semantic Cache?)](#1-背景-why-semantic-cache)
2. [核心概念 (Cache Types, Similarity)](#2-核心概念-cache-types-similarity)
3. [数学原理 (Similarity Thresholds, Hit Rate)](#3-数学原理-similarity-thresholds-hit-rate)
4. [代码实现 (Cache Implementation)](#4-代码实现-cache-implementation)
5. [实验对比 (Cache Impact on Cost/Latency)](#5-实验对比-cache-impact-on-costlatency)
6. [最佳实践与常见陷阱](#6-最佳实践与常见陷阱)
7. [总结](#7-总结)

---

## 1. 背景 (Why Semantic Cache?)

### 1.1 为什么需要语义缓存？

RAG和LLM应用面临高并发查询，许多查询语义相似但表述不同。语义缓存通过存储和复用相似查询的结果，显著降低成本和延迟。

**价值**:
- **成本降低**: 减少重复嵌入计算和LLM调用
- **延迟降低**: 缓存命中时直接返回结果
- **用户体验**: 更快响应时间

### 1.2 语义缓存 vs 精确匹配缓存

| 特性 | 精确匹配 | 语义缓存 |
|------|---------|---------|
| 匹配方式 | 字符串完全相同 | 语义相似 |
| 容错性 | 无 | 高 |
| 命中率 | 低 | 高 |
| 复杂度 | 简单 | 较复杂 |

**示例**:
- 精确匹配: "什么是Python?" ≠ "Python是什么?"
- 语义缓存: 两者视为相似查询

---

## 2. 核心概念 (Cache Types, Similarity)

### 2.1 缓存类型

#### 2.1.1 查询缓存

缓存查询→检索结果或最终答案的映射。

#### 2.1.2 嵌入缓存

缓存查询的嵌入向量，避免重复编码。

#### 2.1.3 KV Cache (推理缓存)

LLM推理时缓存Key-Value对，加速生成。

### 2.2 相似度阈值

**关键决策**: 多相似的查询可以复用缓存？

- **阈值过高**: 命中率低，缓存无效
- **阈值过低**: 误命中，返回不相关结果

**推荐阈值**: 余弦相似度 ≥ 0.95

---

## 3. 数学原理 (Similarity Thresholds, Hit Rate)

### 3.1 缓存命中率

$$
\text{Hit Rate} = \frac{\text{缓存命中次数}}{\text{总查询次数}}
$$

### 3.2 成本节省

$$
\text{成本节省} = \text{Hit Rate} \times (C_{\text{compute}} + C_{\text{llm}}) \times N_{\text{queries}}
$$

### 3.3 相似度阈值选择

**最优阈值**:
$$
\theta^* = \arg\max_\theta [\text{Hit Rate}(\theta) \times \text{Accuracy}(\theta)]
$$

---

## 4. 代码实现 (Cache Implementation)

### 4.1 语义缓存实现

```python
import numpy as np
from typing import Dict, Tuple, Optional
import time

class SemanticCache:
    """语义缓存实现"""
    
    def __init__(self, embedder, similarity_threshold=0.95, max_size=1000):
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.cache = {}  # query_embedding -> (result, timestamp)
        self.access_count = {}
    
    def get(self, query: str) -> Tuple[bool, Optional[str]]:
        """获取缓存结果"""
        query_emb = self.embedder.encode(query)
        
        # 查找相似查询
        for cached_emb, (result, _) in self.cache.items():
            similarity = np.dot(query_emb, cached_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(cached_emb)
            )
            
            if similarity >= self.threshold:
                self.access_count[tuple(cached_emb)] += 1
                return True, result
        
        return False, None
    
    def put(self, query: str, result: str):
        """存入缓存"""
        if len(self.cache) >= self.max_size:
            # LRU淘汰
            self._evict_lru()
        
        query_emb = tuple(self.embedder.encode(query))
        self.cache[query_emb] = (result, time.time())
        self.access_count[query_emb] = 1
    
    def _evict_lru(self):
        """淘汰最少使用"""
        if not self.cache:
            return
        
        lru_key = min(self.access_count, key=self.access_count.get)
        del self.cache[lru_key]
        del self.access_count[lru_key]

# 使用示例
class MockEmbedder:
    def encode(self, text):
        return np.random.random(128)

cache = SemanticCache(MockEmbedder(), similarity_threshold=0.95)

# 存入
cache.put("什么是Python?", "Python是一种编程语言...")

# 查询 (语义相似)
hit, result = cache.get("Python是什么?")
print(f"缓存命中: {hit}")
```

### 4.2 KV Cache实现

```python
class KVCache:
    """KV Cache for LLM inference"""
    
    def __init__(self, max_seq_len=2048):
        self.max_seq_len = max_seq_len
        self.k_cache = {}  # session_id -> key tensors
        self.v_cache = {}  # session_id -> value tensors
    
    def get(self, session_id: str, start_pos: int):
        """获取缓存的KV"""
        if session_id in self.k_cache:
            return self.k_cache[session_id][:start_pos], self.v_cache[session_id][:start_pos]
        return None, None
    
    def put(self, session_id: str, k, v):
        """存入KV"""
        self.k_cache[session_id] = k
        self.v_cache[session_id] = v
    
    def clear(self, session_id: str):
        """清除缓存"""
        if session_id in self.k_cache:
            del self.k_cache[session_id]
            del self.v_cache[session_id]

# 使用
kv_cache = KVCache()
```

---

## 5. 实验对比 (Cache Impact on Cost/Latency)

### 5.1 语义缓存效果

| 命中率 | 延迟降低 | 成本降低 | 适用场景 |
|--------|---------|---------|---------|
| 20% | 20% | 20% | 多样化查询 |
| 40% | 40% | 40% | 中等重复 |
| 60% | 60% | 60% | 高重复场景 |
| 80% | 80% | 80% | FAQ类应用 |

### 5.2 阈值选择影响

| 阈值 | 命中率 | 准确率 | 推荐 |
|------|--------|--------|------|
| 0.90 | 65% | 85% | 宽松 |
| 0.95 | 45% | 95% | 推荐 |
| 0.98 | 25% | 99% | 严格 |

**结论**: 0.95是最佳平衡点。

---

## 6. 最佳实践与常见陷阱

### 6.1 最佳实践

1. **分层缓存**: 嵌入缓存 + 结果缓存 + KV Cache
2. **动态阈值**: 根据准确率反馈调整阈值
3. **TTL策略**: 设置缓存过期时间
4. **LRU淘汰**: 限制缓存大小，淘汰冷数据

### 6.2 常见陷阱

1. **阈值过低**: 误命中率高，返回错误结果
2. **不清理缓存**: 缓存膨胀，内存爆炸
3. **忽视上下文**: 多轮对话中上下文变化未考虑

### 6.3 推荐配置

```markdown
- 相似度阈值: 0.95
- 最大缓存: 10,000条
- 过期时间: 24小时
- 淘汰策略: LRU
```

---

## 7. 总结

语义缓存是降低RAG/LLM成本的有效手段：

1. **核心机制**: 语义相似查询复用结果
2. **关键参数**: 相似度阈值 0.95
3. **分层策略**: 嵌入缓存 + 结果缓存 + KV Cache
4. **命中率**: FAQ场景可达60-80%

**成本节省**: 典型应用可节省30-50%成本。
