**[English](README.md) | [中文](README_CN.md)**

# 检索评估体系 (Retrieval Evaluation)

## 目录

1. [背景 (Why Retrieval Evaluation?)](#1-背景-why-retrieval-evaluation)
2. [核心概念 (Metrics, Relevance)](#2-核心概念-metrics-relevance)
3. [数学原理 (Recall, Precision, nDCG Formulas)](#3-数学原理-recall-precision-ndcg-formulas)
4. [代码实现 (Evaluation Implementation)](#4-代码实现-evaluation-implementation)
5. [实验对比 (Metric Comparison)](#5-实验对比-metric-comparison)
6. [最佳实践与常见陷阱](#6-最佳实践与常见陷阱)
7. [总结](#7-总结)

---

## 1. 背景 (Why Retrieval Evaluation?)

### 1.1 为什么需要评估？

检索质量直接决定RAG系统效果。没有评估就无法：
- 比较不同检索策略
- 发现系统瓶颈
- 指导优化方向
- 监控线上效果

### 1.2 评估挑战

- **相关性主观性**: 不同人对相关性判断不同
- **多维度质量**: 相关性、多样性、时效性
- **大规模评估**: 人工评估成本高

---

## 2. 核心概念 (Metrics, Relevance)

### 2.1 相关性定义

- **二元相关**: 相关/不相关
- **分级相关**: 高度/中度/低度/不相关 (0-3分)
- **细粒度**: 考虑位置、覆盖度等

### 2.2 评估维度

| 维度 | 说明 | 指标 |
|------|------|------|
| **相关性** | 结果与查询匹配度 | Recall, Precision |
| **排序质量** | 相关结果位置 | MRR, nDCG |
| **多样性** | 结果覆盖度 | α-NDCG |
| **时效性** | 信息新鲜度 | 时间衰减 |

---

## 3. 数学原理 (Recall, Precision, nDCG Formulas)

### 3.1 Recall@K

$$
\text{Recall@K} = \frac{|\{\text{相关文档}\} \cap \{\text{Top-K结果}\}|}{|\{\text{相关文档}\}|}
$$

### 3.2 Precision@K

$$
\text{Precision@K} = \frac{|\{\text{相关文档}\} \cap \{\text{Top-K结果}\}|}{K}
$$

### 3.3 MRR (Mean Reciprocal Rank)

$$
\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}
$$

### 3.4 nDCG (Normalized Discounted Cumulative Gain)

$$
\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i+1)}
$$

$$
\text{nDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
$$

---

## 4. 代码实现 (Evaluation Implementation)

### 4.1 基础评估实现

```python
import numpy as np

class RetrievalEvaluator:
    """检索评估器"""
    
    def __init__(self):
        self.metrics = {}
    
    def recall_at_k(self, retrieved: list, relevant: list, k: int = 10):
        """计算Recall@K"""
        retrieved_set = set(retrieved[:k])
        relevant_set = set(relevant)
        
        if not relevant_set:
            return 0.0
        
        return len(retrieved_set & relevant_set) / len(relevant_set)
    
    def precision_at_k(self, retrieved: list, relevant: list, k: int = 10):
        """计算Precision@K"""
        retrieved_set = set(retrieved[:k])
        relevant_set = set(relevant)
        
        if not retrieved_set:
            return 0.0
        
        return len(retrieved_set & relevant_set) / len(retrieved_set)
    
    def mrr(self, retrieved_list: list, relevant_list: list):
        """计算MRR"""
        rr_sum = 0
        
        for retrieved, relevant in zip(retrieved_list, relevant_list):
            for i, doc in enumerate(retrieved, 1):
                if doc in relevant:
                    rr_sum += 1 / i
                    break
        
        return rr_sum / len(retrieved_list)
    
    def dcg_at_k(self, retrieved: list, relevance_scores: dict, k: int = 10):
        """计算DCG@K"""
        dcg = 0
        for i, doc in enumerate(retrieved[:k], 1):
            rel = relevance_scores.get(doc, 0)
            dcg += (2 ** rel - 1) / np.log2(i + 1)
        return dcg
    
    def ndcg_at_k(self, retrieved: list, relevance_scores: dict, k: int = 10):
        """计算nDCG@K"""
        dcg = self.dcg_at_k(retrieved, relevance_scores, k)
        
        # 理想DCG
        ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum((2 ** rel - 1) / np.log2(i + 1) for i, rel in enumerate(ideal_rels, 1))
        
        return dcg / idcg if idcg > 0 else 0

# 使用示例
evaluator = RetrievalEvaluator()

retrieved = [1, 2, 3, 4, 5]
relevant = [1, 3, 6, 7]

print(f"Recall@5: {evaluator.recall_at_k(retrieved, relevant, k=5):.2f}")
print(f"Precision@5: {evaluator.precision_at_k(retrieved, relevant, k=5):.2f}")
```

### 4.2 批量评估

```python
def batch_evaluate(evaluator, queries, retrieved_lists, relevant_lists):
    """批量评估"""
    results = {
        "recall@5": [],
        "precision@5": [],
        "recall@10": [],
        "precision@10": []
    }
    
    for retrieved, relevant in zip(retrieved_lists, relevant_lists):
        results["recall@5"].append(evaluator.recall_at_k(retrieved, relevant, k=5))
        results["precision@5"].append(evaluator.precision_at_k(retrieved, relevant, k=5))
        results["recall@10"].append(evaluator.recall_at_k(retrieved, relevant, k=10))
        results["precision@10"].append(evaluator.precision_at_k(retrieved, relevant, k=10))
    
    # 计算平均值
    return {k: np.mean(v) for k, v in results.items()}

# 使用
queries = ["q1", "q2", "q3"]
retrieved_lists = [[1,2,3], [4,5,6], [7,8,9]]
relevant_lists = [[1,4], [5], [7,10]]

results = batch_evaluate(evaluator, queries, retrieved_lists, relevant_lists)
print(results)
```

---

## 5. 实验对比 (Metric Comparison)

### 5.1 指标对比

| 指标 | 关注点 | 适用场景 | 优点 | 缺点 |
|------|--------|---------|------|------|
| Recall@K | 覆盖率 | 确保找到所有相关内容 | 直观 | 不考虑排序 |
| Precision@K | 准确性 | 控制噪音 | 直观 | 不考虑遗漏 |
| MRR | 首位质量 | 只需要一个好结果 | 重视首位 | 忽略其他位置 |
| nDCG@K | 整体排序 | 评估完整列表质量 | 考虑分级 | 复杂 |

### 5.2 K值选择

| K值 | 适用场景 | 说明 |
|-----|---------|------|
| 5 | 高精度需求 | 只关注最相关结果 |
| 10 | 通用场景 | 平衡覆盖与精度 |
| 50 | 高覆盖需求 | 用于后续Rerank |
| 100 | 粗筛场景 | 最大化召回 |

---

## 6. 最佳实践与常见陷阱

### 6.1 最佳实践

1. **多指标综合**: 同时关注Recall和Precision
2. **分级标注**: 使用0-3分而非二元标注
3. **人工+自动**: 自动评估快速迭代，人工评估验证质量
4. **A/B测试**: 线上对比不同策略

### 6.2 常见陷阱

1. **单一指标**: 只看Recall忽视Precision
2. **测试集污染**: 训练数据和测试数据重叠
3. **静态评估**: 不随数据更新重新评估

### 6.3 评估流程

```markdown
1. 构建测试集 (100+查询)
2. 人工标注相关性
3. 运行检索系统
4. 计算各项指标
5. 对比基线模型
6. 分析错误案例
7. 迭代优化
```

---

## 7. 总结

检索评估是RAG系统优化的基础：

1. **核心指标**: Recall@K, Precision@K, nDCG@K
2. **评估维度**: 相关性、排序质量、多样性
3. **K值选择**: 根据场景选5/10/50/100
4. **持续评估**: 建立线上监控体系

**推荐指标组合**:
- 研发阶段: Recall@10 + nDCG@10
- 线上监控: Recall@5 + Precision@5
- 深度分析: MRR + 人工评估
