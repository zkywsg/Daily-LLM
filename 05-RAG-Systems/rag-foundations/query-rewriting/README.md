[English](README_EN.md) | [中文](README.md)

# Query改写与扩展 (Query Rewriting and Expansion)

## 目录

1. [背景 (Why Query Rewriting?)](#1-背景-why-query-rewriting)
2. [核心概念 (Query Types, Expansion Methods)](#2-核心概念-query-types-expansion-methods)
3. [数学原理 (Expansion Algorithms, HyDE)](#3-数学原理-expansion-algorithms-hyde)
4. [代码实现 (Query Transformation)](#4-代码实现-query-transformation)
5. [实验对比 (Rewrite Impact on Retrieval)](#5-实验对比-rewrite-impact-on-retrieval)
6. [最佳实践与常见陷阱](#6-最佳实践与常见陷阱)
7. [总结](#7-总结)

---

## 1. 背景 (Why Query Rewriting?)

### 1.1 现实检索中的"查询鸿沟"

Query改写与扩展（Query Rewriting and Expansion）是连接用户语言与知识库语言的桥梁。现实场景中，查询鸿沟通常来自以下几类偏差：

- **词汇鸿沟 (Vocabulary Gap)**: 用户用"车"搜索，文档里写"车辆"。
- **语义鸿沟 (Semantic Gap)**: 用户写"续航焦虑"，文档只出现"电池容量不足"。
- **上下文缺失 (Context Loss)**: 对话场景里"它有什么特点？"缺乏指代对象。
- **意图不清 (Intent Uncertainty)**: "苹果"可能是水果、公司或品牌。
- **复杂需求 (Complex Needs)**: "Python创始人创立的公司有哪些？"需要多跳检索。

这些问题导致：召回率（Recall）降低、结果偏离意图、用户体验下降。Query改写通过**理解 (Understanding)**、**扩展 (Expansion)**、**分解 (Decomposition)**、**多跳检索 (Multi-hop Retrieval)** 与 **伪相关反馈 (Pseudo-Relevance Feedback, PRF)**，让检索系统"听懂人话"。

### 1.2 Query改写的价值

- **提升检索质量 (Retrieval Quality)**: 更高的Recall、NDCG、MRR。
- **提升鲁棒性 (Robustness)**: 对拼写错误、口语化表达更不敏感。
- **降低成本 (Cost)**: 通过轻量改写提升效果，减少昂贵LLM调用。
- **支持复杂任务 (Complex Task Support)**: 多跳问题、跨文档推理。

### 1.3 典型应用场景

- **搜索引擎 (Search Engine)**: 词汇扩展与意图识别。
- **RAG系统 (Retrieval-Augmented Generation)**: 生成前的查询重写。
- **企业知识库 (Enterprise KB)**: 实体标准化、缩写展开。
- **对话系统 (Conversational AI)**: 基于上下文补全查询。

### 1.4 Query改写的系统位置 (Pipeline Position)

在典型的RAG或搜索架构中，Query改写位于检索之前：

```
用户输入 → Query理解 → Query改写/扩展 → 检索 → 重排 → 生成/展示
```

在工程上常见的布局：

- **前置轻量改写**：规则化、拼写纠错、缩写扩展。
- **策略选择器**：根据意图和复杂度决定是否使用LLM。
- **多路改写并行**：生成多个改写候选并并行检索。
- **融合与重排**：多路召回合并，再重排。

### 1.5 现实挑战 (Practical Challenges)

- **意图漂移 (Intent Drift)**：改写过度导致查询偏离。
- **噪声累积 (Noise Accumulation)**：扩展词过多引入无关结果。
- **成本与延迟 (Cost/Latency)**：HyDE、LLM改写高成本。
- **长尾查询 (Long-tail Queries)**：训练数据不足。
- **多语言与混合语言 (Mixed Language)**：中英混写、缩写等。

解决这些挑战通常需要**策略分层**、**动态路由**与**监控评估**。

### 1.6 Query改写与扩展的边界

Query改写与扩展并非"越多越好"，需要明确边界：

- **改写 (Rewriting)**: 侧重纠正、规范与语义澄清。
- **扩展 (Expansion)**: 侧重增加候选词以提升召回。
- **重排 (Rerank)**: 侧重在召回结果中调整排序。

当扩展导致噪声显著增加时，应转向"重排策略"而非继续扩展。

| 目标 | 适合方法 | 不适合方法 |
|------|---------|------------|
| 提升召回 | 同义词扩展、PRF | 仅靠重排 |
| 消除歧义 | 意图识别、实体对齐 | 盲目扩展 |
| 降低成本 | 规则改写 | 高成本LLM改写 |

### 1.7 Query改写的评估维度

评估Query改写时不应只看单一指标，建议从以下维度综合评估：

- **准确性**: 改写是否保持原意。
- **覆盖率**: 是否显著提升召回。
- **稳定性**: 高频查询是否稳定提升。
- **成本**: 计算开销与LLM调用成本。
- **用户体验**: 是否减少重复查询与错误点击。

---

## 2. 核心概念 (Query Types, Expansion Methods)

### 2.1 查询类型 (Query Types)

| 类型 | 示例 | 典型问题 | 处理策略 |
|------|------|---------|---------|
| **简单查询** | "Python教程" | 词汇缺失 | 同义词扩展 |
| **歧义查询** | "苹果" | 意图不清 | 意图识别 + 多路检索 |
| **复杂查询** | "比较Python和Java优缺点" | 多维度需求 | 查询分解 |
| **多跳查询** | "Python创始人创立的公司" | 需中间实体 | 多跳检索 |
| **对话查询** | "它有什么特点？" | 上下文缺失 | 指代消解 + 上下文补全 |

### 2.2 Query理解与意图识别 (Query Understanding & Intent Classification)

**Query理解**是改写的前置步骤，通常包括：

- **意图识别 (Intent Classification)**: 确定用户想做什么。
- **实体识别 (NER)**: 提取实体并标准化。
- **主题归一 (Topic Normalization)**: 统一同义表达。
- **结构化解析 (Query Parsing)**: 识别限制条件、时间范围、比较关系。

示例：

查询："苹果 2023 财报"

- 意图：财务信息查询
- 实体：苹果公司 (Apple Inc.)
- 约束：时间=2023

### 2.3 Query改写与扩展方法 (Rewriting & Expansion Methods)

Query改写通常包含以下方法，并可组合使用：

#### 2.3.1 词面级改写 (Lexical Rewriting)

- **拼写纠错 (Spell Correction)**: "pyhton" → "python"
- **大小写与格式规范化 (Normalization)**: "iPhone 15" → "iphone 15"
- **缩写扩展 (Abbreviation Expansion)**: "NLP" → "自然语言处理 (Natural Language Processing)"

#### 2.3.2 同义词扩展 (Synonym Expansion)

- **WordNet**: 词典级同义词扩展，适合通用语言。
- **Embedding-based**: 使用向量近邻扩展语义相似词。
- **LLM-based**: 用模型生成多样表达。

#### 2.3.3 Query分解 (Query Decomposition)

将复杂查询拆成多个可检索子查询，尤其适合比较、因果、多实体问题。

示例：

- "Python和Java优缺点比较" → ["Python 优点 缺点", "Java 优点 缺点"]

#### 2.3.4 多跳检索 (Multi-hop Retrieval)

先检索中间实体，再基于中间实体构造第二跳查询。

示例：

- Q1: "Python 创始人" → "Guido van Rossum"
- Q2: "Guido van Rossum 创建的公司" → "Dropbox"

#### 2.3.5 伪相关反馈 (Pseudo-Relevance Feedback, PRF)

利用检索出的Top-K文档作为"伪相关"样本，自动扩展查询。

#### 2.3.6 HyDE (Hypothetical Document Embedding)

通过LLM生成假设文档（Hypothetical Document），用其Embedding进行检索。

### 2.4 Query变换技术 (Transformation Techniques)

Query改写不仅仅是"加词"，还包括**结构化变换**：

- **标准化 (Normalization)**: 全角转半角、大小写统一、空格处理。
- **拼写纠错 (Spell Correction)**: 编辑距离或统计模型纠错。
- **缩写扩展 (Abbreviation Expansion)**: "RAG" → "检索增强生成 (Retrieval-Augmented Generation)"。
- **实体对齐 (Entity Linking)**: "苹果" → "Apple Inc."。
- **字段增强 (Field Boosting)**: 对标题或摘要字段权重提升。
- **布尔结构化 (Boolean Rewriting)**: "A B" → "A AND B"。
- **时间与范围约束 (Time/Range Constraints)**: "2020-2023"解析为时间过滤。

### 2.5 扩展来源 (Expansion Sources)

扩展词可以来自不同知识源：

- **词典/词库 (Lexicon)**: WordNet、同义词词林。
- **Embedding近邻 (Embedding Neighbors)**: 语义相似度近邻。
- **点击日志 (Click Logs)**: 查询-点击的共现扩展。
- **知识图谱 (Knowledge Graph)**: 实体别名、上下位词。
- **文档反馈 (Document Feedback)**: PRF从检索结果中扩展。

### 2.6 多跳检索模式 (Multi-hop Patterns)

多跳检索不是单一策略，常见模式包括：

- **实体链路 (Entity Chain)**: 人物 → 公司 → 项目。
- **事件链路 (Event Chain)**: 事件 → 时间 → 影响。
- **因果链路 (Causal Chain)**: 原因 → 结果 → 证据。
- **比较链路 (Comparison Chain)**: A特性 → B特性 → 综合对比。

### 2.7 PRF与HyDE的组合策略

PRF与HyDE可以组合使用：

1. 先用PRF扩展词汇，提高召回。
2. 再用HyDE生成假设文档，提高语义对齐。

该策略适合高价值查询，但需注意延迟与成本。

### 2.8 候选改写的选择与排序 (Candidate Selection)

多路改写后需要**候选选择**与**排序融合**：

- **规则过滤**: 去除与原意偏离的候选。
- **长度约束**: 避免过长Query导致稀疏检索性能下降。
- **相似度筛选**: 使用Embedding过滤语义偏离候选。
- **多臂策略**: 使用历史点击反馈动态调整候选权重。

在工程实践中常见的融合策略：

1. 规则候选优先检索，保证稳定性。
2. LLM候选并行检索，提升召回。
3. 结果融合后进行统一重排。

---

## 3. 数学原理 (Expansion Algorithms, HyDE)

### 3.1 同义词扩展权重模型 (Weighted Expansion)

给原始查询词和扩展词分配不同权重：

$$
Q_{\text{expanded}} = \{(t, w_t)\} = \{(t_i, 1)\} \cup \{(s_j, \alpha_j)\}
$$

其中 $t_i$ 是原始词，$s_j$ 是同义词，$\alpha_j \in (0,1)$。

### 3.2 Rocchio算法 (PRF经典公式)

Rocchio算法用于PRF扩展查询向量：

$$
\vec{q}' = \alpha \vec{q} + \frac{\beta}{|D_r|} \sum_{d \in D_r} \vec{d} - \frac{\gamma}{|D_n|} \sum_{d \in D_n} \vec{d}
$$

其中 $D_r$ 为伪相关文档集合，$D_n$ 为非相关集合（可省略），$\alpha,\beta,\gamma$ 为权重。

### 3.2.1 BM25与扩展词融合

BM25是传统检索的核心评分函数：

$$
\text{BM25}(Q, D) = \sum_{t \in Q} IDF(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot (1 - b + b \cdot |D|/\text{avgdl})}
$$

扩展词加入后可以进行权重融合：

$$
\text{BM25}_{\text{expanded}}(Q', D) = \sum_{t \in Q'} w_t \cdot \text{BM25}(t, D)
$$

### 3.2.2 RM3 (Relevance Model 3)

RM3是PRF常用模型之一：

$$
P(w|R) = \sum_{d \in D_r} P(w|d) \cdot P(d|Q)
$$

扩展后的查询模型：

$$
P'(w|Q) = (1-\lambda) P(w|Q) + \lambda P(w|R)
$$

### 3.2.3 KLD扩展 (Kullback-Leibler Divergence)

通过最大化查询扩展词与相关文档分布的接近程度：

$$
\text{KLD}(R||C) = \sum_w P(w|R) \log \frac{P(w|R)}{P(w|C)}
$$

高KLD的词代表在相关文档中更具区分性。

### 3.3 Query分解评分 (Decomposition Scoring)

分解后的多子查询与文档匹配：

$$
\text{Score}(Q, D) = \sum_{i=1}^m w_i \cdot \text{sim}(Q_i, D)
$$

其中 $w_i$ 可由意图重要度、子查询长度或学习模型给出。

### 3.4 意图分类 (Intent Classification)

多类别意图识别通常采用Softmax：

$$
P(y=k|x) = \frac{\exp(\mathbf{w}_k^T \mathbf{x})}{\sum_j \exp(\mathbf{w}_j^T \mathbf{x})}
$$

其中 $x$ 是查询向量表示（TF-IDF或Embedding）。

### 3.5 HyDE原理 (Hypothetical Document Embedding)

HyDE的核心思想是使用LLM生成"假设答案"文本 $H$，再用其Embedding检索：

$$
\text{Score}_{\text{HyDE}}(Q, D) = \cos(\mathbf{e}(H), \mathbf{e}(D))
$$

其中 $H = \text{LLM}(Q)$，$\mathbf{e}(\cdot)$ 为Embedding模型。

### 3.5.1 翻译模型 (Translation Model) 用于Query扩展

将Query扩展视为"翻译问题"：

$$
P(w|Q) = \sum_{t \in Q} P(w|t) P(t|Q)
$$

其中 $P(w|t)$ 可由共现、点击日志或Embedding近似估计。

### 3.5.2 多跳检索的路径评分 (Path Scoring)

将多跳检索建模为路径评分：

$$
\text{Score}(Q, D) = \sum_{t=1}^T \lambda_t \cdot \text{sim}(Q_t, D)
$$

其中 $\lambda_t$ 为跳级权重，保证前几跳影响更大。

### 3.6 多跳检索的状态转移 (Multi-hop Retrieval)

多跳检索可建模为状态转移：

$$
S_{t+1} = f(S_t, R(Q_t))
$$

其中 $S_t$ 是当前推理状态，$R(Q_t)$ 是第 $t$ 跳的检索结果，$f$ 用于更新中间实体。

### 3.7 Query Likelihood与平滑

Query Likelihood模型常用于生成式检索：

$$
P(Q|D) = \prod_{t \in Q} P(t|D)
$$

实际中会采用平滑：

$$
P(t|D) = (1-\mu) \cdot \frac{f(t, D)}{|D|} + \mu \cdot \frac{f(t, C)}{|C|}
$$

其中 $C$ 为语料库，$\mu$ 为平滑系数。

### 3.8 稀疏与稠密融合评分 (Sparse + Dense Fusion)

融合稀疏与稠密评分的线性模型：

$$
\text{Score}(Q, D) = \alpha \cdot \text{BM25}(Q, D) + (1-\alpha) \cdot \cos(\mathbf{e}(Q), \mathbf{e}(D))
$$

其中 $\alpha$ 控制权重，常用在混合检索场景。

### 3.9 意图路由的概率决策

改写策略可以视为一个多臂选择问题，最简单的策略是基于意图概率阈值：

$$
\text{Strategy} = \arg\max_s P(s | Q)
$$

当 $P(\text{complex}|Q) > \tau$ 时，启用Query分解或HyDE。

---

## 4. 代码实现 (Query Transformation)

> 说明：以下示例均为可运行的Python代码（需要标准库，部分示例可选安装`nltk`）。示例包含中文注释，便于理解。

### 4.1 Query理解与意图识别 (Intent Classification)

```python
from typing import List
from collections import Counter


class SimpleIntentClassifier:
    """基于规则的意图分类器（演示版）"""

    def __init__(self):
        # 简单关键词规则
        self.intent_rules = {
            "compare": ["比较", "对比", "区别"],
            "definition": ["是什么", "定义", "含义"],
            "howto": ["怎么", "如何", "步骤"],
            "finance": ["财报", "利润", "营收"],
        }

    def predict(self, query: str) -> str:
        """根据关键词预测意图"""
        for intent, keywords in self.intent_rules.items():
            if any(k in query for k in keywords):
                return intent
        return "general"


class QueryUnderstanding:
    """Query理解：意图 + 实体提取（简化版）"""

    def __init__(self, classifier: SimpleIntentClassifier):
        self.classifier = classifier

    def extract_entities(self, query: str) -> List[str]:
        """极简实体抽取：基于大写英文和数字模式"""
        entities = []
        buffer = []
        for ch in query:
            if ch.isascii() and (ch.isalpha() or ch.isdigit()):
                buffer.append(ch)
            else:
                if buffer:
                    entities.append("".join(buffer))
                    buffer = []
        if buffer:
            entities.append("".join(buffer))
        return entities

    def understand(self, query: str) -> dict:
        """返回结构化Query信息"""
        intent = self.classifier.predict(query)
        entities = self.extract_entities(query)
        return {
            "intent": intent,
            "entities": entities,
            "length": len(query),
        }


classifier = SimpleIntentClassifier()
understanding = QueryUnderstanding(classifier)
print(understanding.understand("苹果 2023 财报"))
```

#### 4.1.1 朴素贝叶斯意图分类（无第三方依赖）

```python
from typing import List, Dict
from collections import defaultdict
import math


class NaiveBayesIntent:
    """朴素贝叶斯意图分类器（演示版）"""

    def __init__(self):
        self.class_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.vocab = set()

    def train(self, samples: List[Dict[str, str]]):
        """训练数据格式: {"text": "...", "label": "intent"}"""
        for s in samples:
            label = s["label"]
            words = s["text"].split()
            self.class_counts[label] += 1
            for w in words:
                self.word_counts[label][w] += 1
                self.vocab.add(w)

    def predict(self, text: str) -> str:
        """输出概率最高的意图"""
        words = text.split()
        total_classes = sum(self.class_counts.values())
        best_label, best_score = None, float("-inf")
        for label, c in self.class_counts.items():
            log_prob = math.log(c / total_classes)
            total_words = sum(self.word_counts[label].values()) + len(self.vocab)
            for w in words:
                # 拉普拉斯平滑
                word_count = self.word_counts[label].get(w, 0) + 1
                log_prob += math.log(word_count / total_words)
            if log_prob > best_score:
                best_label, best_score = label, log_prob
        return best_label or "general"


training = [
    {"text": "如何 安装 python", "label": "howto"},
    {"text": "python 是 什么", "label": "definition"},
    {"text": "苹果 财报 2023", "label": "finance"},
    {"text": "java 和 python 比较", "label": "compare"},
]

clf = NaiveBayesIntent()
clf.train(training)
print(clf.predict("python 是 什么"))
```

#### 4.1.2 Query解析与约束抽取

```python
import re
from typing import Dict


def parse_constraints(query: str) -> Dict[str, str]:
    """解析时间与范围约束（简化版）"""
    constraints = {}
    year = re.findall(r"(19\d{2}|20\d{2})", query)
    if year:
        constraints["year"] = year[-1]
    # 示例：价格区间
    m = re.search(r"(\d+)-(\d+)元", query)
    if m:
        constraints["price_min"] = m.group(1)
        constraints["price_max"] = m.group(2)
    return constraints


print(parse_constraints("苹果 2023 财报"))
print(parse_constraints("手机 3000-5000元"))
```

#### 4.1.3 拼写纠错（编辑距离）

```python
def edit_distance(a: str, b: str) -> int:
    """计算Levenshtein编辑距离"""
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # 删除
                dp[i][j - 1] + 1,      # 插入
                dp[i - 1][j - 1] + cost  # 替换
            )
    return dp[-1][-1]


def correct_spelling(token: str, vocab: List[str], max_dist: int = 2) -> str:
    """从词表中找最接近的词"""
    best = token
    best_dist = max_dist + 1
    for v in vocab:
        d = edit_distance(token, v)
        if d < best_dist:
            best_dist = d
            best = v
    return best


vocab = ["python", "pytorch", "pandas", "numpy"]
print(correct_spelling("pyhton", vocab))
```

### 4.2 同义词扩展 (WordNet + Embedding-based)

#### 4.2.1 基于WordNet的扩展

```python
# pip install nltk
from typing import List


class WordNetExpander:
    """基于WordNet的同义词扩展（英文示例）"""

    def __init__(self):
        try:
            from nltk.corpus import wordnet as wn
        except Exception:
            raise RuntimeError("请先安装nltk并下载wordnet语料")
        self.wn = wn

    def expand(self, term: str, topk: int = 3) -> List[str]:
        """返回同义词扩展列表"""
        synonyms = set()
        for synset in self.wn.synsets(term):
            for lemma in synset.lemmas():
                synonyms.add(lemma.name().replace("_", " "))
        # 简单截断
        return list(synonyms)[:topk]


# 使用示例
expander = WordNetExpander()
print(expander.expand("car"))
```

#### 4.2.2 基于Embedding的扩展（简化版）

```python
import math
from typing import Dict, List, Tuple


def cosine(a: List[float], b: List[float]) -> float:
    """计算向量余弦相似度"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingExpander:
    """基于词向量近邻的同义词扩展（演示版）"""

    def __init__(self, embeddings: Dict[str, List[float]]):
        self.embeddings = embeddings

    def expand(self, term: str, topk: int = 3) -> List[Tuple[str, float]]:
        """返回相似词及相似度"""
        if term not in self.embeddings:
            return []
        target = self.embeddings[term]
        scored = []
        for w, vec in self.embeddings.items():
            if w == term:
                continue
            scored.append((w, cosine(target, vec)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:topk]


# 模拟向量
embeddings = {
    "汽车": [0.9, 0.1, 0.0],
    "车辆": [0.85, 0.15, 0.05],
    "轿车": [0.88, 0.12, 0.02],
    "水果": [0.1, 0.9, 0.0],
}

expander = EmbeddingExpander(embeddings)
print(expander.expand("汽车"))
```

#### 4.2.3 基于点击日志的扩展（共现统计）

```python
from collections import Counter, defaultdict
from typing import List, Dict


class ClickLogExpander:
    """基于点击日志的共现扩展（演示版）"""

    def __init__(self, logs: List[List[str]]):
        # logs格式: [ [query, clicked_doc], ... ]
        self.query_to_docs = defaultdict(list)
        for q, d in logs:
            self.query_to_docs[q].append(d)

    def expand(self, query: str, topk: int = 3) -> List[str]:
        """从共现频次中选取扩展词"""
        docs = self.query_to_docs.get(query, [])
        if not docs:
            return []
        counts = Counter()
        for doc in docs:
            for token in doc.split():
                counts[token] += 1
        return [w for w, _ in counts.most_common(topk)]


logs = [
    ["苹果 财报", "apple inc 2023 revenue"],
    ["苹果 财报", "apple earnings report"],
    ["电动车 续航", "battery range anxiety"],
]

expander = ClickLogExpander(logs)
print(expander.expand("苹果 财报"))
```

#### 4.2.4 扩展词融合与权重控制

```python
from typing import Dict, List


def merge_expansions(query: str, expansions: Dict[str, List[str]], weight: float = 0.5) -> str:
    """融合扩展词，使用权重表示"""
    terms = query.split()
    result = []
    for t in terms:
        result.append(t)
        for e in expansions.get(t, []):
            result.append(f"{e}^{weight}")
    return " ".join(result)


exp = {
    "汽车": ["车辆", "轿车"],
    "保养": ["维护", "检修"],
}
print(merge_expansions("汽车 保养", exp, weight=0.7))
```

### 4.3 Query分解与多跳检索 (Decomposition + Multi-hop)

```python
from typing import List, Dict


class QueryDecomposer:
    """查询分解器（规则示例）"""

    def decompose(self, query: str) -> List[str]:
        # 处理对比类问题
        if "和" in query and "比较" in query:
            parts = query.split("和")
            return [f"{p} 优缺点" for p in parts]
        # 处理多实体列举
        if "、" in query:
            parts = query.split("、")
            return parts
        return [query]


class MultiHopRetriever:
    """多跳检索（演示版）"""

    def __init__(self, kb: Dict[str, str]):
        self.kb = kb

    def retrieve(self, query: str) -> str:
        # 简化：直接用查询作为键
        return self.kb.get(query, "")

    def multi_hop(self, query: str) -> List[str]:
        """两跳检索：先找实体，再找实体相关信息"""
        results = []
        first = self.retrieve(query)
        if not first:
            return results
        results.append(first)

        # 用第一跳结果构造第二跳查询
        second_query = f"{first} 创立的公司"
        second = self.retrieve(second_query)
        if second:
            results.append(second)
        return results


# 模拟知识库
kb = {
    "Python 创始人": "Guido van Rossum",
    "Guido van Rossum 创立的公司": "Dropbox",
}

retriever = MultiHopRetriever(kb)
print(retriever.multi_hop("Python 创始人"))
```

#### 4.3.1 基于图的多跳检索（Graph-based）

```python
from typing import Dict, List


class GraphHopRetriever:
    """基于图的多跳检索（演示版）"""

    def __init__(self, graph: Dict[str, List[str]]):
        self.graph = graph

    def hop(self, start: str, hops: int = 2) -> List[str]:
        """从起点进行多跳扩展"""
        frontier = [start]
        visited = set(frontier)
        for _ in range(hops):
            next_frontier = []
            for node in frontier:
                for nei in self.graph.get(node, []):
                    if nei not in visited:
                        visited.add(nei)
                        next_frontier.append(nei)
            frontier = next_frontier
        return list(visited)


graph = {
    "Python": ["Guido", "语言特性"],
    "Guido": ["Dropbox"],
    "Dropbox": ["云存储"],
}

gh = GraphHopRetriever(graph)
print(gh.hop("Python", hops=2))
```

#### 4.3.2 多跳检索中的中间实体选择

```python
from typing import List, Dict


def select_intermediate_entities(candidates: List[str], keywords: List[str]) -> List[str]:
    """选择与关键词重合度高的中间实体（演示版）"""
    selected = []
    for c in candidates:
        score = sum(1 for k in keywords if k in c)
        if score > 0:
            selected.append(c)
    return selected


candidates = ["Guido van Rossum", "Python语法", "Java历史"]
print(select_intermediate_entities(candidates, ["Python", "创始人"]))
```

### 4.4 伪相关反馈 (PRF / Rocchio)

```python
from typing import List


def rocchio_update(query_vec: List[float], doc_vecs: List[List[float]], alpha=1.0, beta=0.75) -> List[float]:
    """Rocchio算法：根据伪相关文档更新查询向量"""
    if not doc_vecs:
        return query_vec
    avg_doc = [0.0] * len(query_vec)
    for vec in doc_vecs:
        for i, v in enumerate(vec):
            avg_doc[i] += v
    avg_doc = [v / len(doc_vecs) for v in avg_doc]

    new_vec = [alpha * q + beta * d for q, d in zip(query_vec, avg_doc)]
    return new_vec


# 示例向量
query_vec = [0.2, 0.5, 0.1]
top_docs = [
    [0.3, 0.4, 0.2],
    [0.25, 0.55, 0.15],
]
print(rocchio_update(query_vec, top_docs))
```

#### 4.4.1 PRF词项扩展（基于Top-K文档词频）

```python
from collections import Counter
from typing import List, Dict


def prf_expand(query: str, top_docs: List[str], topk: int = 5) -> List[str]:
    """基于Top-K文档进行伪相关反馈扩展"""
    counter = Counter()
    for doc in top_docs:
        for token in doc.split():
            counter[token] += 1
    # 排除原查询词
    for t in query.split():
        if t in counter:
            del counter[t]
    return [w for w, _ in counter.most_common(topk)]


query = "电动车 续航"
docs = ["电动车 续航 电池 里程", "续航 里程 充电" , "电池 衰减 续航"]
print(prf_expand(query, docs))
```

### 4.5 HyDE改写 (Hypothetical Document Embedding)

```python
class HyDERewriter:
    """HyDE：生成假设文档再检索"""

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def rewrite(self, query: str) -> str:
        # 生成假设回答
        prompt = f"基于查询：{query}，生成一段理想回答"
        return self.llm.generate(prompt)

    def search(self, query: str):
        # 生成假设文档
        hyp_doc = self.rewrite(query)
        # 用假设文档进行检索
        return self.retriever.retrieve(hyp_doc)


# 模拟组件
class MockLLM:
    def generate(self, prompt: str) -> str:
        return "这是一个假设的答案文本，用于检索。"


class MockRetriever:
    def retrieve(self, text: str):
        return ["Doc1", "Doc2"]


hyde = HyDERewriter(MockLLM(), MockRetriever())
print(hyde.search("什么是机器学习？"))
```

### 4.6 Query改写整体流程 (Pipeline)

```python
from typing import List, Dict


class QueryRewriter:
    """Query改写流水线"""

    def __init__(self, intent_classifier, synonym_expander):
        self.intent_classifier = intent_classifier
        self.synonym_expander = synonym_expander

    def normalize(self, query: str) -> str:
        """规范化处理"""
        return query.strip().lower()

    def rewrite(self, query: str) -> Dict[str, List[str]]:
        """输出原查询 + 扩展查询"""
        query = self.normalize(query)
        intent = self.intent_classifier.predict(query)

        expanded_terms = []
        for term in query.split():
            expanded_terms.append(term)
            expanded_terms.extend(self.synonym_expander.get(term, []))

        return {
            "intent": intent,
            "queries": [" ".join(expanded_terms)],
        }


# 示例
intent_classifier = SimpleIntentClassifier()
synonyms = {"car": ["auto", "vehicle"]}
rewriter = QueryRewriter(intent_classifier, synonyms)
print(rewriter.rewrite("car maintenance"))
```

### 4.7 多路改写候选生成与融合 (Multi-route Candidates)

在实际系统中常生成多个改写候选并并行检索，然后进行融合与重排。此策略能够在**召回覆盖**与**成本**之间折中。

```python
from typing import List, Dict


def generate_candidates(query: str) -> List[str]:
    """生成多路改写候选（演示版）"""
    candidates = [query]
    # 简单规则扩展
    if "电动车" in query:
        candidates.append(query.replace("电动车", "新能源汽车"))
    # 简单同义词
    candidates.append(query + " 续航")
    return list(dict.fromkeys(candidates))


def fuse_results(results: Dict[str, List[str]]) -> List[str]:
    """融合多路检索结果"""
    merged = []
    for q, docs in results.items():
        for d in docs:
            if d not in merged:
                merged.append(d)
    return merged


# 模拟
query = "电动车 里程"
cands = generate_candidates(query)
mock_results = {c: [f"doc_{i}_{c}" for i in range(2)] for c in cands}
print(fuse_results(mock_results))
```

### 4.8 LLM辅助的Query分解与改写

LLM可以用于复杂Query的结构化拆解，常见Prompt模板如下：

```
你是检索助手，请把用户问题拆成可检索的子问题：
问题：{query}
输出格式：
1) 子问题1
2) 子问题2
```

示例输出：

- 输入："比较Python和Java在Web开发中的优缺点"
- 输出：
  1) Python在Web开发中的优缺点
  2) Java在Web开发中的优缺点

### 4.9 稀疏+稠密混合检索 (Hybrid Retrieval)

混合检索可以有效融合稀疏与稠密检索的优势：

```python
from typing import List, Dict


def hybrid_merge(sparse_results: List[str], dense_results: List[str]) -> List[str]:
    """简单的混合融合：交错合并"""
    merged = []
    i = 0
    while i < max(len(sparse_results), len(dense_results)):
        if i < len(sparse_results):
            merged.append(sparse_results[i])
        if i < len(dense_results):
            merged.append(dense_results[i])
        i += 1
    # 去重
    uniq = []
    for d in merged:
        if d not in uniq:
            uniq.append(d)
    return uniq


print(hybrid_merge(["d1", "d2"], ["d2", "d3"]))
```

### 4.10 对话上下文补全 (Conversational Rewriting)

```python
from typing import List, Dict


class ConversationRewriter:
    """对话场景Query改写（简化版）"""

    def __init__(self):
        self.history: List[Dict[str, str]] = []

    def add_turn(self, query: str, response: str):
        """记录对话轮次"""
        self.history.append({"query": query, "response": response})

    def rewrite(self, query: str) -> str:
        """将指代词替换为上一轮主题"""
        if not self.history:
            return query
        last_query = self.history[-1]["query"]
        pronouns = ["它", "他", "她", "这个", "那个"]
        for p in pronouns:
            if p in query:
                return query.replace(p, last_query)
        return query


conv = ConversationRewriter()
conv.add_turn("Python是什么", "一种编程语言")
print(conv.rewrite("它有什么特点"))
```

### 4.11 简单评估脚本 (Offline Evaluation)

```python
from typing import List


def recall_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
    """计算Recall@K"""
    hit = 0
    for doc in retrieved[:k]:
        if doc in relevant:
            hit += 1
    return hit / max(1, len(relevant))


retrieved = ["doc1", "doc2", "doc3"]
relevant = ["doc2", "doc4"]
print(recall_at_k(retrieved, relevant, k=2))
```

---

## 5. 实验对比 (Rewrite Impact on Retrieval)

### 5.1 指标与评估设置

常见指标包括：

- **Recall@K**: Top-K召回比例
- **NDCG@K**: 考虑排序的相关性
- **MRR**: 首条相关结果位置
- **Latency**: 额外改写开销

### 5.2 改写效果对比（示例）

| 改写策略 | Recall@10 | NDCG@10 | MRR | 平均延迟增加 |
|---------|-----------|---------|-----|-------------|
| 无改写 | 0.62 | 0.49 | 0.42 | +0ms |
| 同义词扩展 | 0.68 | 0.53 | 0.45 | +5ms |
| Query分解 | 0.74 | 0.57 | 0.50 | +35ms |
| PRF | 0.72 | 0.55 | 0.48 | +20ms |
| HyDE | 0.80 | 0.63 | 0.56 | +200ms |

**结论**: HyDE效果最好但成本最高；PRF和Query分解是效果与成本的折中选择。

### 5.3 多跳检索对比（示例）

| 查询类型 | 单跳Recall@5 | 多跳Recall@5 | 提升 |
|---------|-------------|-------------|------|
| "人物-公司" | 0.41 | 0.70 | +29% |
| "事件-时间" | 0.50 | 0.68 | +18% |
| "政策-影响" | 0.38 | 0.60 | +22% |

### 5.4 Query理解的贡献（示例）

| 模块 | Recall@10提升 | 说明 |
|------|--------------|-----|
| 意图识别 | +4% | 选择更合适的改写策略 |
| 实体标准化 | +6% | 统一别名、缩写 |
| 上下文补全 | +5% | 减少对话指代缺失 |

### 5.5 实验设计与数据集说明

为了避免"离线指标很好但线上体验差"的问题，实验应同时覆盖离线与线上：

- **离线评估 (Offline)**: 使用标注的检索相关性数据集。
- **线上评估 (Online)**: A/B测试关注点击率、转化率与停留时长。

常见评估步骤：

1. 选择覆盖高频与长尾查询的测试集合。
2. 标注Query与文档的相关等级。
3. 比较改写策略前后的NDCG/Recall。
4. 在线上灰度发布，监控指标变化。

### 5.6 消融实验 (Ablation Study)

消融实验用于评估各模块贡献：

| 模块 | Recall@10 | NDCG@10 | 说明 |
|------|-----------|---------|------|
| 基线 | 0.62 | 0.49 | 无改写 |
| + 同义词扩展 | 0.68 | 0.53 | 词汇覆盖提升 |
| + 意图识别 | 0.71 | 0.55 | 策略选择更合理 |
| + PRF | 0.72 | 0.55 | 噪声可控 |
| + HyDE | 0.80 | 0.63 | 成本最高 |

### 5.7 成本与延迟分析 (Cost & Latency)

改写策略需平衡成本与效果，示例分析：

| 策略 | 平均Token消耗 | 平均延迟 | 单次成本 | 适用场景 |
|------|-------------|---------|---------|---------|
| 规则改写 | 0 | <1ms | 低 | 高频查询 |
| PRF | 0 | 10-30ms | 低 | 结果稳定的检索 |
| HyDE | 200-800 | 100-300ms | 高 | 高价值复杂查询 |

### 5.8 误差分析 (Error Analysis)

在评估后应对失败案例进行分析：

- **过度扩展**: 同义词引入错误语义。
- **扩展不足**: 召回不提升，说明扩展词覆盖不足。
- **多跳失效**: 中间实体错误导致后续检索失败。
- **意图误判**: 错误的策略选择导致整体性能下降。

建议将失败案例分类并回溯改写链路，定位问题模块。

### 5.9 业务指标与用户体验

除了离线检索指标，还需要关注用户体验指标：

- **点击率 (CTR)**: Query改写是否带来更高点击。
- **停留时长**: 用户是否更快找到目标信息。
- **重复查询率**: 反映用户是否满意结果。
- **投诉与反馈**: 改写是否引发错误理解。

在业务指标与检索指标出现冲突时，应优先保证**用户体验**，再逐步优化检索模型。

---

## 6. 最佳实践与常见陷阱

### 6.1 最佳实践

1. **策略分层**: 80%查询使用轻量规则改写，20%复杂查询使用LLM改写。
2. **意图驱动**: 先做Query理解，再选择改写策略。
3. **多跳缓存**: 对多跳中间实体建立缓存，减少重复检索。
4. **PRF谨慎使用**: 低质量检索会导致噪声扩展。
5. **A/B测试**: 在真实流量中验证效果，不依赖离线指标。

### 6.2 常见陷阱

1. **过度扩展 (Over-expansion)**: 引入过多同义词导致噪声。
2. **意图误判 (Intent Drift)**: 错误改写导致完全偏离需求。
3. **成本失控 (Cost Explosion)**: 对所有查询使用HyDE或LLM改写。
4. **多跳失败 (Multi-hop Failure)**: 中间实体错误会传递到后续跳。
5. **PRF噪声 (PRF Noise)**: Top-K文档不相关时会污染查询。

### 6.3 决策流程建议

```
查询进入
  ↓
Query理解 (Intent + Entity)
  ↓
简单查询? → [是] → 同义词扩展 → 检索
  ↓ [否]
复杂查询? → [是] → Query分解 → 检索
  ↓ [否]
多跳查询? → [是] → Multi-hop → 检索
  ↓ [否]
歧义查询? → [是] → 意图识别 + 多路检索
  ↓ [否]
HyDE改写 → 检索
```

### 6.4 线上监控与回退机制

建议在生产环境中建立指标监控与快速回退：

- **监控指标**: Recall proxy、点击率、失败率、LLM调用耗时。
- **回退机制**: 改写失败或延迟过高时，回退原始Query。
- **熔断策略**: LLM接口异常时自动关闭高成本改写。

### 6.5 常见策略组合建议

- **高频查询**: 规则改写 + 同义词扩展。
- **复杂查询**: Query分解 + 多跳检索。
- **长尾查询**: HyDE + PRF（必要时）。
- **对话查询**: 上下文补全 + 意图识别。

### 6.6 安全与合规 (Safety & Compliance)

Query改写可能引入安全与合规风险：

- **隐私泄露**: 不应扩展出敏感信息。
- **偏见放大**: 扩展词可能引入偏见语义。
- **提示注入 (Prompt Injection)**: LLM改写需过滤恶意输入。

实践建议：

1. 对改写输出进行敏感词过滤。
2. 对LLM调用增加安全提示与过滤层。
3. 记录与审计改写结果，便于追踪问题。

### 6.7 Debug与迭代优化

调试Query改写时建议建立"改写链路日志"以追踪每一步：

```
原始Query
  -> 规范化Query
  -> 意图识别
  -> 扩展词列表
  -> 子查询列表
  -> 最终检索Query
```

通过可视化日志与问题归因，逐步优化策略与权重。

### 6.8 上线检查清单 (Checklist)

- 规则改写与LLM改写均有回退策略。
- 对扩展词与改写候选有最大数量限制。
- 监控HyDE与PRF的失败率与延迟。
- 线上A/B实验具备可回滚能力。
- 改写日志可追溯，支持问题定位。

---

## 7. 总结

Query改写与扩展是RAG和搜索系统的关键能力。通过**Query理解**、**同义词扩展**、**Query分解**、**多跳检索**、**PRF**与**HyDE**等手段，可以显著提升检索系统的召回与排序质量。

**核心结论**:

- Query理解是改写的前提，意图识别决定策略选择。
- 同义词扩展成本低但效果有限，适合高频查询。
- Query分解和多跳检索适合复杂问题。
- PRF可以在无标注情况下自动扩展，但需要控制噪声。
- HyDE效果最好但成本高，适合高价值查询。

**推荐配置**:

- 通用场景: 同义词扩展 + 规则改写
- 复杂场景: Query分解 + 多跳检索
- 高价值场景: HyDE + PRF

通过策略分层与动态选择，可以在成本与效果之间取得最佳平衡。

在实践中，应持续更新同义词库、点击日志与实体别名库，并基于线上反馈调整改写权重。对于高价值查询，可以引入人工或半自动标注，以提升改写质量与评估可靠性。将改写策略作为可配置组件，可以快速适配业务变化并降低维护成本。
