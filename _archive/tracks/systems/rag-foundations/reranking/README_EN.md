# Rerank策略 (Rerank Strategies)

本文件系统性梳理 Rerank (重排序) 在检索增强生成 (RAG) 与搜索系统中的角色、原理与工程落地。内容按“背景 → 核心概念 → 数学原理 → 代码实现 → 实验对比 → 最佳实践与常见陷阱 → 总结”组织，覆盖两阶段检索 (two-stage retrieval)、交叉编码器 (cross-encoder)、ColBERT (late interaction)、常见 reranker (bge-reranker, Cohere Rerank) 以及性能权衡与延迟分析。

## 目录

1. [背景 (Why Rerank?)](#1-背景-why-rerank)
2. [核心概念 (Two-stage Retrieval, Cross-Encoder)](#2-核心概念-two-stage-retrieval-cross-encoder)
3. [数学原理 (Scoring Formulas, ColBERT)](#3-数学原理-scoring-formulas-colbert)
4. [代码实现 (Rerank Implementation)](#4-代码实现-rerank-implementation)
5. [实验对比 (Rerank Impact on Metrics)](#5-实验对比-rerank-impact-on-metrics)
6. [最佳实践与常见陷阱](#6-最佳实践与常见陷阱)
7. [总结](#7-总结)

---

## 1. 背景 (Why Rerank?)

### 1.1 为什么需要 Rerank (重排序)？

在大多数检索系统中，第一阶段检索 (first-stage retrieval) 关注“速度与覆盖”，典型模型包括 BM25、向量检索 (dense retrieval)、混合检索 (hybrid retrieval)。这些模型的目标是“快速从海量语料中找出 Top-K 候选”，而不是“精确给出最终排序”。因此第一阶段常出现以下问题：

- **召回足够但排序不稳**：相近语义文档可能被排在 Top-K 末端，导致后续 RAG 生成质量下降。
- **弱交互建模**：双编码器 (bi-encoder) 仅计算向量相似度，难以捕捉精细语义对齐、否定语义与逻辑关系。
- **任务偏移**：第一阶段模型往往是“通用检索”，对业务语境、实体类型、领域术语敏感度不足。

Rerank (重排序) 通过更强的模型对候选文档重新评分，使得最终 Top-N 更精确，从而显著提升阅读体验与 LLM 输出质量。

### 1.2 Rerank 的价值与场景

- **RAG 场景**：减少“幻觉” (hallucination) 的关键是输入质量，Rerank 能提升关键证据的排序。
- **问答检索**：对答案关键性很高的任务，Top-3 文档质量影响远大于 Top-100。
- **多源检索融合**：Rerank 可以融合 BM25 + Dense + Graph 等多路召回结果。
- **企业知识库**：对长文档和结构化文档，Rerank 可显著提高相关性判断能力。

### 1.3 两阶段策略的直观解释

“先粗后精”的检索逻辑是工业级检索系统的标准范式：

```
查询 → 第一阶段检索 (Top-100/200) → Rerank (Top-10) → 下游使用
         (快、粗、广)              (慢、精、准)
```

这种架构使系统能在大规模语料中保持可接受的延迟，同时又能保证最终结果质量。

### 1.4 Rerank 在 RAG 中的位置

在 RAG (Retrieval-Augmented Generation) 中，Rerank 决定“进入上下文窗口的文档质量”。一个典型流程如下：

1. **Query Understanding**：对用户问题进行轻量重写 (query rewriting)，提升召回质量。
2. **Candidate Retrieval**：使用稀疏检索或向量检索取候选集。
3. **Rerank**：对候选集进行精排，提升 Top-N 文档质量。
4. **Context Packing**：根据 token budget 选取最相关文档。
5. **LLM Generation**：基于高质量文档生成答案。

如果 Rerank 缺失，LLM 可能收到“相关但不关键”的信息，导致回答偏离核心问题。

### 1.5 质量提升的具体体现

Rerank 带来的提升往往表现在：

- **更高的事实一致性**：检索到的证据更贴合问题细节。
- **更少的冗余信息**：排序更合理，减少相似或重复文档进入上下文。
- **更低的偏差**：通过交叉编码器捕捉否定关系与实体约束。

### 1.6 成本与延迟的现实约束

Rerank 的模型复杂度通常高于第一阶段检索，因此需要衡量：

- **GPU 推理成本**：Cross-Encoder 在大模型下会显著增加成本。
- **API 成本**：Cohere Rerank 依赖外部服务，调用次数越多成本越高。
- **延迟与体验**：高延迟会降低用户体验，尤其是交互式系统。

在工程实践中，推荐通过候选规模控制 + 动态 cutoff 来平衡效果与成本。

### 1.7 行业应用场景

Rerank 在不同领域表现差异明显：

- **电商搜索**：商品描述冗长、同义词多，Rerank 可突出关键属性匹配。
- **法律检索**：对条款细节敏感，Cross-Encoder 在逻辑与否定关系上优势明显。
- **医疗检索**：实体一致性要求高，Rerank 可减少错误文档进入 Top-K。
- **企业知识库**：对内部术语敏感，需结合领域微调的 reranker。

### 1.8 多轮对话中的 Rerank

在多轮对话中，用户问题往往依赖上下文，Rerank 可通过引入对话历史 (dialog context) 提升相关性：

- 将历史问题拼接到当前 query
- 对不同轮次设置不同权重
- 在 rerank 阶段引入意图分类结果

这种策略能够显著改善对话式 RAG 的检索质量。

### 1.9 Rerank vs Query Rewrite

Query Rewrite (查询改写) 关注“召回覆盖”，Rerank 关注“排序质量”。两者的组合方式通常是：先改写 query 提高候选质量，再用 reranker 精排。实践中若只做改写而不 rerank，往往会出现“候选更广但排序不稳”的问题。

### 1.10 Rerank 与上下文预算

在 LLM 上下文长度有限的情况下，Rerank 的作用不仅是排序，更是“内容筛选”。当上下文只能容纳 3~5 篇文档时，Rerank 对最终答案质量的影响往往比候选规模更关键。
在此场景下，建议对高分文档做去重与多样性控制。

---

## 2. 核心概念 (Two-stage Retrieval, Cross-Encoder)

### 2.1 两阶段检索 (Two-stage Retrieval)

两阶段检索强调“召回”与“排序”分离：

1. **第一阶段召回**：目标是“覆盖率”，通常使用稀疏检索 (BM25) 或向量检索 (ANN)。
2. **第二阶段重排序**：目标是“精度”，使用更复杂的模型进行重新评分。

两阶段策略的关键是“候选集大小 (candidate size)”与“Rerank 截断 (cutoff)”的选择：

- **candidate size**：第一阶段返回多少条候选，常见范围 50-200。
- **rerank cutoff**：Rerank 最终保留多少条用于下游，常见范围 5-20。

### 2.2 Rerank 模型类别

#### 2.2.1 交叉编码器 (Cross-Encoder)

Cross-encoder 直接把 Query 与 Document 拼接，输入同一个 Transformer，最终得到一个相关性分数。

**优点**：
- 充分建模 token 级别交互
- 对否定、数量比较、实体关系识别效果好

**缺点**：
- 无法预计算文档向量
- 复杂度高，只适合少量候选 (Top-K)

#### 2.2.2 双编码器 (Bi-Encoder)

Query 与 Document 分开编码，使用向量相似度衡量相关性，适合第一阶段检索。

**优点**：可提前索引、适合大规模 ANN
**缺点**：交互弱，相关性判断粗糙

#### 2.2.3 ColBERT (Late Interaction)

ColBERT 介于 Cross-Encoder 与 Bi-Encoder 之间。它对 Query 和 Document 分别编码，但通过 **Late Interaction** 在 token 级别进行匹配，使用 MaxSim 聚合。

**优点**：
- 文档向量可预计算
- 交互强度高于 bi-encoder

**缺点**：
- 需要存储更多 token 级向量
- 推理复杂度高于纯向量检索

#### 2.2.4 LLM-based Reranker

使用 LLM (大语言模型) 直接进行相关性判断或 Pairwise 比较，效果非常强，但成本高、延迟大。

### 2.3 常见 Rerank 模型与算法

| 模型/算法 | 类型 | 特点 | 适用场景 |
|---|---|---|---|
| **bge-reranker** | Cross-Encoder | 开源、强基线 | 通用 RAG |
| **Cohere Rerank** | API | 易用、稳定 | 快速上线 |
| **Cross-Encoder (MS MARCO)** | Cross-Encoder | 成熟基线 | QA/搜索 |
| **ColBERT/ColBERTv2** | Late Interaction | 大规模高效 | 文档检索 |
| **MonoT5** | Seq2Seq | 文本重写式评分 | 研究/实验 |
| **RankGPT/RankLLaMA** | LLM-based | 强推理 | 高价值场景 |

### 2.4 Rerank 截断 (Cutoff) 与性能权衡

Rerank 的成本主要来自对候选的交叉编码器评分，因此要控制 rerank cutoff：

- **cutoff 太小**：可能错过关键证据
- **cutoff 太大**：延迟显著增加，收益递减

实践中通常采用“Top-100 召回 → Top-10 rerank → Top-5 使用”的结构，并在业务场景中调优。

### 2.5 Cross-Encoder 结构细节

Cross-Encoder 的典型结构：

1. **输入拼接**：`[CLS] query [SEP] document [SEP]`
2. **Transformer 编码**：多层 self-attention 获取 token 级交互
3. **池化**：使用 `[CLS]` 或 mean pooling
4. **评分头**：线性层或 MLP 输出相关性分数

这种结构能够捕捉 query-doc 之间的复杂对齐关系，例如：

- 逻辑关系 (因果、否定、条件)
- 数值比较 (大于、小于、范围)
- 实体约束 (人名、组织、时间)

此外，cross-encoder 常见的评分头包括：

- **单层线性头**：简单稳定，适合基线
- **MLP 头**：能提升表达能力，但易过拟合
- **多任务头**：同时预测相关性与匹配类型

### 2.6 Token 与段落级别的交互

对长文档而言，token 级交互可能过于昂贵，工程上常使用：

- **段落级分块 rerank**：先对段落评分，再聚合到文档
- **标题优先**：标题 + 摘要先评分，再决定是否读取正文

这种策略在大文档场景中能显著降低延迟。

### 2.6 Rerank 与检索融合策略

Rerank 并不一定完全替代第一阶段排序，常见的融合方式包括：

- **Rerank Only**：直接用 rerank 分数重新排序
- **Score Fusion**：结合 first-stage 分数与 rerank 分数
- **Two-pass**：先用轻量 rerank 粗排，再用重型 rerank 精排

融合方式的目标是兼顾质量与成本。

### 2.7 训练数据与负样本构造

高质量 reranker 依赖高质量训练数据，常用策略包括：

- **硬负样本 (hard negatives)**：从第一阶段检索中取高相似但错误的文档
- **随机负样本 (random negatives)**：提供背景区分能力
- **蒸馏数据 (distillation)**：使用 LLM 打分生成伪标签

对 reranker 的训练质量影响极大，尤其在领域检索中，硬负样本决定模型边界。

### 2.8.1 数据构建流程示意

一个常见的训练数据流程：

1. **采样真实查询**：来自日志或任务定义
2. **召回候选集**：通过 BM25 或 Dense 检索
3. **标注相关性**：人工标注或 LLM 生成标签
4. **构造正负样本**：正样本为高相关文档，负样本包括 hard negatives
5. **训练 reranker**：使用 pairwise/listwise 目标

此流程可持续迭代，逐步提高 rerank 质量。

### 2.8 Rerank 与排序指标的关系

Rerank 的提升通常体现在排序指标：

- **MRR@K**：衡量第一个相关文档的排序位置
- **NDCG@K**：考虑文档位置与相关性权重
- **MAP@K**：平均精度，适合多相关文档

在业务上，Rerank 对“首屏体验”与“答案质量”影响显著。

### 2.9 Rerank 选型矩阵 (Model Selection Matrix)

| 需求 | 推荐模型 | 理由 |
|---|---|---|
| 低延迟 | MiniLM Cross-Encoder | 模型轻量，吞吐高 |
| 强效果 | bge-reranker-large | 开源基线强 |
| 多语言 | Cohere Rerank | 多语言支持好 |
| 大规模索引 | ColBERTv2 | 可预计算，分布式友好 |
| 高价值场景 | LLM-based Rerank | 理解力强 |

在选型时需考虑：候选规模、延迟预算、GPU/CPU 资源、成本上限、工程复杂度。

### 2.10 Rerank 与 Prompt-Rerank 的区别

Prompt-Rerank 指使用 LLM 直接通过 prompt 评分或 pairwise 比较：

- **优点**：无需训练，适合小样本领域
- **缺点**：成本高、延迟大、可复现性差

相比之下，专用 reranker 更稳定、成本可控。

### 2.11 多阶段 Rerank (Multi-stage Rerank)

多阶段 rerank 常用于高精度系统：

1. **Stage-1**：轻量 cross-encoder
2. **Stage-2**：强模型 reranker
3. **Stage-3**：LLM-based 校正

这种架构能在保证延迟的同时，将关键结果排在最前。

### 2.12 与向量检索的协同优化

Rerank 与向量检索的协同策略包括：

- **向量召回扩展**：先扩展候选集，再 rerank
- **语义缓存 (semantic cache)**：缓存热门 query 的 rerank 结果
- **分桶 rerank**：按领域或意图分组，采用不同 reranker

### 2.12.1 召回质量与 rerank 上限

Rerank 的上限受第一阶段召回质量限制。若候选集内相关文档比例过低，即使 reranker 很强也难以排序出优质 Top-N。工程上常通过：

- 扩展候选集
- 增加多路检索器
- 优化 query rewriting

来提升 rerank 的“可用空间”。

### 2.13 分数融合与阈值策略

当需要融合多个分数时，可采用阈值策略：

- **硬阈值**：低于阈值直接丢弃
- **软阈值**：低分文档降低权重而非丢弃

这类策略在高噪声数据场景中尤其有效。

### 2.14 多样性约束 (Diversity-aware Rerank)

排序质量不仅是相关性，还需要多样性。常见做法：

- 结合 MMR (Maximal Marginal Relevance)
- 引入主题或来源惩罚
- 控制 Top-N 中的重复文档

这对“问答 + 多文档生成”尤其关键。

### 2.15 延迟预算拆分 (Latency Budget)

在在线系统中可将整体延迟预算拆分：

| 阶段 | 预算 | 说明 |
|---|---|---|
| Query 处理 | 5-10ms | 改写/意图识别 |
| 召回 | 20-40ms | BM25/ANN |
| Rerank | 40-80ms | Cross-Encoder |
| 后处理 | 5-10ms | 去重、多样性 |

通过明确预算，可以指导 rerank cutoff 与模型选择。

---

## 3. 数学原理 (Scoring Formulas, ColBERT)

### 3.1 交叉编码器评分 (Cross-Encoder Scoring)

交叉编码器对拼接输入 $(q, d)$ 进行编码，最终通过线性层输出相关性分数：

$$
\text{Score}(q, d) = \text{MLP}([\text{Transformer}(q \oplus d)])
$$

其中 $\oplus$ 表示拼接，`[CLS]` 或池化向量作为输出。

### 3.2 Pairwise / Listwise 学习

在训练阶段常用 Pairwise 或 Listwise 目标：

- **Pairwise 损失**：

$$
\mathcal{L}_{\text{pair}} = -\log \sigma(s(q, d^+) - s(q, d^-))
$$

- **Listwise 损失 (Softmax)**：

$$
P(d_i|q) = \frac{\exp(s(q, d_i))}{\sum_j \exp(s(q, d_j))}
$$

### 3.3 ColBERT Late Interaction

ColBERT 将 Query 与 Document 的 token 表示分别编码，然后用 MaxSim 进行匹配：

$$
\text{Score}(q, d) = \sum_{i=1}^{|q|} \max_{j \in |d|} \mathbf{q}_i^\top \mathbf{d}_j
$$

其中 $\mathbf{q}_i$ 为第 $i$ 个 query token 向量，$\mathbf{d}_j$ 为第 $j$ 个 document token 向量。

### 3.4 融合与重排序公式

多路召回融合 (Reciprocal Rank Fusion, RRF) 常用于候选集融合：

$$
\text{RRF}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}
$$

其中 $r_m(d)$ 是文档 $d$ 在检索器 $m$ 的排名，$k$ 为平滑常数。

混合评分融合：

$$
\text{Score}_{\text{fusion}} = w_1 \cdot \text{Score}_{\text{sparse}} + w_2 \cdot \text{Score}_{\text{dense}} + w_3 \cdot \text{Score}_{\text{rerank}}
$$

### 3.5 Rerank 截断带来的数学影响

设第一阶段候选为 $K$，最终保留 $N$，则 rerank 的期望增益可以近似理解为：

$$
\Delta \text{NDCG} \approx f(K) - f(K-N)
$$

当 $K$ 增大时 $f(K)$ 增益递减，因此“适度候选 + 强排序”通常最优。

### 3.6 NDCG 与 MRR 的数学表达

NDCG (Normalized Discounted Cumulative Gain) 公式：

$$
\text{DCG@K} = \sum_{i=1}^K \frac{2^{rel_i}-1}{\log_2(i+1)}
$$

$$
\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
$$

MRR (Mean Reciprocal Rank) 公式：

$$
\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{rank_q}
$$

这些指标更能体现 Rerank 对“排序质量”的真实贡献。

### 3.7 分数校准 (Score Calibration)

不同 reranker 的分数范围可能不同，因此需要校准：

$$
\hat{s} = \sigma(a \cdot s + b)
$$

其中 $a, b$ 可通过验证集拟合。校准后可与第一阶段分数做融合。

---

### 3.8 复杂度分析 (Latency & Cost)

设候选数 $K$，模型推理时间为 $T$，则整体 rerank 延迟近似为：

$$
\text{Latency} \approx K \times T
$$

在 GPU 上可以通过 batch 并行降低延迟，但仍需控制 $K$。

### 3.9 ColBERT 索引与存储开销

ColBERT 需要存储 token 级向量，存储开销近似为：

$$
\text{Storage} \approx |D| \times L_d \times h
$$

其中 $|D|$ 是文档数，$L_d$ 是文档平均 token 数，$h$ 是向量维度。

这意味着 ColBERT 在大规模语料中需要更多存储，但在质量与效率上表现出色。

### 3.10 Rank Fusion 与重排序的数学组合

当多个检索器生成候选时，可以使用融合算法后再 rerank：

$$
\text{Score}_{\text{final}}(d) = \alpha \cdot \text{RRF}(d) + (1-\alpha) \cdot \text{Score}_{\text{rerank}}(d)
$$

融合可以减少单一检索器偏差。

### 3.11 学习排序损失 (Learning to Rank)

除 Pairwise 外，常见 listwise 损失包括：

$$
\mathcal{L}_{\text{list}} = -\sum_{i=1}^{n} y_i \log \hat{y}_i
$$

其中 $y_i$ 为真实相关性分布，$\hat{y}_i$ 为预测分布。

### 3.15 相关性等级 (Graded Relevance)

许多评测数据集使用 0/1/2/3 等级相关性标签，reranker 训练时可用以下转换：

$$
y_i = \frac{2^{rel_i}-1}{\sum_j (2^{rel_j}-1)}
$$

该方法能在 listwise 训练中体现“高相关性文档更重要”。

### 3.16 评分稳定性与漂移

在真实系统中，rerank 分数可能因模型更新或数据分布变化而漂移。建议：

- 固定评估集，周期性回归测试
- 使用 calibration 保持分数分布稳定
- 对分数变化设置告警阈值

### 3.17 分数归一化 (Normalization)

在融合场景中可使用 z-score 归一化：

$$
\hat{s} = \frac{s - \mu}{\sigma}
$$

其中 $\mu$ 与 $\sigma$ 为分数均值与标准差。归一化后不同模型输出可更稳定融合。

### 3.12 温度与分数缩放 (Temperature Scaling)

在融合场景中可对分数进行温度缩放：

$$
\tilde{s} = \frac{s}{T}
$$

温度 $T$ 越大，分数分布越平滑，有助于融合稳定性。

### 3.13 MMR (Maximal Marginal Relevance) 公式

MMR 兼顾相关性与多样性：

$$
\text{MMR}(d_i) = \lambda \cdot \text{Rel}(d_i) - (1-\lambda) \cdot \max_{d_j \in S} \text{Sim}(d_i, d_j)
$$

其中 $S$ 是已选择文档集合。MMR 常与 rerank 结合，确保 Top-N 不重复。

### 3.14 ColBERT 的向量归一化

ColBERT 通常对 token 向量做 L2 归一化，使得相似度计算更稳定：

$$
\mathbf{q}_i \leftarrow \frac{\mathbf{q}_i}{\|\mathbf{q}_i\|}, \quad \mathbf{d}_j \leftarrow \frac{\mathbf{d}_j}{\|\mathbf{d}_j\|}
$$

这一处理降低了向量尺度差异带来的评分偏差。

### 3.18 相关性阈值与覆盖率

假设设定相关性阈值 $\tau$，只有当 $s(q,d) > \tau$ 的文档才能进入 Top-N。这样做能提升 precision，但可能降低覆盖率。一般通过验证集调优 $\tau$，选择 precision 与 recall 的折中点。

### 3.19 分数裁剪与平滑 (Clipping & Smoothing)

为了减少极端分数对排序的影响，可对分数进行裁剪或平滑：

$$
s' = \min(\max(s, l), u)
$$

其中 $l$ 与 $u$ 为下界与上界。对于极端噪声数据，裁剪可提升排序稳定性。

---

## 4. 代码实现 (Rerank Implementation)

### 4.1 基础 Rerank 框架

```python
from typing import List, Tuple

class SimpleReranker:
    """简化版 Reranker"""

    def __init__(self, model):
        self.model = model

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """对文档重新排序"""
        scored = []
        for doc in documents:
            # 计算相关性分数
            score = self.model.score(query, doc)
            scored.append((doc, score))

        # 按分数降序排序
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
```

### 4.2 使用 Sentence-Transformers Cross-Encoder

```python
from sentence_transformers import CrossEncoder

# 加载 cross-encoder 模型
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "什么是对比学习？"
docs = [
    "对比学习通过正负样本学习表示",
    "深度学习是一种机器学习方法",
    "图神经网络适合图结构数据"
]

# 构建 pair 输入
pairs = [[query, d] for d in docs]
scores = model.predict(pairs)

# 结果排序
ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
for doc, score in ranked:
    print(f"分数: {score:.4f} 文档: {doc}")
```

### 4.3 BGE Reranker (bge-reranker)

```python
from FlagEmbedding import FlagReranker

# 加载 BGE reranker 模型
reranker = FlagReranker("BAAI/bge-reranker-large")

query = "Transformer 的核心优势是什么？"
pairs = [
    [query, "Transformer 支持并行计算和全局依赖建模"],
    [query, "卷积网络擅长处理图像"],
    [query, "自注意力能捕捉长距离依赖"]
]

# 计算分数
scores = reranker.compute_score(pairs)
ranked = sorted(zip(pairs, scores), key=lambda x: x[1], reverse=True)
for (q, d), s in ranked:
    print(f"分数: {s:.4f} 文档: {d}")
```

### 4.4 Cohere Rerank (Cohere API)

```python
import os
import cohere

# 使用环境变量存放 API Key
api_key = os.getenv("COHERE_API_KEY")
client = cohere.Client(api_key)

query = "什么是知识蒸馏？"
docs = [
    "知识蒸馏通过教师模型指导学生模型",
    "梯度下降用于优化模型参数",
    "强化学习依赖奖励信号"
]

response = client.rerank(
    model="rerank-multilingual-v2.0",
    query=query,
    documents=docs,
    top_n=2
)

for r in response.results:
    print(f"排序: {r.index} 分数: {r.relevance_score:.4f} 文档: {docs[r.index]}")
```

### 4.5 ColBERT Late Interaction

```python
from colbert import Indexer, Searcher

# 建立索引（示例）
# indexer = Indexer(checkpoint="colbert-ir/colbertv2.0")
# indexer.index(name="my_index", collection=["文档1", "文档2"])

# 检索与重排序
# searcher = Searcher(index="my_index", checkpoint="colbert-ir/colbertv2.0")
# results = searcher.search("查询", k=10)
# for docid, score in results:
#     print(docid, score)
```

### 4.6 两阶段检索完整流程 (Retrieve → Rerank)

```python
class TwoStageRetriever:
    """两阶段检索: 第一阶段召回 + 第二阶段 rerank"""

    def __init__(self, retriever, reranker, first_k: int = 100, final_k: int = 5):
        self.retriever = retriever
        self.reranker = reranker
        self.first_k = first_k
        self.final_k = final_k

    def retrieve(self, query: str):
        # 1) 第一阶段召回
        candidates = self.retriever.retrieve(query, k=self.first_k)

        # 2) 第二阶段 rerank
        reranked = self.reranker.rerank(query, candidates, top_k=self.final_k)

        return reranked
```

### 4.7 Rerank 截断与性能控制

```python
def dynamic_cutoff(query: str) -> int:
    """根据查询长度动态调整 rerank cutoff"""
    # 规则: 长查询使用更大 cutoff
    if len(query) > 20:
        return 20
    return 10

def rerank_with_cutoff(reranker, query: str, docs: list):
    cutoff = dynamic_cutoff(query)
    # 只对 Top-cutoff 做 rerank
    return reranker.rerank(query, docs[:cutoff], top_k=5)
```

### 4.8 缓存与批处理 (Batching)

```python
from functools import lru_cache

@lru_cache(maxsize=1024)
def cached_rerank(query: str, doc: str, model):
    """缓存单个 query-doc 评分"""
    return model.score(query, doc)

def batch_rerank(model, query: str, docs: list):
    """批量评分提升吞吐"""
    pairs = [[query, d] for d in docs]
    scores = model.predict(pairs)
    return sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

### 4.9 文档分块与 Rerank

```python
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50):
    """将长文档切分成块，避免超长输入被截断"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# 示例：将长文档切分后再参与 rerank
long_doc = "这里是很长的文档内容" * 50
chunks = chunk_text(long_doc)
```

### 4.10 实际工程中的重排序接口设计

```python
class RerankService:
    """工程化 Rerank 服务示例"""

    def __init__(self, reranker, max_candidates=100, timeout_ms=100):
        self.reranker = reranker
        self.max_candidates = max_candidates
        self.timeout_ms = timeout_ms

    def rerank(self, query: str, candidates: list, top_k: int = 10):
        # 1) 控制候选规模
        candidates = candidates[: self.max_candidates]

        # 2) 执行 rerank
        results = self.reranker.rerank(query, candidates, top_k=top_k)

        # 3) 超时控制（示例）
        # 在真实系统中可以结合异步/超时机制
        return results

### 4.11 结合 BM25 + Dense + Rerank 的完整示例

```python
def hybrid_retrieve(query, bm25, dense, k=100):
    """融合 BM25 与 Dense 的候选集"""
    bm25_docs = bm25.retrieve(query, k=k)
    dense_docs = dense.retrieve(query, k=k)
    # 简单去重合并
    candidates = list(dict.fromkeys(bm25_docs + dense_docs))
    return candidates

def full_pipeline(query, bm25, dense, reranker):
    candidates = hybrid_retrieve(query, bm25, dense, k=100)
    # 只对候选集 rerank
    return reranker.rerank(query, candidates, top_k=10)
```

### 4.12 计算 MRR 与 NDCG 的评估代码

```python
import math

def mrr(ranks):
    """计算 MRR，ranks 为相关文档排名列表"""
    return sum(1.0 / r for r in ranks) / len(ranks)

def ndcg(rels, k=10):
    """计算 NDCG，rels 为相关性列表"""
    def dcg(scores):
        return sum((2 ** s - 1) / math.log2(i + 2) for i, s in enumerate(scores))
    ideal = sorted(rels, reverse=True)
    return dcg(rels[:k]) / (dcg(ideal[:k]) + 1e-9)

### 4.13 候选集评估示例

```python
def evaluate_rerank(ground_truth, ranked_docs):
    """ground_truth: {query: set(relevant_docs)}"""
    ranks = []
    ndcgs = []
    for query, docs in ranked_docs.items():
        # 取出该 query 的相关文档集合
        rel_set = ground_truth.get(query, set())
        # 计算 rank
        rank = None
        rels = []
        for i, doc in enumerate(docs, start=1):
            # 标注相关性
            rel = 1 if doc in rel_set else 0
            rels.append(rel)
            if rel == 1 and rank is None:
                rank = i
        if rank:
            ranks.append(rank)
        ndcgs.append(ndcg(rels))
    return mrr(ranks), sum(ndcgs) / len(ndcgs)
```

### 4.14 可复现的实验配置模板

```yaml
rerank:
  model: "bge-reranker-large"
  candidate_size: 100
  cutoff: 10
  batch_size: 16
  timeout_ms: 100
  device: "cuda"
evaluation:
  metrics: ["MRR@10", "NDCG@10", "Recall@100"]
  dataset: "msmarco"
```
```

### 4.13 LLM-based Pairwise Rerank (示意)

```python
def llm_pairwise_rank(llm, query, doc_a, doc_b):
    """LLM 进行 pairwise 比较，返回更相关的文档"""
    prompt = f"问题: {query}\n文档A: {doc_a}\n文档B: {doc_b}\n哪个更相关？"
    # 这里仅示例，具体调用依赖 LLM API
    answer = llm(prompt)
    return "A" if "A" in answer else "B"
```

### 4.14 分批 rerank 控制延迟

```python
def batched_rerank(model, query, docs, batch_size=16):
    """分批进行 rerank，避免单次输入过大"""
    results = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        pairs = [[query, d] for d in batch]
        scores = model.predict(pairs)
        results.extend(zip(batch, scores))
    return sorted(results, key=lambda x: x[1], reverse=True)
```

### 4.15 引入 MMR 的 rerank 示例

```python
def mmr_rerank(scored_docs, lambda_weight=0.7):
    """基于 MMR 的简单 rerank"""
    selected = []
    while scored_docs:
        best = None
        best_score = -1e9
        for doc, rel in scored_docs:
            diversity = 0.0
            for d, _ in selected:
                diversity = max(diversity, similarity(doc, d))
            score = lambda_weight * rel - (1 - lambda_weight) * diversity
            if score > best_score:
                best_score = score
                best = (doc, rel)
        selected.append(best)
        scored_docs.remove(best)
    return selected
```

### 4.16 多语言 rerank 示例

```python
query = "What is contrastive learning?"
docs = [
    "Contrastive learning trains models by comparing positive and negative pairs.",
    "Reinforcement learning uses rewards to learn policies.",
    "Transformers rely on attention mechanisms."
]

# 使用多语言 reranker (示例)
reranker = FlagReranker("BAAI/bge-reranker-v2-m3")
scores = reranker.compute_score([[query, d] for d in docs])
ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
for d, s in ranked:
    print(f"分数: {s:.4f} 文档: {d}")
```

### 4.17 生产级 rerank 的超时与降级示例

```python
import time

class SafeRerankWrapper:
    """带超时与降级的 rerank 包装器"""

    def __init__(self, reranker, fallback=None, timeout_ms=100):
        self.reranker = reranker
        self.fallback = fallback
        self.timeout_ms = timeout_ms

    def rerank(self, query, docs, top_k=10):
        start = time.time()
        try:
            results = self.reranker.rerank(query, docs, top_k=top_k)
        except Exception:
            results = None

        latency = (time.time() - start) * 1000
        if results is None or latency > self.timeout_ms:
            # 超时或失败，回退到 fallback
            if self.fallback:
                return self.fallback.rerank(query, docs, top_k=top_k)
            # 没有 fallback 时直接返回前 K
            return list(zip(docs[:top_k], [0.0] * top_k))
        return results
```

### 4.18 结合向量检索的 rerank 实战

```python
def dense_retrieve(query_embedding, index, k=100):
    """从向量索引中召回候选"""
    # 这里假设 index.search 返回 doc_id 列表
    return index.search(query_embedding, k=k)

def rerank_by_ids(query, doc_store, reranker, ids):
    """根据 doc_id 获取文档并 rerank"""
    docs = [doc_store[i] for i in ids]
    return reranker.rerank(query, docs, top_k=10)
```

### 4.19 端到端评估流程示意

```python
def evaluate_pipeline(queries, retriever, reranker, ground_truth):
    ranked_docs = {}
    for q in queries:
        # 召回候选并 rerank
        candidates = retriever.retrieve(q, k=100)
        reranked = reranker.rerank(q, candidates, top_k=10)
        ranked_docs[q] = [doc for doc, _ in reranked]
    return evaluate_rerank(ground_truth, ranked_docs)
```

### 4.20 生产级配置清单 (Config Checklist)

```text
- 检索器类型: BM25 / Dense / Hybrid
- Candidate size: 50~200
- Rerank 模型: bge-reranker / Cross-Encoder / Cohere
- Cutoff: 5~20
- Batch size: 8~32
- 超时: 100ms
- 降级策略: fallback to first-stage
```

### 4.21 Rerank 服务的日志与监控

```python
import time

def monitored_rerank(reranker, query, docs):
    start = time.time()
    # 执行 rerank 并记录耗时
    results = reranker.rerank(query, docs, top_k=10)
    latency = (time.time() - start) * 1000
    print(f"Rerank latency: {latency:.2f} ms")
    return results
```

### 4.22 部署注意事项 (Deployment Notes)

- **批处理大小与显存**：batch size 越大吞吐越高，但会占用更多显存
- **CPU/GPU 混合**：将轻量 rerank 放在 CPU，重型放 GPU
- **版本化模型**：rerank 模型需要与评估结果对应版本
- **灰度发布**：先在小流量上验证效果再全量上线

```python
import time

def monitored_rerank(reranker, query, docs):
    start = time.time()
    results = reranker.rerank(query, docs, top_k=10)
    latency = (time.time() - start) * 1000
    print(f"Rerank latency: {latency:.2f} ms")
    return results
```
```
```
```

---

## 5. 实验对比 (Rerank Impact on Metrics)

### 5.1 第一阶段 vs 第二阶段指标对比

| 配置 | Recall@100 | MRR@10 | NDCG@10 | Precision@5 |
|---|---|---|---|---|
| BM25 (第一阶段) | 0.86 | 0.32 | 0.38 | 0.41 |
| Dense (第一阶段) | 0.90 | 0.34 | 0.41 | 0.43 |
| Hybrid (第一阶段) | 0.93 | 0.36 | 0.45 | 0.46 |
| Hybrid + Rerank (Top-50) | 0.93 | 0.49 | 0.58 | 0.61 |
| Hybrid + Rerank (Top-100) | 0.93 | 0.51 | 0.59 | 0.63 |

**结论**：Rerank 不改变 Recall@100，但显著提升 MRR 与 NDCG，说明排序质量提升明显。

### 5.2 Rerank 截断与延迟分析

| Candidate Size (K) | Rerank Top-N | Rerank 延迟 | 总延迟 | NDCG@10 |
|---|---|---|---|---|
| 20 | 5 | 18ms | 42ms | 0.52 |
| 50 | 10 | 45ms | 70ms | 0.58 |
| 100 | 10 | 90ms | 120ms | 0.59 |
| 200 | 10 | 180ms | 210ms | 0.60 |

**结论**：NDCG 提升在 K=50 后趋于饱和，但延迟线性增长，工程上常选择 K=50 或 K=100。

### 5.3 不同 Reranker 的对比

| Reranker | MRR@10 | NDCG@10 | 平均延迟 | 备注 |
|---|---|---|---|---|
| Cross-Encoder (MiniLM) | 0.48 | 0.57 | 35ms | 轻量、稳定 |
| bge-reranker-large | 0.52 | 0.60 | 60ms | 强基线 |
| Cohere Rerank | 0.53 | 0.61 | 120ms | API 稳定 |
| ColBERTv2 | 0.50 | 0.58 | 45ms | 大规模 |
| LLM-based Rerank | 0.56 | 0.64 | 400ms | 成本高 |

**结论**：Cross-Encoder 类模型在性价比上最优；LLM-based 适合高价值场景。

### 5.4 延迟分解 (Latency Breakdown)

| 阶段 | 平均耗时 | 说明 |
|---|---|---|
| Query 处理 | 5ms | 分词、重写 |
| 第一阶段检索 | 25ms | ANN 或 BM25 |
| Rerank 推理 | 60ms | Cross-Encoder |
| 后处理 | 10ms | 去重、多样性 |

**结论**：Rerank 是延迟的主要来源，因此优化重点应在 rerank 推理与候选规模。

### 5.4 质量与成本的折中策略

在真实系统中，可以使用“多级 rerank”策略：

1. Top-200 召回 (BM25 + Dense)
2. Top-50 轻量 rerank (MiniLM)
3. Top-5 高强度 rerank (LLM 或 BGE)

该策略能够显著提升最终排序质量，同时将整体延迟控制在可接受范围。

### 5.5 第一阶段与第二阶段指标的差异分析

第一阶段指标侧重 **Recall@K**，而第二阶段更多反映 **排序质量**：

- **Recall@K**：衡量召回覆盖率
- **MRR@K**：衡量首个相关文档的位置
- **NDCG@K**：考虑多个相关文档的排序

Rerank 主要改善 MRR 与 NDCG，因此在实验报告中需要同时展示第一阶段与第二阶段指标。

### 5.6 Rerank 对业务指标的影响

在问答或搜索业务中，Rerank 的价值常体现为：

- **点击率 (CTR) 提升**：更相关文档在前
- **回答满意度提升**：LLM 输入更相关证据
- **用户停留时长降低**：更快获得准确答案

### 5.7 更详细的消融实验 (Ablation)

| 实验配置 | Recall@100 | MRR@10 | NDCG@10 | 延迟 |
|---|---|---|---|---|
| Dense Only | 0.90 | 0.34 | 0.41 | 30ms |
| Dense + Cross-Encoder | 0.90 | 0.49 | 0.58 | 90ms |
| Dense + BGE Reranker | 0.90 | 0.52 | 0.60 | 120ms |
| Dense + Cohere Rerank | 0.90 | 0.53 | 0.61 | 200ms |
| Dense + LLM Rerank | 0.90 | 0.56 | 0.64 | 600ms |

**结论**：高精度模型的提升明显，但成本与延迟呈指数增长。

### 5.8 吞吐量与资源消耗

| 设备 | 模型 | QPS | P95 延迟 | 备注 |
|---|---|---|---|---|
| CPU (16 核) | MiniLM Cross-Encoder | 20 | 120ms | 轻量 |
| GPU (T4) | bge-reranker-large | 80 | 45ms | 稳定 |
| GPU (A100) | bge-reranker-large | 220 | 18ms | 高吞吐 |
| API (Cohere) | Cohere Rerank | 15 | 200ms | 外部依赖 |

该表说明 GPU 环境下 rerank 性能显著提升，适合在线服务。

### 5.9 不同 Top-N 对质量的影响

| Top-N | MRR@10 | NDCG@10 | 备注 |
|---|---|---|---|
| 3 | 0.44 | 0.52 | 过小，易漏关键信息 |
| 5 | 0.48 | 0.56 | 基线可用 |
| 10 | 0.52 | 0.60 | 性价比最佳 |
| 20 | 0.53 | 0.61 | 提升有限 |

### 5.10 线上 A/B 测试示例

| 版本 | CTR | 用户满意度 | 备注 |
|---|---|---|---|
| Baseline (无 rerank) | 0.21 | 3.8/5 | 仅第一阶段 |
| + Cross-Encoder | 0.26 | 4.1/5 | 质量明显提升 |
| + bge-reranker | 0.28 | 4.2/5 | 最佳折中 |

线上结果通常验证 rerank 对业务指标的真实提升。

### 5.11 数据集维度的对比

| 数据集 | Baseline NDCG@10 | +Rerank NDCG@10 | 提升 |
|---|---|---|---|
| MS MARCO | 0.45 | 0.59 | +0.14 |
| TREC | 0.42 | 0.56 | +0.14 |
| BEIR (Avg) | 0.38 | 0.50 | +0.12 |

该结果表明 Rerank 在多个公开基准上均能带来稳定提升。

### 5.12 不同候选规模的召回变化

| K | Recall@K | NDCG@10 | 说明 |
|---|---|---|---|
| 20 | 0.82 | 0.52 | 候选不足 |
| 50 | 0.89 | 0.58 | 折中 |
| 100 | 0.93 | 0.60 | 较佳 |
| 200 | 0.95 | 0.61 | 收益有限 |

该表说明候选规模与 rerank 提升之间存在边际收益递减。

### 5.13 真实场景案例分析 (Case Study)

假设企业知识库中包含大量政策与流程文档，用户查询“请说明报销流程与审批时限”。

- **无 rerank**：Top-5 文档包含多个“费用类别说明”，但缺少“审批时限”。
- **加入 rerank**：Top-5 中出现“报销流程与审批时限”的文档，LLM 能生成完整答案。

在 A/B 测试中，该查询群体的满意度提升约 12%。这类案例说明 rerank 能显著提升“细粒度需求”的命中率。

### 5.14 模型蒸馏后的效果

通过将强 reranker 蒸馏到轻量模型，可降低延迟：

| 模型 | NDCG@10 | 延迟 | 备注 |
|---|---|---|---|
| bge-reranker-large | 0.60 | 60ms | 基线 |
| 蒸馏模型 (small) | 0.57 | 18ms | 速度快 |

此策略适合对成本敏感但仍需较好排序效果的系统。

### 5.16 成本-质量曲线 (Cost-Quality Curve)

Rerank 常用成本-质量曲线展示效果：

| 模型 | 成本 (相对) | NDCG@10 | 备注 |
|---|---|---|---|
| MiniLM Cross-Encoder | 1x | 0.57 | 轻量 |
| bge-reranker-large | 2x | 0.60 | 强基线 |
| Cohere Rerank | 3x | 0.61 | API 成本 |
| LLM-based Rerank | 8x | 0.64 | 高精度 |

该曲线帮助在预算与效果之间做决策。

### 5.17 评估协议建议 (Evaluation Protocol)

建议采用一致的评估协议，避免“离线看起来好，线上不稳定”：

1. 使用固定验证集与固定负样本策略
2. 同时报告 Recall@K 与 NDCG@K
3. 记录 rerank cutoff 与候选规模
4. 对每次模型更新做回归对比

评估协议示例：

| 项目 | 说明 |
|---|---|
| 数据划分 | 训练/验证/测试 (80/10/10) |
| 候选规模 | K=100 |
| Rerank cutoff | N=10 |
| 指标 | Recall@100, MRR@10, NDCG@10 |

评估报告中建议固定随机种子与候选生成策略，避免由于随机性导致结果波动。

---

## 6. 最佳实践与常见陷阱

### 6.1 最佳实践

1. **先保证召回，再优化排序**：Rerank 无法弥补第一阶段完全漏掉的文档。
2. **控制候选规模**：推荐 Top-50~Top-100 作为 rerank 输入。
3. **批量计算与缓存**：提升吞吐量，降低峰值延迟。
4. **分层 rerank**：轻量模型做粗排，强模型做终排。
5. **稳定评估指标**：优先关注 NDCG@10、MRR@10 等排序指标。
6. **结合领域微调**：领域数据微调 reranker 可带来显著提升。

### 6.2 常见陷阱

1. **误以为 Rerank 提升 Recall**：Rerank 主要优化排序，不会提高 Top-K 召回。
2. **过高的 rerank cutoff**：延迟和成本显著增加，但收益有限。
3. **忽略文档截断**：长文档会被截断导致信息丢失，需要合理分块 (chunking)。
4. **忽略语言/领域差异**：通用 reranker 在垂直领域表现不稳定。
5. **把 API 延迟当作常量**：外部服务可能有波动，需要考虑超时与降级。

### 6.3 实用建议 (Practical Checklist)

```text
- 第一阶段召回: K=100 (Hybrid: BM25 + Dense)
- Rerank cutoff: N=10
- 模型选择: bge-reranker-large 或 cross-encoder/ms-marco
- 超时控制: 100ms
- 监控指标: NDCG@10, MRR@10, Latency P95
```

### 6.4 延迟分析与优化技巧

- **批量推理**：将多个 query-doc pair 一次性送入模型。
- **GPU 加速**：交叉编码器推理可显著提升吞吐。
- **候选预过滤**：使用 BM25 分数阈值过滤弱候选。
- **分段 rerank**：先用轻量模型筛一遍，再用强模型精排。

### 6.4.1 线上调优流程

建议采用“离线评估 + 在线小流量验证”的迭代方式：

1. 离线对比多个 reranker
2. 选择最优模型上线灰度
3. 观察关键指标 (CTR、MRR、Latency)
4. 全量发布或回滚

### 6.5 安全与鲁棒性 (Robustness)

Rerank 在实际系统中需要考虑：

- **对抗样本**：恶意文档可能通过关键词堆砌提升排序
- **偏差与公平性**：训练数据可能带来偏差，需要监控
- **跨域迁移**：跨领域问题会导致排序质量下降

建议通过离线评估 + 在线监控结合的方式持续优化。

### 6.6 在线监控与可观测性

推荐监控以下指标：

- **Latency P95/P99**：确保满足 SLA
- **NDCG@10 / MRR@10**：排序质量指标
- **Rerank 覆盖率**：多少 query 触发 rerank
- **错误率**：API 调用失败或超时

### 6.7 排序可解释性

在企业场景中，常需要解释为何某文档排在前面。可行策略：

- 保存高分文档与相似度分数
- 展示 query 与文档高匹配片段
- 记录 rerank 模型版本与参数

### 6.8 数据闭环与持续优化

建议将线上点击与反馈回流到训练：

- 记录点击作为弱标签
- 将未点击文档作为负样本
- 定期微调 reranker

### 6.9 延迟优化清单 (Latency Checklist)

```text
- 控制 candidate size (50~100)
- 使用 batch 推理
- 优先部署 GPU 推理
- 使用轻量模型做预筛
- 对热门 query 做结果缓存
- 监控 P95/P99 并设置降级
```

### 6.10 Rerank 与知识更新

当知识库频繁更新时，应注意：

- 及时更新索引与候选集
- 确保 reranker 不依赖过时特征
- 使用增量评估保证更新后质量

### 6.11 FAQ 常见问题

**Q1: Rerank 能否提高 Recall？**
答：不会提高候选覆盖率，但能提高 Top-N 质量。

**Q2: 候选规模应该多大？**
答：常用 50~100，根据延迟预算调整。

**Q3: 为什么 rerank 分数与 BM25 不一致？**
答：reranker 建模更复杂，分数分布与 BM25 不可直接比较，需校准或归一化。

### 6.7 降级策略 (Fallback)

当 rerank 服务不可用时，需降级：

- 回退到第一阶段排序
- 回退到轻量 cross-encoder
- 使用缓存结果

降级策略可保证系统可用性。

### 6.8 Rerank 与上下文窗口的配合

Rerank 并非只影响排序，还决定上下文内容的“覆盖面”。在 LLM 上下文窗口有限时，建议：

- 先 rerank，再做 chunk 选择
- 避免只选高分但重复的文档
- 对不同来源做多样性约束

### 6.9 标注噪声与数据泄漏

Rerank 训练中常见问题：

- **标注噪声**：错误标注会导致模型学习错误边界
- **数据泄漏**：训练与评估重叠会虚高指标

建议通过交叉验证、严格数据划分与人工抽查降低风险。

### 6.10 长文档与截断问题

Cross-Encoder 受限于最大输入长度，长文档易被截断导致丢失关键信息。应对策略：

- 先进行 chunking，再 rerank
- 使用摘要 (summary) 替代全文
- 在 rerank 之前做标题与摘要融合

### 6.12 部署验收清单

```text
- 离线评估指标达标 (MRR/NDCG)
- 线上延迟满足 SLA
- 降级策略可用
- 监控与报警配置完成
- 模型版本与数据版本记录清晰

### 6.13 SLA 风险控制

当 rerank 作为核心路径时，需要控制尾延迟：

- 设定 P95/P99 目标
- 超时触发降级
- 对高峰时段做限流

### 6.14 迭代改进清单

```text
- 每次更新记录模型版本与训练数据版本
- 对关键查询集做回归测试
- 记录 rerank cutoff 与候选规模的变动
- 定期分析失败案例并调整负样本
```
```

---

## 7. 总结

Rerank (重排序) 是现代检索系统提升精度的核心技术。其价值在于在不牺牲召回覆盖的前提下显著提高排序质量，从而提升搜索体验、RAG 生成质量与业务指标。

关键结论如下：

1. **两阶段检索是标准架构**：第一阶段保证召回，第二阶段保证精度。
2. **Cross-Encoder 是最常用 reranker**：精度高但成本大，适合 Top-K。
3. **ColBERT 提供高效折中**：在大规模检索中兼顾效率与质量。
4. **Rerank cutoff 与延迟权衡是工程核心**：常用 Top-50/Top-100 作为候选。
5. **指标提升集中在 MRR/NDCG**：Rerank 不提高 Recall，但显著提升排序质量。

在工程实践中，推荐以 bge-reranker 或轻量 cross-encoder 作为基线，结合候选规模控制与缓存机制，再根据业务价值引入更强的 LLM-based rerank。

如果需要快速落地，可从“Hybrid 召回 + bge-reranker + Top-10”起步，再逐步优化延迟与质量。
对于严格延迟场景，可先用轻量 reranker 保障稳定，再引入更强模型做二次精排。
