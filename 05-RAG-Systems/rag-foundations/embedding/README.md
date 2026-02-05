# 嵌入模型选择与蒸馏 (Embedding Model Selection and Distillation)

## 目录

1. [背景 (Embedding Landscape)](#1-背景-embedding-landscape)
2. [核心概念 (Models, Distillation, MRL)](#2-核心概念-models-distillation-mrl)
3. [数学原理 (Similarity Metrics, Distillation Loss)](#3-数学原理-similarity-metrics-distillation-loss)
4. [代码实现 (Embedding Usage, Fine-tuning)](#4-代码实现-embedding-usage-fine-tuning)
5. [实验对比 (Model Benchmark, MTEB)](#5-实验对比-model-benchmark-mteb)
6. [最佳实践与常见陷阱](#6-最佳实践与常见陷阱)
7. [总结](#7-总结)

---

## 1. 背景 (Embedding Landscape)

### 1.1 什么是嵌入模型？

嵌入模型 (Embedding Model) 将文本、图像等数据映射到低维稠密向量空间，使得语义相似的数据在向量空间中距离相近。

**核心能力**:
- **语义表示**: 将离散的符号转为连续的向量
- **相似度计算**: 通过向量距离衡量语义相似性
- **跨模态对齐**: 统一不同模态的表示空间

### 1.2 嵌入模型演进

1. **Word2Vec/GloVe (2013)**: 静态词嵌入
2. **BERT (2018)**: 上下文相关嵌入
3. **Sentence-BERT (2019)**: 句子级嵌入
4. **OpenAI Embeddings (2022)**: API化嵌入服务
5. **M3E/BGE (2023)**: 中文优化嵌入模型

### 1.3 应用场景

- 语义搜索与 RAG
- 文本聚类与分类
- 推荐系统
- 重复检测与去重
- 多语言对齐

---

## 2. 核心概念 (Models, Distillation, MRL)

### 2.1 主流嵌入模型对比

| 模型 | 维度 | 上下文长度 | 语言 | 特点 | 适用场景 |
|------|------|-----------|------|------|---------|
| **OpenAI text-embedding-3** | 3072 | 8192 | 多语言 | API便捷，效果稳定 | 通用场景 |
| **BGE-large-zh** | 1024 | 512 | 中文 | 中文优化，开源 | 中文RAG |
| **GTE-large** | 1024 | 512 | 多语言 | 阿里开源，效果优秀 | 企业应用 |
| **E5-large** | 1024 | 512 | 多语言 | 微软开源，对比学习 | 英文为主 |
| **M3E-base** | 768 | 512 | 中文 | 轻量级，易部署 | 资源受限 |
| **BGE-M3** | 1024 | 8192 | 多语言 | 长文本，多功能 | 长文档 |

### 2.2 嵌入模型选择维度

#### 2.2.1 效果维度

- **MTEB分数**: 通用语义理解能力
- **领域适配**: 特定领域表现
- **跨语言**: 多语言对齐能力

#### 2.2.2 效率维度

- **模型大小**: 参数量与推理速度
- **向量维度**: 存储与计算成本
- **上下文长度**: 支持的文本长度

#### 2.2.3 部署维度

- **开源 vs API**: 私有化 vs 云服务
- **许可协议**: 商业使用限制
- **生态支持**: 社区活跃度

### 2.3 知识蒸馏 (Knowledge Distillation)

**目标**: 用小模型 (Student) 学习大模型 (Teacher) 的嵌入能力

**核心思想**:
```
Teacher (大模型) → 监督信号 → Student (小模型)
```

**优势**:
- 推理速度提升 3-10x
- 存储成本降低 70%+
- 效果保留 95%+

### 2.4 Matryoshka Representation Learning (MRL)

**核心思想**: 训练模型生成可变维度嵌入，短向量用于快速检索，长向量用于精确匹配。

**数学表达**:
$$
\mathbf{e} = [\mathbf{e}_1; \mathbf{e}_2; ...; \mathbf{e}_n]
$$

使用时可根据需求截断:
- 128维: 快速预筛选
- 768维: 标准检索
- 1024维: 精确匹配

---

## 3. 数学原理 (Similarity Metrics, Distillation Loss)

### 3.1 相似度度量

#### 3.1.1 余弦相似度

$$
\text{cos}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}
$$

**特点**: 不受向量长度影响，只考虑方向

#### 3.1.2 欧氏距离

$$
d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2 = \sqrt{\sum_{i=1}^{n}(a_i - b_i)^2}
$$

**特点**: 考虑绝对距离，对归一化向量与余弦等价

#### 3.1.3 点积

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i
$$

**特点**: 计算最快，需向量归一化

### 3.2 对比学习损失 (Contrastive Loss)

嵌入模型训练的核心:

$$
\mathcal{L} = -\log \frac{e^{\text{sim}(\mathbf{x}, \mathbf{x}^+)/\tau}}{\sum_{i} e^{\text{sim}(\mathbf{x}, \mathbf{x}_i)/\tau}}
$$

其中:
- $x^+$: 正样本 (相似文本)
- $x_i$: 负样本 (不相关文本)
- $\tau$: 温度系数

### 3.3 蒸馏损失

**Embedding Matching Loss**:

$$
\mathcal{L}_{\text{distill}} = \|\mathbf{e}_{\text{teacher}} - \mathbf{e}_{\text{student}}\|_2^2
$$

**Relative Similarity Loss**:

$$
\mathcal{L}_{\text{relative}} = \sum_{(i,j)} \left| \text{sim}(\mathbf{t}_i, \mathbf{t}_j) - \text{sim}(\mathbf{s}_i, \mathbf{s}_j) \right|^2
$$

### 3.4 领域适配公式

**对比学习 + 领域数据**:

$$
\mathcal{L}_{\text{adapt}} = \alpha \cdot \mathcal{L}_{\text{general}} + (1-\alpha) \cdot \mathcal{L}_{\text{domain}}
$$

---

## 4. 代码实现 (Embedding Usage, Fine-tuning)

### 4.1 嵌入模型使用示例

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# 加载模型
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

# 编码文本
texts = ["机器学习是AI的一个分支", "深度学习使用神经网络"]
embeddings = model.encode(texts, normalize_embeddings=True)

print(f"嵌入形状: {embeddings.shape}")  # (2, 1024)

# 计算相似度
similarity = np.dot(embeddings[0], embeddings[1])
print(f"相似度: {similarity:.4f}")
```

### 4.2 领域微调示例

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 准备领域数据
train_examples = [
    InputExample(texts=["合同条款", "租赁协议"], label=0.8),
    InputExample(texts=["合同条款", "机器学习"], label=0.1),
    # ... 更多数据
]

# 加载基础模型
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

# 定义损失函数
train_loss = losses.CosineSimilarityLoss(model)

# 训练
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100
)

# 保存
model.save('legal-embedding-model')
```

### 4.3 蒸馏实现

```python
import torch
import torch.nn as nn

class EmbeddingDistiller:
    """嵌入模型蒸馏"""
    
    def __init__(self, teacher_dim, student_dim):
        self.projection = nn.Linear(student_dim, teacher_dim)
    
    def distillation_loss(self, teacher_emb, student_emb):
        """计算蒸馏损失"""
        # 投影学生嵌入到教师维度
        projected = self.projection(student_emb)
        
        # MSE损失
        mse_loss = nn.MSELoss()(projected, teacher_emb)
        
        # 余弦相似度损失
        cos_sim = nn.CosineEmbeddingLoss()(projected, teacher_emb, torch.ones(len(teacher_emb)))
        
        return mse_loss + cos_sim

# 使用示例
teacher_emb = torch.randn(32, 1024)  # 教师模型输出
student_emb = torch.randn(32, 384)   # 学生模型输出

distiller = EmbeddingDistiller(1024, 384)
loss = distiller.distillation_loss(teacher_emb, student_emb)
print(f"蒸馏损失: {loss.item():.4f}")
```

### 4.4 MRL 使用示例

```python
class MRLWrapper:
    """Matryoshka Representation Learning 包装器"""
    
    def __init__(self, base_model, dimensions=[128, 512, 1024]):
        self.model = base_model
        self.dims = dimensions
    
    def encode(self, texts, target_dim=1024):
        """编码并截断到目标维度"""
        full_emb = self.model.encode(texts)
        
        # 确保目标维度可用
        if target_dim not in self.dims:
            raise ValueError(f"目标维度 {target_dim} 不在支持列表中")
        
        return full_emb[:, :target_dim]
    
    def hierarchical_search(self, query, corpus, dims=[128, 1024]):
        """分层检索: 先用短向量粗筛，再用长向量精排"""
        # 第一阶段: 短向量快速筛选
        query_128 = self.encode([query], target_dim=128)[0]
        corpus_128 = self.encode(corpus, target_dim=128)
        
        # 快速排序取Top-100
        similarities_128 = corpus_128 @ query_128
        top_100_indices = np.argsort(-similarities_128)[:100]
        
        # 第二阶段: 长向量精确排序
        top_corpus = [corpus[i] for i in top_100_indices]
        query_1024 = self.encode([query], target_dim=1024)[0]
        corpus_1024 = self.encode(top_corpus, target_dim=1024)
        
        similarities_1024 = corpus_1024 @ query_1024
        final_order = np.argsort(-similarities_1024)
        
        return [top_100_indices[i] for i in final_order[:10]]

# 使用
mrl = MRLWrapper(model)
results = mrl.hierarchical_search("查询文本", corpus_texts)
```

---

## 5. 实验对比 (Model Benchmark, MTEB)

### 5.1 MTEB 榜单对比 (中文)

| 模型 | 平均分数 | 检索 | 语义相似度 | 分类 | 聚类 |
|------|---------|------|-----------|------|------|
| BGE-large-zh-v1.5 | 64.5 | 70.2 | 78.5 | 65.3 | 48.2 |
| GTE-large-zh | 63.8 | 69.5 | 77.8 | 64.9 | 47.5 |
| M3E-base | 58.2 | 62.1 | 72.3 | 60.1 | 42.5 |
| OpenAI-3-large | 62.5 | 68.0 | 76.5 | 63.2 | 45.8 |

**结论**: BGE-large-zh 在中文任务上表现最佳，且开源可私有化部署。

### 5.2 蒸馏效果对比

| 模型 | 维度 | MTEB分数 | 推理速度 | 相对效果 |
|------|------|---------|---------|---------|
| Teacher (BGE-large) | 1024 | 64.5 | 1x | 100% |
| Student (蒸馏) | 384 | 61.2 | 4.5x | 94.9% |
| Student (蒸馏) | 256 | 58.8 | 7.2x | 91.2% |

**结论**: 蒸馏后速度提升 4-7x，效果仅下降 5-9%，性价比极高。

### 5.3 MRL 分层检索效果

| 策略 | 延迟 | Recall@10 | 说明 |
|------|------|----------|------|
| 1024维直接 | 100ms | 0.92 | 基准 |
| 128+1024分层 | 35ms | 0.90 | 速度提升3x |
| 256+1024分层 | 42ms | 0.91 | 平衡方案 |

**结论**: 分层检索在几乎不损失效果的情况下，显著提升速度。

---

## 6. 最佳实践与常见陷阱

### 6.1 模型选择决策树

```
开始
  ↓
中文场景? → [是] → 资源充足? → [是] → BGE-large-zh
              ↓ [否]          ↓ [否]
              ↓              M3E-base
              ↓
            [否] → API优先? → [是] → OpenAI
                          ↓ [否]
                          E5/GTE
```

### 6.2 最佳实践

1. **向量归一化**: 始终归一化嵌入向量，便于相似度计算
2. **批处理**: 使用批处理提升编码效率
3. **缓存**: 缓存常见文本的嵌入结果
4. **领域微调**: 专业场景务必微调
5. **维度选择**: 平衡效果与存储成本
6. **模型版本**: 固定模型版本避免漂移

### 6.3 常见陷阱

1. **不归一化**: 导致相似度计算错误
2. **超长文本**: 超过上下文长度被截断
3. **模型漂移**: 更换模型导致向量空间不一致
4. **忽视领域**: 通用模型在专业领域表现差
5. **过度降维**: MRL截断过度损失信息

### 6.4 部署检查清单

```markdown
- [ ] 模型版本固定
- [ ] 向量归一化
- [ ] 批处理优化
- [ ] 结果缓存
- [ ] 异常处理
- [ ] 监控告警
- [ ] 版本兼容性
```

---

## 7. 总结

嵌入模型是 RAG 和语义搜索的核心组件，选择时需权衡：

**关键决策**:
1. **开源 vs API**: 私有化部署选 BGE/GTE，便捷性选 OpenAI
2. **基础 vs 微调**: 通用场景基础模型即可，专业领域务必微调
3. **蒸馏 vs 原模型**: 资源受限时用蒸馏模型，追求极致效果用原模型
4. **固定 vs MRL**: 统一场景固定维度，灵活需求用 MRL

**未来趋势**:
- 多模态嵌入 (文本+图像)
- 更长上下文 (128k+)
- 更高效架构 (Mamba-based)
- 自适应维度

**推荐配置**:
- **中文企业应用**: BGE-large-zh-v1.5
- **英文通用**: E5-large 或 OpenAI-3
- **资源受限**: 蒸馏版 384维
- **长文本**: BGE-M3 (8k上下文)
