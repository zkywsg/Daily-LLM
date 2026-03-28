# Embedding Model Selection and Distillation

**Documentation**: [**English**](README_EN.md) | [**中文**](README.md)

## Table of Contents

1. [Background (Embedding Landscape)](#1-background-embedding-landscape)
2. [Core Concepts (Models, Distillation, MRL)](#2-core-concepts-models-distillation-mrl)
3. [Mathematical Foundations (Similarity Metrics, Distillation Loss)](#3-mathematical-foundations-similarity-metrics-distillation-loss)
4. [Code Implementation (Embedding Usage, Fine-tuning)](#4-code-implementation-embedding-usage-fine-tuning)
5. [Experimental Comparison (Model Benchmark, MTEB)](#5-experimental-comparison-model-benchmark-mteb)
6. [Best Practices and Common Pitfalls](#6-best-practices-and-common-pitfalls)
7. [Summary](#7-summary)

---

## 1. Background (Embedding Landscape)

### 1.1 What Are Embedding Models?

Embedding models map data such as text and images into low-dimensional dense vector spaces, where semantically similar data points are close together in vector space.

**Core Capabilities**:
- **Semantic Representation**: Convert discrete symbols into continuous vectors
- **Similarity Computation**: Measure semantic similarity through vector distance
- **Cross-modal Alignment**: Unify representation spaces across different modalities

### 1.2 Evolution of Embedding Models

1. **Word2Vec/GloVe (2013)**: Static word embeddings
2. **BERT (2018)**: Context-dependent embeddings
3. **Sentence-BERT (2019)**: Sentence-level embeddings
4. **OpenAI Embeddings (2022)**: API-based embedding services
5. **M3E/BGE (2023)**: Chinese-optimized embedding models

### 1.3 Use Cases

- Semantic search and RAG
- Text clustering and classification
- Recommendation systems
- Duplicate detection and deduplication
- Multilingual alignment

---

## 2. Core Concepts (Models, Distillation, MRL)

### 2.1 Mainstream Embedding Model Comparison

| Model | Dimensions | Context Length | Languages | Features | Use Cases |
|-------|-----------|---------------|-----------|----------|-----------|
| **OpenAI text-embedding-3** | 3072 | 8192 | Multilingual | Convenient API, stable quality | General purpose |
| **BGE-large-zh** | 1024 | 512 | Chinese | Chinese-optimized, open-source | Chinese RAG |
| **GTE-large** | 1024 | 512 | Multilingual | Alibaba open-source, excellent performance | Enterprise |
| **E5-large** | 1024 | 512 | Multilingual | Microsoft open-source, contrastive learning | English-centric |
| **M3E-base** | 768 | 512 | Chinese | Lightweight, easy deployment | Resource-constrained |
| **BGE-M3** | 1024 | 8192 | Multilingual | Long-context, multi-functional | Long documents |

### 2.2 Embedding Model Selection Criteria

#### 2.2.1 Effectiveness

- **MTEB Score**: General semantic understanding ability
- **Domain Adaptation**: Performance on specific domains
- **Cross-lingual**: Multilingual alignment capability

#### 2.2.2 Efficiency

- **Model Size**: Parameter count and inference speed
- **Vector Dimensions**: Storage and computation cost
- **Context Length**: Maximum supported text length

#### 2.2.3 Deployment

- **Open-source vs API**: Self-hosted vs cloud service
- **License**: Commercial use restrictions
- **Ecosystem Support**: Community activity level

### 2.3 Knowledge Distillation

**Goal**: Train a small model (Student) to learn the embedding capabilities of a large model (Teacher)

**Core Idea**:
```
Teacher (large model) → Supervision signal → Student (small model)
```

**Advantages**:
- 3-10x inference speed improvement
- 70%+ storage cost reduction
- 95%+ effectiveness retained

### 2.4 Matryoshka Representation Learning (MRL)

**Core Idea**: Train models to produce variable-dimension embeddings — shorter vectors for fast retrieval, longer vectors for precise matching.

**Mathematical Expression**:
$$
\mathbf{e} = [\mathbf{e}_1; \mathbf{e}_2; ...; \mathbf{e}_n]
$$

Truncate based on requirements:
- 128 dimensions: Fast pre-filtering
- 768 dimensions: Standard retrieval
- 1024 dimensions: Precise matching

---

## 3. Mathematical Foundations (Similarity Metrics, Distillation Loss)

### 3.1 Similarity Metrics

#### 3.1.1 Cosine Similarity

$$
\text{cos}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}
$$

**Property**: Independent of vector magnitude, considers only direction

#### 3.1.2 Euclidean Distance

$$
d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2 = \sqrt{\sum_{i=1}^{n}(a_i - b_i)^2}
$$

**Property**: Considers absolute distance; equivalent to cosine for normalized vectors

#### 3.1.3 Dot Product

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i
$$

**Property**: Fastest computation; requires vector normalization

### 3.2 Contrastive Loss

The core of embedding model training:

$$
\mathcal{L} = -\log \frac{e^{\text{sim}(\mathbf{x}, \mathbf{x}^+)/\tau}}{\sum_{i} e^{\text{sim}(\mathbf{x}, \mathbf{x}_i)/\tau}}
$$

Where:
- $x^+$: Positive sample (similar text)
- $x_i$: Negative samples (unrelated text)
- $\tau$: Temperature coefficient

### 3.3 Distillation Loss

**Embedding Matching Loss**:

$$
\mathcal{L}_{\text{distill}} = \|\mathbf{e}_{\text{teacher}} - \mathbf{e}_{\text{student}}\|_2^2
$$

**Relative Similarity Loss**:

$$
\mathcal{L}_{\text{relative}} = \sum_{(i,j)} \left| \text{sim}(\mathbf{t}_i, \mathbf{t}_j) - \text{sim}(\mathbf{s}_i, \mathbf{s}_j) \right|^2
$$

### 3.4 Domain Adaptation Formula

**Contrastive Learning + Domain Data**:

$$
\mathcal{L}_{\text{adapt}} = \alpha \cdot \mathcal{L}_{\text{general}} + (1-\alpha) \cdot \mathcal{L}_{\text{domain}}
$$

---

## 4. Code Implementation (Embedding Usage, Fine-tuning)

### 4.1 Embedding Model Usage Example

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

# Encode text
texts = ["Machine learning is a branch of AI", "Deep learning uses neural networks"]
embeddings = model.encode(texts, normalize_embeddings=True)

print(f"Embedding shape: {embeddings.shape}")  # (2, 1024)

# Compute similarity
similarity = np.dot(embeddings[0], embeddings[1])
print(f"Similarity: {similarity:.4f}")
```

### 4.2 Domain Fine-tuning Example

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Prepare domain data
train_examples = [
    InputExample(texts=["contract terms", "lease agreement"], label=0.8),
    InputExample(texts=["contract terms", "machine learning"], label=0.1),
    # ... more data
]

# Load base model
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

# Define loss function
train_loss = losses.CosineSimilarityLoss(model)

# Train
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100
)

# Save
model.save('legal-embedding-model')
```

### 4.3 Distillation Implementation

```python
import torch
import torch.nn as nn

class EmbeddingDistiller:
    """Embedding model distillation"""

    def __init__(self, teacher_dim, student_dim):
        self.projection = nn.Linear(student_dim, teacher_dim)

    def distillation_loss(self, teacher_emb, student_emb):
        """Compute distillation loss"""
        # Project student embeddings to teacher dimension
        projected = self.projection(student_emb)

        # MSE loss
        mse_loss = nn.MSELoss()(projected, teacher_emb)

        # Cosine similarity loss
        cos_sim = nn.CosineEmbeddingLoss()(projected, teacher_emb, torch.ones(len(teacher_emb)))

        return mse_loss + cos_sim

# Usage example
teacher_emb = torch.randn(32, 1024)  # Teacher model output
student_emb = torch.randn(32, 384)   # Student model output

distiller = EmbeddingDistiller(1024, 384)
loss = distiller.distillation_loss(teacher_emb, student_emb)
print(f"Distillation loss: {loss.item():.4f}")
```

### 4.4 MRL Usage Example

```python
class MRLWrapper:
    """Matryoshka Representation Learning wrapper"""

    def __init__(self, base_model, dimensions=[128, 512, 1024]):
        self.model = base_model
        self.dims = dimensions

    def encode(self, texts, target_dim=1024):
        """Encode and truncate to target dimension"""
        full_emb = self.model.encode(texts)

        # Ensure target dimension is available
        if target_dim not in self.dims:
            raise ValueError(f"Target dimension {target_dim} not in supported list")

        return full_emb[:, :target_dim]

    def hierarchical_search(self, query, corpus, dims=[128, 1024]):
        """Hierarchical retrieval: coarse filtering with short vectors, precise ranking with long vectors"""
        # Stage 1: Fast filtering with short vectors
        query_128 = self.encode([query], target_dim=128)[0]
        corpus_128 = self.encode(corpus, target_dim=128)

        # Quick sort to get Top-100
        similarities_128 = corpus_128 @ query_128
        top_100_indices = np.argsort(-similarities_128)[:100]

        # Stage 2: Precise ranking with long vectors
        top_corpus = [corpus[i] for i in top_100_indices]
        query_1024 = self.encode([query], target_dim=1024)[0]
        corpus_1024 = self.encode(top_corpus, target_dim=1024)

        similarities_1024 = corpus_1024 @ query_1024
        final_order = np.argsort(-similarities_1024)

        return [top_100_indices[i] for i in final_order[:10]]

# Usage
mrl = MRLWrapper(model)
results = mrl.hierarchical_search("query text", corpus_texts)
```

---

## 5. Experimental Comparison (Model Benchmark, MTEB)

### 5.1 MTEB Leaderboard Comparison (Chinese)

| Model | Average Score | Retrieval | Semantic Similarity | Classification | Clustering |
|-------|--------------|-----------|-------------------|----------------|------------|
| BGE-large-zh-v1.5 | 64.5 | 70.2 | 78.5 | 65.3 | 48.2 |
| GTE-large-zh | 63.8 | 69.5 | 77.8 | 64.9 | 47.5 |
| M3E-base | 58.2 | 62.1 | 72.3 | 60.1 | 42.5 |
| OpenAI-3-large | 62.5 | 68.0 | 76.5 | 63.2 | 45.8 |

**Conclusion**: BGE-large-zh achieves the best performance on Chinese tasks and is open-source for self-hosted deployment.

### 5.2 Distillation Effectiveness Comparison

| Model | Dimensions | MTEB Score | Inference Speed | Relative Performance |
|-------|-----------|-----------|----------------|---------------------|
| Teacher (BGE-large) | 1024 | 64.5 | 1x | 100% |
| Student (distilled) | 384 | 61.2 | 4.5x | 94.9% |
| Student (distilled) | 256 | 58.8 | 7.2x | 91.2% |

**Conclusion**: Post-distillation achieves 4-7x speedup with only 5-9% performance drop — excellent cost-effectiveness.

### 5.3 MRL Hierarchical Retrieval Performance

| Strategy | Latency | Recall@10 | Description |
|----------|---------|-----------|-------------|
| 1024-dim direct | 100ms | 0.92 | Baseline |
| 128+1024 hierarchical | 35ms | 0.90 | 3x speed improvement |
| 256+1024 hierarchical | 42ms | 0.91 | Balanced approach |

**Conclusion**: Hierarchical retrieval significantly improves speed with negligible performance loss.

---

## 6. Best Practices and Common Pitfalls

### 6.1 Model Selection Decision Tree

```
Start
  ↓
Chinese scenario? → [Yes] → Sufficient resources? → [Yes] → BGE-large-zh
                     ↓ [No]                         ↓ [No]
                     ↓                              M3E-base
                     ↓
                   [No] → API preferred? → [Yes] → OpenAI
                                         ↓ [No]
                                         E5/GTE
```

### 6.2 Best Practices

1. **Vector Normalization**: Always normalize embedding vectors for consistent similarity computation
2. **Batch Processing**: Use batch encoding to improve throughput
3. **Caching**: Cache embeddings for frequently queried texts
4. **Domain Fine-tuning**: Always fine-tune for specialized domains
5. **Dimension Selection**: Balance effectiveness against storage cost
6. **Model Versioning**: Pin model versions to prevent drift

### 6.3 Common Pitfalls

1. **Missing Normalization**: Leads to incorrect similarity calculations
2. **Exceeding Context Length**: Text beyond the context window gets truncated
3. **Model Drift**: Swapping models breaks vector space consistency
4. **Ignoring Domain Gap**: General models underperform on specialized domains
5. **Over-truncation**: Aggressive MRL truncation loses critical information

### 6.4 Deployment Checklist

```markdown
- [ ] Model version pinned
- [ ] Vector normalization enabled
- [ ] Batch processing optimized
- [ ] Result caching configured
- [ ] Error handling in place
- [ ] Monitoring and alerting set up
- [ ] Version compatibility verified
```

---

## 7. Summary

Embedding models are the core component of RAG and semantic search. Key trade-offs to consider:

**Key Decisions**:
1. **Open-source vs API**: Choose BGE/GTE for self-hosted deployment, OpenAI for convenience
2. **Base vs Fine-tuned**: Base models suffice for general use; always fine-tune for specialized domains
3. **Distilled vs Original**: Use distilled models when resources are limited; use originals for maximum quality
4. **Fixed vs MRL**: Use fixed dimensions for uniform workloads; use MRL for flexible requirements

**Future Trends**:
- Multimodal embeddings (text + image)
- Longer context windows (128k+)
- More efficient architectures (Mamba-based)
- Adaptive dimensionality

**Recommended Configurations**:
- **Chinese enterprise**: BGE-large-zh-v1.5
- **English general**: E5-large or OpenAI-3
- **Resource-constrained**: Distilled 384-dim model
- **Long documents**: BGE-M3 (8k context)
