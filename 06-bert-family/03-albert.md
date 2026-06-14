---
name: "ALBERT"
year: 2019
family: "06-bert-family"
order: 3
paper: "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"
authors: ["Zhenzhong Lan", "Mingda Chen", "Sebastian Goodman", "Kevin Gimpel", "Piyush Sharma", "Radu Soricut"]
key_idea: "用跨层参数共享 + embedding 因式分解把 BERT-large 参数从 334M 压到 18M 而效果接近,同时把 NSP 改成更难的 SOP(句子顺序预测)"
---

## 前作进展

[BERT-large](01-bert.md) 是 334M 参数,2019 年的工业部署仍然吃力——单 V100 推理 batch 1 大约 30ms,batch 8 就要 200ms,显存占用 1.2GB。[RoBERTa](02-roberta.md) 推到 355M,效果更好但部署成本也更高。

NLP 社区开始有声音质疑:**预训练模型一定要这么大吗?BERT-large 里这么多参数,有多少是真正必要的、有多少是冗余的?**

Google 的 Lan 等人 2019 年 9 月发表 *ALBERT: A Lite BERT* 给出了一个激进答案:**BERT 的大部分参数是冗余的**,具体表现在两处:

**1. Token embedding 维度等于 hidden size 是浪费**——BERT 的 token embedding 和 hidden state 都用同一个维度 `H`(BERT-base 768,BERT-large 1024)。但语言学上,**token 本身只承载词汇信息**(`vocab_size = 30K` 用 128 维就够编码),hidden state 需要承载**上下文相关表征**(需要 768+ 维)。强制让两者维度相等是浪费

**2. 24 层之间参数互相独立是冗余**——BERT-large 24 层每层有独立的 attention + FFN 参数。但 ALBERT 团队怀疑深层 Transformer 的不同层学到的是**相似变换的不同尺度**,完全共享一组参数应该也能 work

ALBERT 把这两个想法做出来,产生了几个"超级压缩"的模型:

| 模型 | 参数量 | 层数 | hidden size | 备注 |
|------|------|------|------|------|
| BERT-base | 110M | 12 | 768 | 原版对照 |
| BERT-large | 334M | 24 | 1024 | 原版对照 |
| ALBERT-base | **12M** | 12 | 768 | 比 BERT-base 小 **9×** |
| ALBERT-large | **18M** | 24 | 1024 | 比 BERT-large 小 **18×** |
| ALBERT-xlarge | 60M | 24 | 2048 | 仍比 BERT-large 小 5.6× |
| ALBERT-xxlarge | **235M** | 12 | **4096** | 24 → 12 层 + hidden 4× |

**最反直觉的结果是 ALBERT-xxlarge**(235M)在 GLUE 上击败 BERT-large(334M)且效果显著更好——参数少 30% 但能力更强。这是 2019 年最有冲击力的"参数效率"实证。

但 ALBERT 也有代价——**参数共享降低参数量但不降推理时间**(每层仍要算同样的 forward),实际部署上 ALBERT 没有 [DistilBERT](04-distilbert.md) 那么受欢迎。

## 改动 1:Embedding 因式分解

[BERT](01-bert.md) 的 input embedding 矩阵是 `V × H`,其中 `V = vocab_size = 30K`,`H = hidden_size`(BERT-base 768,BERT-large 1024)。这一矩阵参数量:

| 模型 | V × H | 占总参数比例 |
|------|------|------|
| BERT-base | 30K × 768 = **23M** | 21% |
| BERT-large | 30K × 1024 = **30M** | 9% |

embedding 这一层占 BERT-base 总参数 21%——非常大的开销,而它做的事其实很简单(token id → 向量查表)。

ALBERT 的因式分解:**先把 token id 投影到一个小的 `E` 维空间,再升到 `H`**:

$$
\text{Embedding}(x) = W_E \cdot W_V \cdot \text{one\_hot}(x)
$$

其中 `W_V \in R^{V \times E}` 是 `V × E` 的查表矩阵,`W_E \in R^{E \times H}` 是 `E × H` 的升维投影。`E` 典型取 128(比 H 小 6–32×)。

参数对比:

| 模型 | 原版 V×H | 因式分解 V×E + E×H | 减少 |
|------|------|------|------|
| BERT-base(H=768)| 23M | 30K×128 + 128×768 = 3.9M + 0.1M = **4M** | -83% |
| ALBERT-xxlarge(H=4096)| 122M | 30K×128 + 128×4096 = 3.9M + 0.5M = **4.4M** | -96% |

因式分解的合理性来自一个语言学观察:**token embedding 的"信息容量"由 vocab_size 决定,不应该随 hidden_size 线性增长**。当模型变大(H 从 768 推到 4096),token embedding 不需要变大——它们仍然只是"30K 个词的查表"。强制让 V × H 同步增长是浪费。

## 改动 2:跨层参数共享

[BERT-large](01-bert.md) 有 24 层,每层独立的参数:`24 × (4 × H² + 8 × H²) = 24 × 12 H²`(attention + FFN 估算)。这是模型参数的大头(BERT-large 总参数的 ~75%)。

ALBERT 提出**所有 24 层共享同一组参数**——只存一份 Transformer block 的权重,forward 时重复用 24 次:

```python
class ALBERTLayer(nn.Module):
    """单个 Transformer block,会被所有层共享"""
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.attn = ...
        self.ffn = ...

class ALBERT(nn.Module):
    def __init__(self, num_layers, ...):
        # 注意:不是 nn.ModuleList,只有一个 layer 实例
        self.layer = ALBERTLayer(...)
        self.num_layers = num_layers

    def forward(self, x):
        for _ in range(self.num_layers):
            x = self.layer(x)  # 同一个 layer 重复用 24 次
        return x
```

参数压缩对比(以 BERT-large 24 层为例):

- BERT-large:24 × 12M = 288M(全部 attention + FFN)
- **ALBERT-large(共享)**:1 × 12M = **12M**——少 24×

加上 embedding 因式分解,ALBERT-large 总参数只有 18M(BERT-large 是 334M,**少 18×**)。

但**推理时间不变**——forward 时仍要走 24 层,每层都是同一组参数但要算 24 次。参数共享只节省**显存和文件大小**,不节省 FLOPs。这是 ALBERT 在工业部署不如 [DistilBERT](04-distilbert.md) 受欢迎的根本原因——DistilBERT 是真把 24 层变成 6 层,推理快 2×。

ALBERT 论文里也比较了几个共享策略(Table 7):

| 共享策略 | SQuAD F1 | 参数 |
|------|------|------|
| 全部独立(BERT-style) | 90.4 | 89M |
| 只共享 attention | 89.9 | 64M |
| 只共享 FFN | 90.4 | 38M |
| **全部共享**(默认) | **90.0** | **12M** |

观察:**只共享 FFN 几乎不损性能,全部共享损 0.4 分**——说明 FFN 的参数冗余度比 attention 还高,可以完全共享而几乎不掉点。

## 改动 3:NSP → SOP

[BERT 原版](01-bert.md) 的 NSP(Next Sentence Prediction)被 [RoBERTa](02-roberta.md) 证明几乎无用——negative sample 是从不同文档随机选的句子,模型用浅层主题信号就能区分,学不到深表征。

ALBERT 不简单删 NSP,而是**改成 SOP(Sentence Order Prediction)**——任务是判断两个句子的**顺序是否正确**:

- **正例**:文档里的连续句子 (A, B),按原顺序
- **负例**:同一对句子但**调换顺序** (B, A)

关键差异:**正负样本都是同一对句子,只是顺序不同**。模型不能用主题信号(主题完全一样),必须学到真正的"句间连贯性"——什么样的 B 应该在 A 之后,什么样的不应该。

SOP 的消融(Table 5):

| 预训练任务 | SQuAD F1 | RACE |
|------|------|------|
| 仅 MLM(无 NSP, 无 SOP) | 81.0 | 64.0 |
| MLM + NSP | 81.5 | 64.5 |
| **MLM + SOP** | **82.1** | **65.5** |

SOP 比 NSP 涨 0.6–1.0 分,**证明任务设计本身有真实信号**——不只是"任务有没有"的问题,而是"任务难度合不合适"的问题。

SOP 这一思想后来被广泛借鉴:**对比学习里"正负样本要语义相近但有差异"** 是 ALBERT 的精神延续。今天的 Sentence-BERT、SimCSE 等检索模型也用类似策略构造 hard negative。

## 性能与权衡

ALBERT 各版本和 BERT 的对比(论文 Table 2):

| 模型 | 参数 | SQuAD 1.1 F1 | MNLI | RACE | 训练时间 |
|------|------|------|------|------|------|
| BERT-base | 108M | 90.4 | 84.6 | 64.3 | 4.7 h |
| BERT-large | 334M | 92.2 | 86.6 | 70.4 | 11.1 h |
| ALBERT-base | **12M** | 89.3 | 81.6 | 63.5 | 5.6 h |
| ALBERT-large | **18M** | 90.9 | 83.9 | 66.0 | 17.7 h |
| ALBERT-xlarge | **60M** | 93.0 | 86.4 | 73.9 | 41.8 h |
| **ALBERT-xxlarge** | **235M** | **94.1** | **88.1** | **82.3** | **77.4 h** |

观察:

- **ALBERT-base(12M)比 BERT-base(108M)少 9×,效果稍弱**——参数压缩有代价
- **ALBERT-xxlarge(235M)击败 BERT-large(334M)**,且 RACE 上涨 12 分——参数共享让深层模型更易训
- **训练时间反而更长**——因为 ALBERT-xxlarge hidden_size 推到 4096,FFN 计算量大幅增加;且 12 层时每层负担重

第三点是 ALBERT 的关键 trade-off:**参数少不等于训练或推理快**。ALBERT 在部署上的真正优势是**显存占用低**(BERT-large 1.2GB → ALBERT-large 70MB),适合内存受限场景(移动端、嵌入式)。但 CPU/GPU 时间几乎不省,这让它在大规模部署里不如 DistilBERT 受欢迎。

## 训练细节

| 维度 | ALBERT-xxlarge |
|------|------|
| 架构 | 12 层(注意比 BERT-large 少一半), d_model=4096, h=64, d_ff=16384 |
| Embedding 维度 E | 128(因式分解到 H=4096) |
| 参数共享 | **all-shared**(attention + FFN 都跨层共享) |
| 参数量 | 235M(对比 BERT-large 334M,少 30%) |
| 预训练任务 | MLM(动态 mask)+ **SOP**(不是 NSP) |
| 数据 | BookCorpus + Wiki(同 BERT,~16GB) |
| 训练 token | ~125B(对比 RoBERTa 2T,少 16×) |
| Batch | 4096 序列 |
| 训练步 | 125K |
| 优化器 | LAMB(大 batch 友好,Layer-wise Adaptive Moments) |
| 训练硬件 | Cloud TPU v3 Pod(64–512 cores) |
| 训练时间 | ~77 小时(对比 BERT-large 11 小时) |

注意 **LAMB 优化器**——它是为大 batch(>32K)训练设计的 Adam 变体,通过 layer-wise normalization 让每层都有合适的有效 learning rate。这一选择后来被多个大 batch 训练工作沿用(BERT-large with LAMB 把 batch 推到 32K)。

## 关键代码

ALBERT 实现的核心是**参数共享**:

```python
import torch
import torch.nn as nn

class ALBERTLayer(nn.Module):
    """单个 Transformer block,会被所有层共享调用"""
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x

class ALBERT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads,
                 intermediate_size, embedding_size=128):
        super().__init__()
        # Embedding 因式分解:V × E,然后 E × H 升维
        self.token_emb = nn.Embedding(vocab_size, embedding_size)
        self.emb_proj = nn.Linear(embedding_size, hidden_size)
        self.pos_emb = nn.Embedding(512, hidden_size)
        # 关键:只有一个 layer 实例,而非 ModuleList
        self.shared_layer = ALBERTLayer(hidden_size, num_heads, intermediate_size)
        self.num_layers = num_layers

    def forward(self, input_ids, attention_mask=None):
        x = self.token_emb(input_ids)
        x = self.emb_proj(x)  # 升到 hidden_size
        x = x + self.pos_emb(torch.arange(x.size(1), device=x.device))
        # 重复用同一个 layer
        for _ in range(self.num_layers):
            x = self.shared_layer(x, mask=attention_mask)
        return x
```

SOP 的数据构造:

```python
def build_sop_examples(documents, max_seq_len=512):
    """构造 SOP 训练数据:正例同序、负例调序"""
    examples = []
    for doc in documents:
        sentences = sent_tokenize(doc)
        for i in range(len(sentences) - 1):
            a, b = sentences[i], sentences[i+1]
            # 50% 概率构造负例(调换顺序)
            if random.random() < 0.5:
                examples.append({"text_a": a, "text_b": b, "label": 1})  # 正序
            else:
                examples.append({"text_a": b, "text_b": a, "label": 0})  # 调序
    return examples
```

注意 SOP 和 NSP 的关键差异——**SOP 的负例从同一对句子的调换得来,NSP 是随机句子**。这一改动让任务从"主题分类"变成了"语义连贯性",真正考验模型对句间逻辑的理解。

## 影响 / 后续

ALBERT 在 BERT-family 历史的位置:**第一次系统化"参数效率"路线**。具体影响:

**1. 参数共享思想被推广**——后续多个工作探索跨层参数共享:Universal Transformer(2019)、CT-BERT(2020)、PaLM(2022 的并行参数化)都受 ALBERT 启发。但完全共享在大模型上效果通常不如部分共享(只共享 attention 或 FFN)

**2. SOP 推动了对比学习范式**——SOP 的"hard negative"思想被 Sentence-BERT(2019)、SimCSE(2021)、E5(2022)等检索模型大规模借鉴。今天的所有现代检索模型都用某种 hard negative mining 策略

**3. Embedding 因式分解被沿用**——大词表场景(多语言模型、字符级模型)广泛用因式分解。XLM-R(2019)、mT5(2020)等多语言模型有时词表上百万,因式分解是必需的

**4. ALBERT 自身在工业部署上不如 DistilBERT**——参数少但速度不快这一局限让 ALBERT 主要在**显存受限的边缘部署**(移动端、IoT)有市场;大规模 API 服务仍用 DistilBERT 或量化的 BERT

**5. "参数效率不等于推理效率"成为新认知**——后续工作开始更明确区分"参数数量"(决定显存和文件大小)和"FLOPs"(决定推理时间)。这一区分在 MoE 时代变得至关重要——MoE 模型的"总参数"和"激活参数"是两个不同概念

ALBERT 留下的开放问题被后续家族节点承接:

- **真正的推理效率**:跨层共享不省 FLOPs → [DistilBERT](04-distilbert.md) 用层数减半 + 蒸馏
- **更高效的预训练任务**:MLM 只有 15% token 参与 loss,信号利用率低 → ELECTRA(2020)用 replaced token detection 推到 100%
- **极端长尾词表**:E=128 处理 30K 词表勉强够,但多语言模型词表上百万时仍要更精细的因式分解 → XLM-R 等

→ [04-distilbert.md](04-distilbert.md) · 知识蒸馏路线,工业部署的真正首选
→ [02-roberta.md](02-roberta.md) · 同年姊妹工作,RoBERTa 改训练 / ALBERT 改架构
→ [01-bert.md](01-bert.md) · 父结构;ALBERT 在它基础上做参数效率改造
→ [../13-moe-efficient/](../13-moe-efficient/) · MoE 是参数和算力解耦的另一条路
→ [../11-peft-lora/](../11-peft-lora/) · LoRA 的参数高效也是 ALBERT 精神的延伸(只是从"共享"换成"低秩")
