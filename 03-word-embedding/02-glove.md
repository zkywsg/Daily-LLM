---
name: "GloVe"
year: 2014
family: "03-word-embedding"
order: 2
paper: "GloVe: Global Vectors for Word Representation"
authors: ["Jeffrey Pennington", "Richard Socher", "Christopher D. Manning"]
key_idea: "直接对全局 word-word 共现矩阵做加权 log-bilinear 分解,而不是用局部窗口预测;count-based 路线,理论比 Word2Vec 更清晰,Stanford 团队预训练好的 GloVe 向量成开源标配"
---

## 前作进展

[Word2Vec](01-word2vec.md)(2013)用 prediction-based 方法(skip-gram / CBOW)学词向量,效果好但有几个理论上不优雅的地方:

**1. 只用局部窗口** —— Skip-gram 每次只看 ±5 词上下文,**忽略了全局共现统计**(整个语料里"king"和"queen"出现在一起的总次数)

**2. NEG 不是真正的 likelihood** —— Negative Sampling 是工程 trick,理论上没有清晰的最优解

**3. 训练目标不直观** —— "预测上下文"是辅助任务,词向量是副产品

历史上 NLP 还有一条 **count-based**(计数式)路线:

- **LSA(Latent Semantic Analysis,1990)** —— 对 document-term 矩阵做 SVD,得到词向量。但 SVD 在大矩阵上慢
- **HAL / COALS** —— 对 word-word 共现矩阵做处理,得到词向量
- **Hellinger PCA** —— 对共现概率开根号后 PCA

这些方法理论清晰但**效果不如 Word2Vec**——直接矩阵分解没有抓住对的统计结构。

Pennington 等人(Stanford NLP,2014 年 10 月,EMNLP 2014)的 GloVe 论文给出关键洞察:**应该分解的不是共现次数本身,而是共现概率的比值**。这个量编码了词之间的语义关系。

GloVe 的工程影响立竿见影:

- 论文同时发布 **预训练 GloVe 向量**(Wikipedia + Gigaword 6B / Common Crawl 42B / Common Crawl 840B 三个版本)
- 这些 GloVe 向量成为 2014-2018 NLP 工业 / 学术界的事实标配 input
- Stanford NLP 组的开源策略让 GloVe 影响力一度超过 Word2Vec

GloVe 与 Word2Vec 不是替代关系,而是 **两条路线并存** —— 不同任务上各有优势,大量论文同时用两者做实验对比。

## 核心思想:共现概率比值

### 关键观察

考虑三个词 i = "ice", j = "steam", k 是探针词。看共现概率 $P(k|i) = X_{ik}/X_i$:

| 探针词 k | P(k|ice) | P(k|steam) | P(k|ice)/P(k|steam) |
|------|------|------|------|
| solid | 1.9e-4 | 2.2e-5 | **8.9**(高,"solid" 与 ice 关) |
| gas | 6.6e-5 | 7.8e-4 | **0.085**(低,"gas" 与 steam 关) |
| water | 3.0e-3 | 2.2e-3 | 1.36(中,都相关) |
| fashion | 1.7e-5 | 1.8e-5 | 0.96(中,都不相关) |

**关键**:$P_{ik} / P_{jk}$ 这个比值编码了"k 在多大程度上区分 i 和 j"。GloVe 的目标:**让词向量直接拟合这个比值**。

### 数学模型

GloVe 假设词向量内积应等于 log 共现次数:

$$
v_i^T v_j + b_i + b_j = \log X_{ij}
$$

- $v_i, v_j$ —— 词向量
- $b_i, b_j$ —— bias
- $X_{ij}$ —— 词 i, j 共现次数

### 加权 loss

但直接 fit log $X_{ij}$ 有几个问题:rare co-occurrence($X_{ij}=0$)是 log(0)、common pair 主导 loss。GloVe 加一个加权函数 $f$:

$$
\mathcal{L} = \sum_{i,j=1}^V f(X_{ij}) (v_i^T v_j + b_i + b_j - \log X_{ij})^2
$$

加权函数 $f$:

$$
f(x) = \begin{cases} (x / x_{\max})^\alpha & \text{if } x < x_{\max} \\ 1 & \text{otherwise} \end{cases}
$$

- $x_{\max} = 100$(论文设)
- $\alpha = 3/4$(经验值,与 Word2Vec NEG 巧合相同)

直觉:rare pair 给小权重(噪声大),common pair 给上限权重(防止主导)。

### 算法流程

```
1. 扫一遍语料,构建 word-word 共现矩阵 X_{ij}(用滑窗,通常 ±10)
2. 初始化 v, b 为随机
3. 用 AdaGrad 优化 weighted squared loss
4. 训完后 v_i 就是词 i 的 GloVe 向量
```

### 与 Word2Vec 的关系

后续研究(Levy & Goldberg 2014)发现 Word2Vec skip-gram 的 NEG 实际上隐式地做了 PMI(pointwise mutual information)矩阵分解。GloVe 显式做的是 log $X_{ij}$ 分解。两者在数学上有深刻联系,这部分解释了为何它们效果接近。

### 共现矩阵的预处理

GloVe 的共现矩阵不是简单计数。论文用 **distance weighting**:窗口里距离 d 的词共现权重 = 1/d。所以紧邻词共现 = 1,距离 5 词的共现 = 1/5。这让近词更重要。

## 关键代码

GloVe 训练 PyTorch 简化版(实际官方实现用 C):

```python
import torch
import torch.nn as nn
from collections import Counter, defaultdict
import math

# Step 1: 构建共现矩阵
def build_cooccurrence(corpus, window=10):
    co_matrix = defaultdict(float)
    for sentence in corpus:
        for i, word_i in enumerate(sentence):
            for j in range(max(0, i-window), min(len(sentence), i+window+1)):
                if i != j:
                    distance = abs(i - j)
                    co_matrix[(word_i, sentence[j])] += 1.0 / distance
    return co_matrix

# Step 2: GloVe 模型
class GloVe(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, x_max=100, alpha=0.75):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.context_embed = nn.Embedding(vocab_size, embed_dim)
        self.bias = nn.Embedding(vocab_size, 1)
        self.context_bias = nn.Embedding(vocab_size, 1)
        self.x_max = x_max
        self.alpha = alpha

    def forward(self, i_ids, j_ids, x_ij):
        """
        i_ids: (B,) 中心词
        j_ids: (B,) 上下文词
        x_ij:  (B,) 共现次数
        """
        v_i = self.embed(i_ids)              # (B, D)
        v_j = self.context_embed(j_ids)      # (B, D)
        b_i = self.bias(i_ids).squeeze()
        b_j = self.context_bias(j_ids).squeeze()

        pred = (v_i * v_j).sum(dim=-1) + b_i + b_j  # (B,)
        target = torch.log(x_ij + 1e-8)

        # 加权 squared loss
        weight = torch.where(
            x_ij < self.x_max,
            (x_ij / self.x_max) ** self.alpha,
            torch.ones_like(x_ij)
        )
        return (weight * (pred - target) ** 2).mean()

# Step 3: 训练
vocab_size = 50000
model = GloVe(vocab_size, embed_dim=300).cuda()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.05)

for epoch in range(50):
    for i_batch, j_batch, x_batch in cooc_loader:
        loss = model(i_batch.cuda(), j_batch.cuda(), x_batch.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 训完后:词向量通常 = (embed + context_embed) / 2(论文建议)
final_vec = (model.embed.weight + model.context_embed.weight) / 2
```

实际使用:直接下载 Stanford 预训练 GloVe:

```python
import numpy as np

def load_glove(path):
    word2vec = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(' ')
            word = parts[0]
            vec = np.array(parts[1:], dtype=float)
            word2vec[word] = vec
    return word2vec

# glove.6B.300d.txt:400K vocab,300 维
vectors = load_glove("glove.6B.300d.txt")
print(vectors["king"][:10])
# → [-0.30, 0.42, ...]
```

## 性能数据

GloVe 论文与 Word2Vec 在 word analogy / similarity / NER 任务上对比:

### Word Analogy(GloVe vs Word2Vec)

| Model | Corpus(tokens)| Semantic | Syntactic | Total |
|------|------|------|------|------|
| ivLBL | 1.5B | 60.0 | 50.1 | 53.2 |
| HPCA | 1.6B | 4.2 | 16.4 | 10.8 |
| GloVe(300d, 6B) | 6B | 77.4 | 67.0 | 71.7 |
| **Word2Vec SG**(6B) | 6B | 73.0 | 66.0 | 69.1 |
| **GloVe**(300d, 42B) | 42B | **81.9** | **69.3** | **75.0** |

GloVe 在 semantic analogy 上比 Word2Vec 略好(77 vs 73),syntactic 上接近。论文强调 GloVe 在大语料(42B / 840B)上扩展性更好。

### Word Similarity(Spearman correlation,越高越好)

| Model | WS353 | MC | RG | SCWS | RW |
|------|------|------|------|------|------|
| Word2Vec SG | 65.6 | 75.4 | 72.4 | 60.7 | 38.5 |
| **GloVe** | **75.9** | **83.6** | **82.9** | **62.9** | **47.8** |

GloVe 在 word similarity 任务上全面胜过 Word2Vec,差距 5-10 个点。

### NER(CoNLL-2003 F1)

| Input | F1 |
|------|------|
| Discrete features only | 84.43 |
| HPCA + features | 85.65 |
| GloVe + features | **88.30** |
| Word2Vec + features | 87.93 |

GloVe 在下游 NER 任务上略优于 Word2Vec。

### Stanford 预训练向量发布版本

GloVe 论文之后 Stanford NLP 发布了几个流行版本:

- **glove.6B.zip**(Wikipedia + Gigaword,6B tokens,400K vocab,50d/100d/200d/300d)
- **glove.42B.300d.zip**(Common Crawl,42B tokens,1.9M vocab)
- **glove.840B.300d.zip**(Common Crawl,840B tokens,2.2M vocab)
- **glove.twitter.27B.zip**(Twitter,27B tokens,1.2M vocab,25d/50d/100d/200d)

这些向量被全球 NLP 研究者下载使用,**glove.6B.300d 在 2014-2018 几乎是所有 NLP baseline 的默认输入**。

## 影响 / 后续

GloVe 在 NLP 历史的位置:**与 Word2Vec 并列为静态词嵌入两大代表,工业 / 学术界事实标配 input**。

**1. Stanford 预训练 GloVe 向量被全球大量使用** —— 2014-2018 年发表的 NLP 论文中,GloVe 是使用最广的预训练词向量。一定意义上 GloVe 的开源策略(预训练 + 论文 + 代码)是 Word2Vec 之外的重要补充

**2. count-based vs prediction-based 路线之争** —— GloVe 论文 + Levy & Goldberg 2014 等工作让"prediction vs counting"成 NLP 圈热议话题。结论:**两条路线本质相同,只是 framing 不同**

**3. 启发后续 embedding 方法** —— GloVe 的"加权矩阵分解"框架被推广到多种场景:
   - LexVec(2016)用 PPMI 加权
   - SPPMI(Levy 2015)Shifted PPMI 分解
   - 跨语言 GloVe(MUSE)

**4. 大规模共现统计计算的工程标杆** —— GloVe 在 840B Common Crawl 上训练,共现矩阵巨大。Stanford 团队的工程实现成为大规模 NLP 数据处理的参考

**5. Manning + Socher 团队的 NLP 影响** —— 论文作者 Manning 是 Stanford NLP 元老,Socher 是 NLP 圈最重要的早期实践者之一,后来创办 MetaMind(被 Salesforce 收购)。GloVe 是 Stanford NLP 组对深度学习 NLP 时代的关键贡献

**6. 仍在某些场景被使用** —— 即使在 BERT/GPT 时代,GloVe 在轻量推理场景(嵌入式设备、实时搜索匹配)、跨语言对齐研究、词嵌入可解释性分析等仍被使用

GloVe 留下的开放问题(由后续工作解答):

- **静态词向量无法处理多义** —— "bank" 河岸 vs 银行 → [ELMo](04-elmo.md) / BERT
- **OOV 词无法处理** —— Common Crawl 预训练再大也覆盖不了所有词 → [FastText](03-fasttext.md)
- **训练目标仍简单** —— 只看 word-word 共现,没有句子 / 文档级语义 → Doc2Vec / Sentence-BERT
- **没有上下文** —— 与 Word2Vec 同样的根本局限 → contextualized embedding 时代

→ [01-word2vec.md](01-word2vec.md) · 兄弟方法,prediction-based 路线
→ [03-fasttext.md](03-fasttext.md) · subword 扩展,处理 OOV
→ [04-elmo.md](04-elmo.md) · 静态到动态的桥梁
→ [../06-bert-family/01-bert.md](../06-bert-family/01-bert.md) · contextualized embedding 集大成
