---
name: "Word2Vec"
year: 2013
family: "03-word-embedding"
order: 1
paper: "Efficient Estimation of Word Representations in Vector Space / Distributed Representations of Words and Phrases and their Compositionality"
authors: ["Tomas Mikolov", "Kai Chen", "Greg Corrado", "Jeffrey Dean"]
key_idea: "用极简的 shallow network(skip-gram / CBOW)从大语料自监督学 300 维词向量,negative sampling 替代 softmax 让训练扩展到 100 亿词;king-man+woman=queen 线性算术成立,NLP 进入向量时代"
---

## 前作进展

2013 年之前 NLP 用什么表示词?主流是 **one-hot + 手工特征**:

```
"king" = [0, 0, ..., 1, ..., 0]  (50000 维 vocabulary,只有一个 1)
```

问题:

**1. 维度爆炸** —— 词表 5 万 → one-hot 5 万维,内存 / 计算都不友好

**2. 词间无相似度** —— "cat" 和 "dog" 的 one-hot 向量内积 = 0,模型不知道两者相关

**3. 数据稀疏** —— 测试时遇到训练里没出现的词组合,模型完全没法泛化

NLP 社区从 2003 年开始尝试 **分布式表示**(distributed representation):

- **Bengio 2003 NPLM** —— 第一个神经语言模型,中间 hidden layer 的副产品就是词向量。但训练慢(softmax over 50K words 每步 O(NV))
- **Collobert & Weston 2008** —— 用 ranking loss 而不是 likelihood,加速训练
- **LSA / LDA** —— 矩阵分解 / 主题模型,有词表示但不是 task-driven

这些方法理论可行但**没法扩展到 10 亿词以上语料**——计算太慢,效果有限。

Mikolov 等人(Google,2013 年 1 月发第一篇 "Efficient Estimation...",10 月发第二篇 "Distributed Representations...")给出两个关键工程突破:

- **极简架构**(skip-gram / CBOW)—— 砍掉 hidden layer,只保留 input embedding → projection → output
- **negative sampling**(NEG)/ hierarchical softmax —— 替代 full softmax,把每步训练复杂度从 O(V) 降到 O(K) (K 通常 5-20)

这两个 trick 让 Word2Vec 能在**单机 1 天训完 1 亿词**,而前作要 1 周。论文发布后 6 个月,Word2Vec 成为所有 NLP 任务的事实标准 input。**没有 Word2Vec 就没有 2013-2018 年的深度 NLP 兴起**。

## 核心思想:Skip-gram + Negative Sampling

Word2Vec 有两个变体:

### CBOW(Continuous Bag-of-Words)

给定上下文窗口的词,预测中心词。比如句子 "the quick brown fox jumps":

```
context: [the, quick, fox, jumps] → predict: brown
```

### Skip-gram

反过来,给定中心词,预测窗口内的上下文词:

```
center: brown → predict: [the, quick, fox, jumps]
```

实际中 skip-gram 在罕见词上效果更好,是更常用的版本。

### Skip-gram 数学公式

目标:最大化窗口内上下文词的对数概率:

$$
\frac{1}{T} \sum_{t=1}^T \sum_{-c \le j \le c, j \ne 0} \log p(w_{t+j} | w_t)
$$

其中:

$$
p(w_O | w_I) = \frac{\exp(v'_{w_O}{}^T v_{w_I})}{\sum_{w=1}^V \exp(v'_w{}^T v_{w_I})}
$$

- $v_w$ —— 词 w 作为 input 时的"中心词向量"(embedding matrix W)
- $v'_w$ —— 词 w 作为 output 时的"上下文向量"(output matrix W')
- 训完后通常用 $v_w$ 作为词向量

### 关键瓶颈:Softmax 太慢

分母要遍历整个词表 V(5 万词),每步训练 O(V),100 亿词训练根本跑不完。

### Negative Sampling(NEG)

Mikolov 的核心 trick:**不计算真分母,转而做"判别 vs 噪声"的二分类**。

每个正样本 $(w_I, w_O)$,采样 K 个"假"的 negative word $w_{n_1}, ..., w_{n_K}$(从噪声分布 $P_n$ 采样,通常用 unigram^(3/4)),把任务变成:

$$
\log \sigma(v'_{w_O}{}^T v_{w_I}) + \sum_{k=1}^K \mathbb{E}_{w_{n_k} \sim P_n} [\log \sigma(-v'_{w_{n_k}}{}^T v_{w_I})]
$$

直觉:让正样本对 (center, real context) 得高分,K 个负样本对 (center, random word) 得低分。**每步复杂度从 O(V) 降到 O(K)**,K=5-20 通常足够。

NEG 在数学上不再优化原 likelihood,但实测效果同样好,训练快 100×+。

### 线性结构的发现

Word2Vec 论文最震撼的发现:词向量空间里有 **线性语义结构**:

```
vec("king") - vec("man") + vec("woman") ≈ vec("queen")
vec("Paris") - vec("France") + vec("Italy") ≈ vec("Rome")
vec("walked") - vec("walking") + vec("swimming") ≈ vec("swam")
```

这意味着不同语义关系(性别、国家-首都、动词时态)分别对应词向量空间里的某个固定方向。**这是 distributional hypothesis 在向量空间里第一次有可视化、可计算的证据**。

为什么会有线性结构?后续研究(Arora 2016 等)给出理论解释:在 log-bilinear 假设下,词共现统计与词向量的 cosine 相似度成对应关系,语义关系自然对应线性方向。

## 关键代码

最小 Skip-gram + Negative Sampling(PyTorch):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramNeg(nn.Module):
    def __init__(self, vocab_size, embed_dim=300):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)   # 中心词向量
        self.out_embed = nn.Embedding(vocab_size, embed_dim)  # 上下文向量
        # 初始化(论文推荐)
        self.in_embed.weight.data.uniform_(-0.5/embed_dim, 0.5/embed_dim)
        self.out_embed.weight.data.zero_()

    def forward(self, center_ids, pos_context_ids, neg_context_ids):
        """
        center_ids: (B,) 中心词 id
        pos_context_ids: (B,) 正样本上下文词 id
        neg_context_ids: (B, K) 负样本上下文词 id
        """
        v_c = self.in_embed(center_ids)             # (B, D)
        v_pos = self.out_embed(pos_context_ids)      # (B, D)
        v_neg = self.out_embed(neg_context_ids)      # (B, K, D)

        # 正样本 loss
        pos_score = (v_c * v_pos).sum(dim=-1)        # (B,)
        pos_loss = -F.logsigmoid(pos_score)

        # 负样本 loss
        neg_score = torch.bmm(v_neg, v_c.unsqueeze(-1)).squeeze(-1)  # (B, K)
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=-1)             # (B,)

        return (pos_loss + neg_loss).mean()


# 训练流程
vocab_size = 50000
model = SkipGramNeg(vocab_size, embed_dim=300).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# negative sample 分布:unigram^(3/4)
word_freqs = compute_unigram_freqs(corpus)  # (V,)
neg_sample_probs = word_freqs ** 0.75
neg_sample_probs = neg_sample_probs / neg_sample_probs.sum()
neg_sample_probs = torch.from_numpy(neg_sample_probs).float()

for batch in train_loader:
    center, pos_context = batch  # (B,), (B,)
    # 采样 K 个负样本
    neg_context = torch.multinomial(neg_sample_probs, num_samples=len(center)*5, replacement=True)
    neg_context = neg_context.view(len(center), 5).cuda()

    loss = model(center.cuda(), pos_context.cuda(), neg_context)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 训完后取 in_embed 的权重作为词向量
word_vectors = model.in_embed.weight.data.cpu().numpy()
```

实际使用 gensim 一行调用:

```python
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=300, window=5, sg=1,  # sg=1 → skip-gram
                 negative=10, workers=8, epochs=5)
print(model.wv.most_similar("king"))
# → [('queen', 0.85), ('prince', 0.78), ...]
print(model.wv.most_similar(positive=["king", "woman"], negative=["man"]))
# → [('queen', 0.81), ...]
```

## 性能数据

Word2Vec 论文用 Google News 语料(60 亿词)训 300 维词向量,在 word analogy 任务上测试:

### Semantic / Syntactic Analogy(共 19544 个问题)

| Method | Semantic Acc | Syntactic Acc | Total Acc |
|------|------|------|------|
| Collobert & Weston | 9.3% | 12.3% | 11.0% |
| NPLM(Bengio 2003)| 23.0% | 13.0% | 17.4% |
| **Word2Vec CBOW** | **24.0%** | 64.0% | 47.0% |
| **Word2Vec Skip-gram** | **55.0%** | **59.0%** | **57.0%** |

关键观察:

- **Word2Vec Skip-gram 在 semantic analogy 上准确率 55%**,前作不到 25%。这是**第一次词向量在语义关系上有量化效果**
- **CBOW 在 syntactic 上更强**(64 vs 59),Skip-gram 在 semantic 上更强(55 vs 24)
- 训练时间:Skip-gram on Google News 60B tokens,单机 8 core 约 1 天

### 训练效率

Mikolov 2013 论文展示的训练规模 / 时间(2013 年硬件):

| 模型 | 训练数据 | 训练时间 |
|------|------|------|
| NPLM(Bengio)| 1B tokens | 几周 |
| **Word2Vec Skip-gram** | **60B tokens** | **1 天** |

Word2Vec 把训练规模推大 60×,时间反而短 100×+。这种工程效率突破是后来词嵌入广泛使用的根本原因。

### 下游任务

后续(2013-2015)大量 NLP 任务用 Word2Vec 作为 input:

- **Sentiment Classification** —— 准确率提升 3-5 个点
- **POS Tagging** —— 提升 1-2 个点
- **Named Entity Recognition** —— 提升 2-3 个点

虽然提升不算巨大,但 Word2Vec 是 NLP 任务的"标配 input",几乎所有 2014-2018 年的 NLP 论文都用 Word2Vec 或 GloVe 作为 baseline 输入。

## 影响 / 后续

Word2Vec 在 NLP 历史的位置:**深度学习 NLP 时代的起点,所有现代 NLP 模型的"史前史"**。

**1. 词向量成 NLP 任务标配 input** —— 2013-2018 几乎所有 NLP 论文用 Word2Vec / GloVe 作为 input。RNN / LSTM 模型 + 词向量 = 这一时期 NLP 主流架构

**2. 线性结构成研究热点** —— 词向量的语义算术(king-man+woman=queen)启发大量后续研究:
   - 性别 bias 研究(Bolukbasi 2016 "Man is to Computer Programmer as Woman is to Homemaker")
   - Cross-lingual word embedding(用线性变换对齐不同语言词向量)
   - Concept editing(在词向量空间里修改语义)

**3. 训练范式深远影响** —— Skip-gram 的"用预测任务学表示"思想直接催生:
   - Doc2Vec / Sentence2Vec(段落 / 句子表示)
   - Node2Vec / DeepWalk(图节点表示)
   - Item2Vec(推荐系统物品表示)
   - **BERT/GPT 的 self-supervised pretraining** —— 本质都是 Word2Vec 思想的扩展

**4. Negative Sampling 成训练标配** —— NEG 后来被广泛用于推荐系统(Sampled Softmax)、对比学习(SimCLR / CLIP 都用对比 loss)、知识图谱 embedding 等

**5. Mikolov 个人影响** —— 论文一作 Mikolov 后来加入 Facebook AI,继续做 FastText 等工作。Word2Vec 让他成为 NLP 圈最重要的工程师之一

**6. 直接通向 BERT 时代** —— Word2Vec → GloVe → ELMo → BERT 是一条清晰演化路径:
   - Word2Vec(2013):静态词向量
   - ELMo(2018):动态词向量
   - BERT(2018):带 task adaptation 的动态表示
   - GPT-3+(2020+):用 in-context learning 完全替代词向量

Word2Vec 留下的开放问题(由后续工作解答):

- **静态词向量无法处理多义** —— "bank" 河岸 vs 银行 → [ELMo](04-elmo.md) / BERT 解决
- **OOV 词没法处理** —— 训练时没见过的词无向量 → [FastText](03-fasttext.md) 用 subword
- **没有上下文信息** —— Word2Vec 学的是"全局平均"的语义 → contextualized embedding
- **只用局部窗口** —— 忽略全局共现统计 → [GloVe](02-glove.md) 解决

→ [02-glove.md](02-glove.md) · 同期另一条路线,count-based 词嵌入
→ [03-fasttext.md](03-fasttext.md) · subword 扩展,处理 OOV
→ [04-elmo.md](04-elmo.md) · 静态到动态的桥梁,通向 BERT
→ [../06-bert-family/01-bert.md](../06-bert-family/01-bert.md) · contextualized embedding 集大成
→ [../05-transformer/01-transformer.md](../05-transformer/01-transformer.md) · Transformer input embedding layer 概念继承
