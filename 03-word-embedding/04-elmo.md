---
name: "ELMo"
year: 2018
family: "03-word-embedding"
order: 4
paper: "Deep Contextualized Word Representations"
authors: ["Matthew E. Peters", "Mark Neumann", "Mohit Iyyer", "Matt Gardner", "Christopher Clark", "Kenton Lee", "Luke Zettlemoyer"]
key_idea: "用双向 LSTM 预训练语言模型,每个词的向量是 LSTM 各层 hidden state 的加权和;同一个 \"bank\" 在 \"river bank\" 和 \"money bank\" 里向量不同;contextualized embedding 起源,直接催生 BERT"
---

## 前作进展

2013-2017 年,词嵌入家族([Word2Vec](01-word2vec.md) / [GloVe](02-glove.md) / [FastText](03-fasttext.md))成为 NLP 任务标配 input。但所有这些方法有一个**共同根本局限**:**静态词向量**。

```
vec("bank") = [0.2, -0.5, ...]  # 永远固定

在 "I walked along the river bank" 里 → vec("bank") = [0.2, -0.5, ...]
在 "I deposited money at the bank" 里 → vec("bank") = [0.2, -0.5, ...]
```

完全一样!但 "bank" 的两个义项("河岸" vs "银行")应该有不同的向量。这就是 **多义词(polysemy)** 问题。

2014-2017 年大量研究尝试解决:

- **Sense Embedding** —— 每个义项一个向量。但需要词义标注数据 / 难以自动判断有几个义项
- **Topic-Word Embedding** —— 用 LDA 主题区分。粗粒度,不解决细粒度上下文差异
- **context2vec**(Melamud 2016) —— 用双向 LSTM 学上下文向量,但只用 top layer 输出
- **CoVe**(McCann 2017) —— 用机器翻译 encoder 输出作 contextualized representation。受限于翻译数据规模

Peters 等人(AllenAI + UW,2018 年 2 月,NAACL 2018 best paper)的 ELMo(**E**mbeddings from **L**anguage **Mo**dels)给出关键创新:

**1. 预训练双向语言模型** —— 用大语料无监督训练 deep biLM(两层双向 LSTM)
**2. 取所有层的 hidden state** —— 而不只是 top layer
**3. 任务相关的加权组合** —— 不同任务对不同层信息的偏好不同

这一组合让 ELMo 在 6 个 NLP 任务上全面 SOTA,被誉为 "NLP 的 ImageNet 时刻"。

ELMo 之后 6 个月,Google 发布 [BERT](../06-bert-family/01-bert.md),用 Transformer + masked LM 把 contextualized embedding 推到更高水平。**ELMo 是 LSTM 时代的最后辉煌,也是 BERT/GPT 时代的直接前驱**。

## 核心思想:Contextualized Embedding from biLM

### 双向语言模型(biLM)

ELMo 训练一个双向 LSTM 语言模型:

**正向 LM**:从左到右预测下一个词

$$
p(t_1, ..., t_N) = \prod_{k=1}^N p(t_k | t_1, ..., t_{k-1})
$$

**反向 LM**:从右到左预测前一个词

$$
p(t_1, ..., t_N) = \prod_{k=1}^N p(t_k | t_{k+1}, ..., t_N)
$$

联合目标:

$$
\sum_{k=1}^N \log p(t_k | t_{1:k-1}; \Theta_{\text{fwd}}) + \log p(t_k | t_{k+1:N}; \Theta_{\text{bwd}})
$$

注意 ELMo 的"双向"是**两个独立 LSTM**(forward + backward),最后 concat。不像 BERT 那种真正双向的 self-attention(后续 BERT 论文专门指出 ELMo 不是"真双向")。

### 网络结构

```
Input: char-level CNN encoder(类似 FastText subword 思想)
   ↓
biLSTM Layer 1: 4096 hidden + 512 projected, 双向
   ↓
biLSTM Layer 2: 4096 hidden + 512 projected, 双向
   ↓
Output:每个位置每层都输出 hidden state
```

对一个长度 N 的句子,每个位置 $k$ 有:

- $h_k^{LM,0}$ —— input 层(char-CNN 输出)
- $h_k^{LM,1}$ —— LSTM 第 1 层 hidden(forward + backward concat,4096 维)
- $h_k^{LM,2}$ —— LSTM 第 2 层 hidden(4096 维)

总共 **2L+1 = 5 个表示**(L=2 层 LSTM,加 input 层 ×1)。

### Task-specific 加权组合

ELMo 的核心创新:**不固定用哪一层,而是对每个下游任务学习加权**。

$$
\text{ELMo}_k^{\text{task}} = \gamma^{\text{task}} \sum_{j=0}^L s_j^{\text{task}} \cdot h_k^{LM,j}
$$

- $s_j^{\text{task}}$ —— softmax-normalized 权重(对 L+1 层归一化)
- $\gamma^{\text{task}}$ —— 全局 scaling

不同任务学到不同 layer 偏好:

- **POS Tagging / syntactic 任务** —— 偏好底层 LSTM(语法信息)
- **Word Sense Disambiguation / 语义任务** —— 偏好高层 LSTM(语义信息)
- **NER** —— 中间层最重要

### 多义词的 Contextualized 区分

ELMo 论文给的例子:用 ELMo 向量找 "play" 不同义项最近邻:

```
"play(N)的乐器演奏": → nearest: "musical concert", "performance"
"play(V)在比赛中比赛": → nearest: "league match", "fixture"
```

同一个 "play" 在不同上下文里,ELMo 输出的向量明显不同,聚类到对应义项。这是静态词向量(Word2Vec/GloVe/FastText)做不到的。

### 使用方式:作为 input feature

ELMo 不替换原 word embedding,而是**作为额外 feature 拼接**:

```python
input = concat([word2vec_or_glove_embedding, elmo_embedding])
```

这样下游任务可以同时利用预训练静态向量和动态向量。这是 ELMo 与 BERT 的最大使用方式差异——**BERT 是端到端 fine-tune,ELMo 通常是 frozen feature**。

## 关键代码

ELMo 加载与使用(用 AllenNLP 库):

```python
from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# num_output_representations=1 表示输出一层(用学到的权重组合)
elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)

# 输入:list of token list
sentences = [
    ["I", "deposited", "money", "at", "the", "bank"],
    ["I", "walked", "along", "the", "river", "bank"],
]
character_ids = batch_to_ids(sentences)  # (B, seq, 50)
elmo_output = elmo(character_ids)

# elmo_representations: list of (B, seq, 1024) tensor(1024 = 2 × 512 双向)
elmo_vecs = elmo_output["elmo_representations"][0]
print(elmo_vecs.shape)  # torch.Size([2, 6, 1024])

# 同一个 "bank" 在两个句子里的向量
bank_vec_1 = elmo_vecs[0, 5]  # "money at the bank" 里的 bank
bank_vec_2 = elmo_vecs[1, 5]  # "river bank" 里的 bank
similarity = torch.cosine_similarity(bank_vec_1.unsqueeze(0), bank_vec_2.unsqueeze(0))
print(similarity)  # ~0.3-0.5(不像静态向量是 1.0)
```

下游任务集成(典型 NER pipeline):

```python
class NER_with_ELMo(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.glove = nn.Embedding(vocab_size, 300)  # 预训练 GloVe
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0.5)
        # GloVe (300) + ELMo (1024) → concat
        self.lstm = nn.LSTM(300 + 1024, 200, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(400, num_tags)

    def forward(self, word_ids, char_ids):
        glove_vec = self.glove(word_ids)        # (B, seq, 300)
        elmo_vec = self.elmo(char_ids)["elmo_representations"][0]  # (B, seq, 1024)
        x = torch.cat([glove_vec, elmo_vec], dim=-1)  # (B, seq, 1324)
        x, _ = self.lstm(x)
        return self.classifier(x)
```

## 性能数据

ELMo 论文在 6 个主流 NLP 任务上对比 baseline + ELMo:

| Task | Previous SOTA | Baseline | **+ ELMo** | Improvement |
|------|------|------|------|------|
| **SQuAD**(QA F1) | 84.4 | 81.1 | **85.8** | +4.7 |
| **SNLI**(NLI Acc) | 88.6 | 88.0 | **88.7** | +0.7 |
| **SRL**(Semantic Role Labeling F1) | 81.7 | 81.4 | **84.6** | +3.2 |
| **Coref**(Coreference F1) | 67.2 | 67.2 | **70.4** | +3.2 |
| **NER**(F1) | 91.93 | 90.15 | **92.22** | +2.06 |
| **SST-5**(Sentiment Acc) | 53.7 | 51.4 | **54.7** | +3.3 |

关键观察:

- **6 个任务全部 SOTA** —— ELMo 是 2018 上半年的 NLP 全能王
- **平均提升 2-5 个 F1/Acc 点** —— 在 NLP 任务上是巨大改进(很多任务在 90%+ 区间,每点都很难)
- **SRL / SQuAD 提升最大** —— 这些任务需要语义理解,ELMo 的 contextualized 优势最明显

### Layer Importance 分析

ELMo 论文分析不同任务的 layer weight:

| Task | Layer 0(char-CNN)| Layer 1(LSTM 底)| Layer 2(LSTM 顶) |
|------|------|------|------|
| Coref | 0.31 | 0.36 | 0.33 |
| SQuAD | 0.27 | **0.39** | 0.34 |
| SST-5 | 0.34 | 0.34 | 0.32 |
| POS Tagging | **0.49** | 0.30 | 0.21 |
| WSD | 0.25 | 0.30 | **0.45** |

- **POS Tagging 偏好底层**(语法信息浅)
- **WSD 偏好顶层**(语义信息深)
- 大多数任务平均使用各层

这一发现验证了 LSTM 不同层学到不同抽象层次的语言信息,是 ELMo 设计的核心 insight。

## 影响 / 后续

ELMo 在 NLP 历史的位置:**Contextualized Embedding 时代起点,词嵌入家族的终章,BERT/GPT 的直接前驱**。

**1. NLP 的 ImageNet 时刻** —— ELMo 在 6 个任务上同时 SOTA,引发 NLP 圈巨大震动。媒体把 ELMo 称为 "NLP 的 ImageNet 时刻"(类比 AlexNet 2012 在 ImageNet 引爆 CV 深度学习时代)

**2. 直接催生 BERT** —— ELMo 之后 8 个月,Google 发布 BERT(2018.10)。BERT 论文明确把 ELMo 作为主要 baseline,在结构上做关键改进:
   - Transformer 替代 LSTM(更并行)
   - 真正双向(masked LM)替代两个单向拼接
   - 端到端 fine-tune 替代 frozen feature

BERT 几乎全面超越 ELMo,但**没有 ELMo 就没有 BERT 的设计思路**

**3. AllenNLP 库的兴起** —— ELMo 由 AllenAI 发布,配 AllenNLP 库提供端到端 NLP 实验框架。AllenNLP 在 BERT 时代被 HuggingFace transformers 取代,但作为 ELMo 时代的标杆框架影响巨大

**4. 启发"用 LM 预训练学表示"范式** —— ELMo 第一次大规模验证"预训练语言模型 + 下游任务"范式,后续 BERT / GPT / T5 / RoBERTa 全部基于这一思路。**ELMo 是 LLM 时代的真正起点**

**5. 多任务通用表示研究热潮** —— ELMo 启发后续 USE(Universal Sentence Encoder)、InferSent、CoVe 等 sentence-level 通用表示工作

**6. NLP 任务标配从 Word2Vec/GloVe 切换到 ELMo / BERT** —— 2018 年下半年,NLP 论文标配 input 从静态词向量切换到 ELMo / BERT。Word2Vec / GloVe / FastText 退守到资源受限场景

**7. LSTM 时代的终章** —— ELMo 是 LSTM 在 NLP 上的最后辉煌。BERT 之后 LSTM 在 NLP 主流任务上几乎被 Transformer 完全取代。ELMo 是承前启后的关键节点

ELMo 留下的开放问题(由后续工作解答):

- **不是真双向** —— forward + backward 独立训,只在最后 concat → BERT 的 masked LM 是真双向
- **LSTM 慢** —— 不能并行 → Transformer 解决
- **frozen feature 不够灵活** —— 下游任务不能修改 ELMo 参数 → BERT fine-tuning 范式
- **每个任务要单独训 weighted average** —— 工程复杂 → BERT 直接 fine-tune CLS
- **模型规模小**(94M 参数) —— 受限于 LSTM 计算 → BERT-large(340M)、GPT-3(175B)

→ [../06-bert-family/01-bert.md](../06-bert-family/01-bert.md) · 直接继承者,真正引爆 NLP 革命
→ [../06-bert-family/02-roberta.md](../06-bert-family/02-roberta.md) · BERT 的工程化改进
→ [../05-transformer/01-transformer.md](../05-transformer/01-transformer.md) · BERT 用的 Transformer 架构
→ [../02-rnn-lstm/02-lstm.md](../02-rnn-lstm/02-lstm.md) · ELMo 的 LSTM 基础
→ [01-word2vec.md](01-word2vec.md) · 静态词嵌入起源,ELMo 的对照
→ [03-fasttext.md](03-fasttext.md) · ELMo 的 char-CNN 受 FastText subword 思想启发
