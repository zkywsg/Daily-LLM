---
name: "BERT"
year: 2018
family: "06-bert-family"
order: 1
paper: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
authors: ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"]
key_idea: "用 encoder-only Transformer + masked LM 学双向上下文表征,GLUE 11 任务全面 SOTA,把 NLP 拖进预训练时代"
---

## 前作进展

2018 年中,NLP 社区已经达成共识:**预训练 + 微调是正确方向**,但具体怎么做还有几条平行路线在竞争:

- **Word2Vec / GloVe**(2013–2014)—— 词向量,但 embedding 是静态的(同一个词在所有上下文里向量相同),且只解决最底层
- **ELMo**(Peters 2018)—— BiLSTM 双向语言模型,产出上下文相关词向量,在多个 GLUE 任务上拿了 SOTA。但 backbone 是 LSTM,无法 scale
- **ULMFiT**(Howard 2018)—— AWD-LSTM + 三阶段微调(语言模型预训练 → 任务特定语言模型微调 → 分类器微调),证明了"完整模型迁移"比"只迁 embedding"更好
- **[GPT-1](../07-gpt-scaling/01-gpt1.md)**(Radford 2018, June)—— decoder-only Transformer + 自回归预训练 + 任务微调,12 任务 9 SOTA

GPT-1 的方法已经很完整,但有一个明显问题:**自回归预训练只能用单向上下文**——模型预测第 `t` 个 token 时只能看到 `1..t-1`,看不到 `t+1..T`。对许多 NLP 理解任务,这是结构性缺陷:

- **句子分类**:看完整个句子再决定情感才合理,只看前半句容易判错
- **命名实体识别**:判断"Apple"是公司还是水果,需要看上下文(后面有"stock"还是"pie")
- **抽取式 QA**:在文档里定位答案 span,前后文都要看
- **自然语言推理**:判断两句话蕴含关系,要看完整两句

GPT-1 的左到右单向预训练对这些任务来说"自缚一手"。ELMo 用 BiLSTM 解决了双向性,但 LSTM 串行且无法 scale 到深层。

Google 团队 2018 年 10 月发表 *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*,给出了"双向 + 可 scale"的解法:**用 Transformer encoder + 一个特殊的 masked language modeling 任务,让模型在预训练时就学到双向上下文表征**。

这一选择把 NLP 推进了一大步。BERT-base 仅 110M 参数,在 GLUE 11 个任务上拿了 SOTA(平均提升 4.5 分);BERT-large(340M)再涨 2 分。在 SQuAD 1.1 上 BERT-large 的 F1 是 93.2,**首次超过人类基准 91.2**。这一波结果直接定义了 2018–2020 的 NLP 研究范式:**所有任务从 BERT 起步**。

## 核心思想:Masked Language Modeling

BERT 的关键创新是**预训练任务的设计**——不是架构层面的(BERT 用的就是标准 Transformer encoder),而是损失层面的:

**Masked Language Modeling(MLM)**:在输入序列里**随机选 15% 的 token 替换成 `[MASK]`**,让模型预测被遮的原 token。例如:

```
Input:    The [MASK] sat on the [MASK].
Output:   The   cat  sat on the   mat.
```

预测 `[MASK]` 时模型可以**同时看到左右两边的上下文**——这是 self-attention 不带 causal mask 自然就有的能力。BERT 用 cross-entropy 损失:

$$
\mathcal{L}_{\text{MLM}} = -\sum_{t \in \text{Masked}} \log p(x_t \mid x_{\setminus \text{Masked}})
$$

**为什么是 15%?** Devlin 团队做了消融:5% 太少(信号不够);30% 太多(模型看到的上下文太残缺,学不到东西)。15% 是质量和效率的折中。后续 RoBERTa 探索过其他比例,最终也确认 15% 接近最优。

**关键 trick:三类 mask**。如果总是把 token 换成 `[MASK]`,**预训练和微调之间存在 distribution mismatch**——微调时输入永远没有 `[MASK]`,模型看不惯。BERT 的解法是把那 15% 选中的 token 分成三种处理:

- **80% 真换成 `[MASK]`**——`my dog is hairy → my dog is [MASK]`
- **10% 换成随机词**——`my dog is hairy → my dog is apple`
- **10% 保持原 token**——`my dog is hairy → my dog is hairy`(但模型仍要在这位置预测)

后两类引入"噪声",让模型不能依赖 `[MASK]` 出现这一信号——必须对**每个**位置都学到"我能不能用上下文重建当前 token"的能力。这一 trick 显著提升下游任务性能,后续所有 MLM 模型都沿用。

## Next Sentence Prediction(NSP)

BERT 的第二个预训练任务,后来被证明几乎没用,但原版有。

**NSP**:训练时给模型两个句子 `(A, B)`,任务是判断 B 是不是真的紧跟在 A 后面。数据构造:50% 时 B 是真实的下一句,50% 时 B 是从语料里随机选的句子。`[CLS]` 位置的 hidden state 接一个二分类头。

Devlin 团队的初衷是希望 NSP 让模型学到"句间关系"——这对 NLI、QA 等多句任务有用。但 2019 [RoBERTa](02-roberta.md) 的系统消融显示**去掉 NSP 反而效果略好**——可能因为 NSP 任务太简单(模型很快学会用浅层信号区分),且占用了 50% 的训练数据预算。

但 BERT 原版的 NSP 至少没有副作用,所以训练时是和 MLM 联合优化:

$$
\mathcal{L}_{\text{BERT}} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}
$$

## 输入表示

BERT 的输入格式(图 2 in paper):

```
[CLS] my dog is hairy [SEP] he likes playing [SEP]
```

每个 token 的输入 embedding 是**三件事相加**:

1. **Token embedding** —— 30K WordPiece 词表
2. **Segment embedding** —— 标记 token 属于句子 A 还是句子 B(为 NSP / 句对任务设计)
3. **Position embedding** —— learned absolute position(同 [GPT-1](../07-gpt-scaling/01-gpt1.md))

**特殊 token**:

- **`[CLS]`** —— 序列开头的"分类 token"。整个序列的语义被聚合到这个位置的最终 hidden state,可以接 head 做句子级分类
- **`[SEP]`** —— 分隔符,放在每个句子末尾。多句任务里用来标记句子边界
- **`[MASK]`** —— 预训练时的遮罩 token,微调时不会出现
- **`[PAD]`** —— 填充 token(用于 batch 等长)

`[CLS]` 这个设计是 BERT 的一个工程亮点——它把"句子级表征"这个任务**显式编码进序列**,模型在预训练时就学到了"如何把整个句子的信息聚合到 `[CLS]` 位置"。这让下游分类任务的接口极其简洁(无脑接 head),也催生了 Sentence-BERT 等用 `[CLS]` 做语义匹配的工作。

## Encoder-only vs Decoder-only:架构选择

BERT 和 [GPT-1](../07-gpt-scaling/01-gpt1.md) 同年发表,但选了完全相反的架构方向。这一选择背后的取舍:

| 维度 | BERT(encoder-only + MLM) | GPT-1(decoder-only + LM) |
|------|------|------|
| Self-attention mask | **双向**(每位置看所有位置) | 单向 causal mask |
| 预训练任务 | MLM(预测遮罩 token) | LM(预测下一 token) |
| 上下文窗口利用率 | 100%(每位置看全文) | 平均 50%(只看左侧) |
| 适合的任务 | 理解类:分类、提取、QA | 生成类:续写、对话 |
| 输出形式 | 每位置一个表征 | 序列生成 |

在 2018 年的 GLUE 上,BERT 完胜——双向上下文对理解类任务是结构性优势。但 5 年后回头看,**decoder-only 路线赢了**——因为它能做生成,这是 LLM 时代的关键能力。BERT 系在 2022 年 ChatGPT 之后逐渐淡出,但 encoder-only 模型在**纯理解任务**(语义搜索、检索、NER)上仍是工业默认——因为这些任务**不需要生成**,encoder-only 模型快 10-100×。

## 性能数据

BERT 在 GLUE benchmark 11 任务上的成绩(论文 Table 1):

| 任务 | 之前 SOTA | BERT-base(110M) | BERT-large(340M) |
|------|------|------|------|
| MNLI(自然语言推理) | 80.6 | **84.6** | **86.7** |
| QQP(问题改写) | 66.1 | **71.2** | **72.1** |
| QNLI(QA 推理) | 84.3 | **90.5** | **92.7** |
| SST-2(情感) | 91.6 | **93.5** | **94.9** |
| CoLA(语法接受度) | 35.0 | **52.1** | **60.5** |
| STS-B(语义相似度) | 78.0 | **85.8** | **86.5** |
| MRPC(改写) | 86.0 | **88.9** | **89.3** |
| RTE(蕴含) | 61.7 | **66.4** | **70.1** |
| **平均** | **75.1** | **79.6** | **82.1** |

BERT-base 平均 +4.5,BERT-large 平均 +7.0。在 GLUE 上 +7 分是当时几年才能达到的进步,BERT **一篇论文把它实现了**。

SQuAD 1.1(抽取式 QA)更震撼:

| 模型 | F1 |
|------|------|
| Previous SOTA(BiDAF + Self-Attn) | 88.5 |
| BERT-base | 90.9 |
| **BERT-large** | **93.2** |
| **人类基准** | 91.2 |

BERT-large 第一个在 SQuAD 上超越人类水平,差距 2 分。

## 训练细节

| 维度 | BERT-large |
|------|------|
| 架构 | 24 层 encoder-only Transformer, d_model=1024, h=16, d_ff=4096, **340M 参数** |
| 上下文窗口 | 512 token |
| Tokenization | WordPiece, 30K 词表 |
| 预训练数据 | BooksCorpus(800M token)+ English Wikipedia(2500M token)= 3.3B token |
| Norm | Post-LN(同 GPT-1) |
| 激活 | GELU |
| 预训练任务 | MLM + NSP 联合 |
| 优化器 | Adam(β1=0.9, β2=0.999), warmup 10K 步 |
| Learning rate | 1e-4,linear warmup 后 linear decay |
| Batch | 256 序列 × 512 token = 128K token / batch |
| 训练步 | 1M steps(BERT-base)/ 1M steps(BERT-large) |
| 训练时间 | BERT-base: 4 days on 4 Cloud TPUs;BERT-large: 4 days on 16 Cloud TPUs |
| Dropout | 0.1 |

BERT-base 是 110M 参数,BERT-large 是 340M——参数是 [GPT-1](../07-gpt-scaling/01-gpt1.md) 的 3 倍(110M)和 3 倍(340M),但训练数据 3.3B token 比 GPT-1 的 0.8B 大 4 倍。**BERT 比 GPT-1 大且训练得更多**——这是它结果更好的部分原因,不全是架构胜出。

## 关键代码

BERT 的核心 block 就是 [Transformer](../05-transformer/01-transformer.md) encoder block(无 causal mask,Post-LN):

```python
import torch
import torch.nn as nn

class BertBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        # Post-LN(原版 BERT)— 没有 causal mask,完全双向
        attn_out, _ = self.attn(x, x, x, key_padding_mask=attention_mask)
        x = self.ln1(x + self.drop(attn_out))
        x = self.ln2(x + self.drop(self.ffn(x)))
        return x
```

MLM 损失实现:

```python
def mlm_loss(model, input_ids, mlm_labels, mask_idx):
    """
    input_ids: [B, T] - 已经把 15% 的 token 按 80/10/10 替换好
    mlm_labels: [B, T] - 被遮位置是原 token,其他位置是 -100(忽略)
    """
    # 模型 forward 得到每个位置的 hidden state
    hidden = model(input_ids)  # [B, T, d_model]

    # MLM head:LayerNorm + GELU + Linear → vocab_size logits
    # 注意 BERT 的 MLM head 用了 weight tying(和 embedding 共享)
    logits = mlm_head(hidden)  # [B, T, vocab_size]

    # 只对被遮位置算 loss(label 不是 -100 的)
    loss = nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        mlm_labels.view(-1),
        ignore_index=-100,
    )
    return loss

def random_mask(input_ids, vocab_size, mask_token_id, mask_prob=0.15):
    """构造 BERT 风格的 80/10/10 masking"""
    labels = input_ids.clone()
    mask = torch.rand(input_ids.shape) < mask_prob  # 15% 选中
    labels[~mask] = -100  # 不参与 loss

    # 选中的 token 里:80% → [MASK], 10% → 随机词, 10% → 不变
    rand = torch.rand(input_ids.shape)
    replace_mask = mask & (rand < 0.8)
    replace_random = mask & (rand >= 0.8) & (rand < 0.9)
    # 剩下 10% 保持原样

    input_ids[replace_mask] = mask_token_id
    input_ids[replace_random] = torch.randint(0, vocab_size, input_ids.shape)[replace_random]
    return input_ids, labels
```

注意 `key_padding_mask` 用来忽略 padding token——这是 BERT 用得起 batch 不等长序列的关键(同 batch 不同长度,padding 部分 attention 被 mask 掉)。

## 影响 / 后续

BERT 在 NLP 历史的地位:**把"预训练 + 微调"从学术想法变成工业默认**。具体几条影响线:

**1. NLP 研究范式定型**——2018 之后所有 NLP 论文从 BERT 开始,做特定任务的"非预训练 baseline"被认为过时。HuggingFace Transformers 库 2019 年发布,围绕 BERT 系建立了完整的工程生态

**2. encoder-only 路线的旗手**——BERT 之后所有理解类任务都用 encoder-only:RoBERTa / ALBERT / DistilBERT / DeBERTa / ELECTRA / XLM-R 都是 BERT 变种。今天的语义搜索、检索、NER、分类仍以这条线为主

**3. masked LM 作为通用预训练任务**——MLM 思想被推广到视觉(MAE, BEiT)、语音(wav2vec, WavLM)、跨模态(BEiT-3),成为自监督学习的标准 paradigm 之一

**4. `[CLS]` 句向量做语义匹配**——Sentence-BERT(2019)用 `[CLS]` 或 mean pooling 把 BERT 改造成句子编码器,在语义检索上击败几乎所有传统方法。今天的 E5、BGE、GTE 等检索模型都是 Sentence-BERT 思路的演化

**5. 2022 之后的"BERT 退场" + "encoder-only 仍存在"双轨**——LLM 时代生成任务全部转向 GPT,但理解任务的成本/性能曲线让 encoder-only 仍是默认。一个语义搜索系统用 BERT-base + FAISS 比用 GPT-4 embedding 便宜 100×、快 100×、效果相当

BERT 留下的几个问题推动了后续节点:

- **训练不足**:1M steps 看似多但其实只过了 33 个 epoch → [RoBERTa](02-roberta.md) 推到 1500B token + 500K steps
- **NSP 没用**:浅层学习信号 → RoBERTa 去掉,ALBERT 换成 SOP
- **参数太大**:BERT-large 340M 部署贵 → [ALBERT](03-albert.md) 参数共享,[DistilBERT](04-distilbert.md) 知识蒸馏
- **绝对 PE 外推差**:同 GPT-1 → DeBERTa 加 relative PE,后来被 [RoPE](../05-transformer/04-rope.md) 推广

→ [02-roberta.md](02-roberta.md) · 训练优化版,证明 BERT 严重 under-trained
→ [03-albert.md](03-albert.md) · 参数共享 + 因式分解,把 BERT-large 压到 18M
→ [04-distilbert.md](04-distilbert.md) · 知识蒸馏,工业 BERT 部署默认
→ [../05-transformer/01-transformer.md](../05-transformer/01-transformer.md) · 父结构,BERT 用 Transformer encoder
→ [../07-gpt-scaling/01-gpt1.md](../07-gpt-scaling/01-gpt1.md) · 同年姊妹工作,decoder-only 路线对照
→ [../14-rag-agent/](../14-rag-agent/) · Sentence-BERT 等是 RAG 检索环节的主流编码器
