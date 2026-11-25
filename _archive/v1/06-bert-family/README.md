# 预训练语言模型(BERT 系)

> **用 encoder-only Transformer + 双向 masked LM 学到真正的"上下文敏感词表征",GLUE 全面 SOTA 把 NLP 拖进预训练时代。**

## 一句话定位

这家族解决的是和 [GPT 系](../07-gpt-scaling/) 同一时期(2018)的同一个问题——**怎么让神经网络学到通用语言表征,可以迁移到几乎所有 NLP 下游任务**——但选了完全相反的路线。GPT-1 用 decoder-only + 自回归(从左到右预测下一个 token),BERT 用 encoder-only + 双向 masked LM(随机遮住 token 让模型预测,可以同时看到左右两侧)。这一选择在 2018 年看起来是 BERT 完胜——同一规模下 BERT 在 GLUE / SQuAD 等理解类 benchmark 上全面碾压 GPT-1,把 NLP 社区直接拖进"预训练 + 微调"时代,所有任务从头训成为例外。但 2020 年之后 GPT-3 的 in-context learning 涌现让 decoder-only 路线后来居上,2022 年 ChatGPT 之后 BERT 几乎完全退出主流应用——除了**搜索引擎的语义匹配 / NER 等纯理解任务**,BERT 系仍是工业默认。这家族要回答的问题是:**双向预训练 + 微调这条路线是怎么定型的,以及它的工程优化(RoBERTa)、参数效率(ALBERT)、知识蒸馏(DistilBERT)三个分支演化**。

## 概念本身

BERT 系的核心是 **encoder-only Transformer + 两个自监督预训练任务 + 任务特定微调**。

**架构**——只用 [Transformer](../05-transformer/01-transformer.md) 的 encoder 部分,去掉 decoder。Self-attention 不用 causal mask(每个位置可以看序列里所有位置,**双向**),输入序列前置一个特殊 `[CLS]` token 用于分类任务,句对任务用 `[SEP]` 分隔。

**预训练任务 1:Masked Language Modeling(MLM)**——随机选 15% 的 token 替换成 `[MASK]`,让模型预测被遮的原 token。损失是被遮位置的 cross-entropy:

$$
\mathcal{L}_{\text{MLM}} = -\sum_{t \in \text{Masked}} \log p(x_t | x_{\setminus t})
$$

这一任务的关键性质是**双向上下文**——预测 `[MASK]` 时模型可以同时看到左右两边的 token,这是 GPT 系 causal LM 做不到的。理论上**双向表征对理解类任务(分类、提取、相似度)更有利**——因为这些任务本来就需要看完整个句子再决策。

**预训练任务 2:Next Sentence Prediction(NSP)**——给模型两个句子 `A, B`,让它判断 `B` 是否真的是 `A` 的下一句。这一任务后来被证明几乎没用(RoBERTa / ALBERT 都去掉了或替换了),但 BERT 原版是把它和 MLM 联合训练的。

**微调**——预训练后,给具体任务接一个 task-specific head:

- **句子分类**:`[CLS]` 位置的 hidden state → linear → softmax
- **token 分类(NER 等)**:每个 token 位置的 hidden state → linear → softmax
- **句对任务**:`[CLS]` 位置 hidden state → linear
- **抽取式 QA**:每个 token 输出 start/end 概率

BERT 系的几条演化主线:

- **训练优化**:[RoBERTa](02-roberta.md) 去掉 NSP + 增大 batch + 更长训练 + 动态 masking,证明 BERT 严重训练不足
- **参数效率**:[ALBERT](03-albert.md) 用跨层参数共享 + 因式分解 embedding 压参数 18×,效果接近 BERT-large
- **知识蒸馏**:[DistilBERT](04-distilbert.md) 用 teacher-student 蒸馏,40% 参数 60% 速度保留 97% 性能,工业部署默认
- **位置编码改进**:DeBERTa(2020)给 BERT 加 relative PE + disentangled attention,在 GLUE/SuperGLUE 多次 SOTA(可作为 callout)

BERT 系在 2018–2020 是 NLP 绝对主流(几乎所有论文都从 BERT 起步),2022 之后逐渐让位给 LLM。但它**没有死**——今天的语义搜索(Sentence-BERT / E5 / BGE)、生产 NER、文本分类、检索任务仍然以 BERT 系为骨干,因为 encoder-only 模型在这些任务上比 LLM **快 10-100×、便宜 100×、效果相当**。

## 子时间线

| 年份 | 名字 | 关键贡献 | 之前卡在哪 |
|------|------|---------|-----------|
| 2018 | **BERT** | encoder-only Transformer + masked LM 双向预训练,GLUE 11 任务全面 SOTA,把 NLP 拖进预训练时代 | GPT-1 单向预训练只能看历史;ELMo BiLSTM 双向但 backbone 不能 scale |
| 2019 | **RoBERTa** | 去 NSP + 动态 masking + 大 batch + 10× 数据 + 更长训练,证明 BERT 严重训练不足,GLUE 再涨 5+ 分 | BERT 调参不细 + 训练不足,大量"BERT 改进"其实改的是训练 recipe 而非架构 |
| 2019 | **ALBERT** | 跨层参数共享 + embedding 因式分解,把 BERT-large 参数从 334M 压到 18M 而效果接近;NSP → SOP(句子顺序) | BERT-large 参数大、显存吃,大多数任务上学界部署不起 |
| 2019 | **DistilBERT** | 知识蒸馏:6 层 student 模仿 12 层 teacher,40% 参数 60% 速度保留 97% 性能;工业 BERT 部署默认 | BERT-base 推理慢,生产场景需要更轻的模型 |

## 依赖与延伸

**前置(foundations):**
- [../05-transformer/01-transformer.md](../05-transformer/01-transformer.md) —— BERT 是 Transformer encoder 的第一个大型应用
- [../02-rnn-lstm/05-attention.md](../02-rnn-lstm/05-attention.md) —— attention 机制的起源
- [../07-gpt-scaling/01-gpt1.md](../07-gpt-scaling/01-gpt1.md) —— 同年姊妹工作,decoder-only 路线对照

**通向哪些家族:**
- [../07-gpt-scaling/](../07-gpt-scaling/) —— GPT 系 decoder-only 路线在 2020 后压过 BERT 系成为主流
- [../14-rag-agent/](../14-rag-agent/) —— BERT 系是 RAG 检索环节的主流 encoder(Sentence-BERT、E5、BGE)
- [../11-peft-lora/](../11-peft-lora/) —— LoRA 等 PEFT 方法的早期实验大多在 BERT 系上做
- [../08-vit/](../08-vit/) —— ViT 是 BERT 思想在视觉上的对应版本(patch as token + 类似预训练)
