# Word Embedding 词嵌入

> **把离散的词变成稠密向量,让"king - man + woman ≈ queen"成为深度学习里的可计算事实——为后来所有 NLP 模型铺好底层语言地基。**

## 一句话定位

这家族解决的是 NLP 时代之前的一个根本难题——**怎么让神经网络处理离散的词**。2013 年之前,NLP 主流是把每个词表示为 one-hot 向量(50000 维,只有一个 1),不同词之间没有任何相似度,模型无法泛化。2013 年 Mikolov 等人的 **Word2Vec** 给出第一个工业级答案:**用 skip-gram / CBOW 通过"预测上下文"的辅助任务**,让网络在 hidden layer 学到 300 维稠密词向量;训完后 `king - man + woman ≈ queen` 等向量算术成立,词向量第一次具备语义结构。2014 年 **GloVe**(斯坦福)从另一个角度切入——直接对全局共现矩阵做加权矩阵分解,理论更优雅,效果与 Word2Vec 接近。2016 年 **FastText**(Facebook)把词拆成 character n-gram(`apple` = `<ap, app, ppl, ple, le>`),既能处理 OOV(未登录词)也能在形态丰富语言(土耳其语 / 芬兰语)上做好。2018 年 **ELMo**(AllenAI)是这家族的终章——用双向 LSTM 给每个词输出 **依赖上下文** 的动态词向量,同一个 "bank" 在 "river bank" 和 "money bank" 里向量不同;ELMo 开启了 "contextualized embedding" 时代,直接通向 BERT。这家族要回答的问题是:**深度学习时代的 NLP 怎么从 one-hot 走到 contextualized representation,3-5 年里完成语言表示的革命**。

## 概念本身

词嵌入(word embedding)的核心是 **把每个词映射到一个低维稠密向量**:

```
one-hot(传统):
  "king" = [0, 0, 0, ..., 1, 0, 0, ..., 0]  (50000 维,1 个 1)
  "queen" = [0, 0, ..., 0, 1, 0, ..., 0]
  两者相似度 = 0

word embedding:
  "king"  = [0.2, -0.5, 0.1, 0.8, ..., -0.3]  (300 维,稠密)
  "queen" = [0.3, -0.4, 0.2, 0.9, ..., -0.2]
  两者相似度 ≈ 0.85(cosine)
```

### 几条主要路线

**1. Prediction-based(预测式)** —— 通过"预测上下文"的辅助任务学词向量。代表:Word2Vec(skip-gram / CBOW)

**2. Count-based(计数式)** —— 直接对共现矩阵(word-word co-occurrence)做矩阵分解。代表:GloVe、LSA

**3. Subword-aware(子词感知)** —— 把词拆成 character n-gram,词向量 = subword 向量和。代表:FastText

**4. Contextualized(上下文相关)** —— 用语言模型给每个词输出依赖上下文的动态向量。代表:ELMo → BERT → GPT

### 几个核心 insight

**1. 分布假设(Distributional Hypothesis)** —— 出自 Firth 1957:"You shall know a word by the company it keeps"。**词的语义由它出现的上下文决定**。这是所有词嵌入方法的哲学基础

**2. 线性结构** —— Word2Vec 论文发现的著名现象:

```
vec("king") - vec("man") + vec("woman") ≈ vec("queen")
vec("Paris") - vec("France") + vec("Italy") ≈ vec("Rome")
```

词向量空间里**线性方向编码了语义关系**(性别、国家-首都、动词时态等)。这是词嵌入的"魔法时刻"

**3. 维度足够低** —— 100-300 维就够,不需要 50000 维 one-hot。低维稠密让深度网络高效学习

**4. 从静态到动态** —— Word2Vec/GloVe/FastText 都是 **静态**(每个词一个向量),ELMo 之后变成 **动态**(向量随上下文变化),这一变化直接通向 BERT 时代

### 与后来 LLM 时代的关系

Word2Vec 时代的"词向量"在 BERT/GPT 之后被 **token embedding + contextual representation** 取代。但词嵌入家族的几个核心思想延续到今天:

- **分布假设** —— GPT 的 next-token prediction 本质上还是分布假设
- **embedding layer** —— 所有现代 LLM 的第一层都是 token embedding,概念上是 Word2Vec 的扩展
- **subword tokenization** —— BPE / WordPiece / SentencePiece 都是 FastText subword 思想的发展
- **线性结构** —— 即使在 GPT-4 时代,LLM 的 embedding 空间仍保留某种语义线性结构

围绕词嵌入演化的几条主线:

- **算法**:Word2Vec(预测式)→ GloVe(计数式)→ FastText(subword)
- **静态到动态**:Word2Vec/GloVe → context2vec → ELMo → BERT
- **覆盖范围**:词级 → subword 级 → 字符级 → byte 级(GPT-2/3)

理解词嵌入家族 = 理解深度学习 NLP 的"史前史",以及今天 LLM embedding layer 的底层逻辑。

## 子时间线

| 年份 | 名字 | 关键贡献 | 之前卡在哪 |
|------|------|---------|-----------|
| 2013 | **Word2Vec** | Skip-gram / CBOW + negative sampling,从大语料自监督学 300 维词向量;king-man+woman=queen 线性算术成立;NLP 进入向量时代 | one-hot 表示稀疏 / 词间无相似度,神经网络 NLP 缺乏可用的词表示 |
| 2014 | **GloVe** | 直接对全局 word-word 共现矩阵做加权 log-bilinear 分解;count-based 路线,理论清晰,效果与 Word2Vec 持平 | Word2Vec 只用局部窗口,没有充分利用全局共现统计信息 |
| 2016 | **FastText** | 词向量 = subword character n-gram 向量和;处理 OOV / 形态丰富语言(土耳其 / 芬兰语)/ 罕见词 | Word2Vec/GloVe 把词当原子,稀有词训不好、新词(OOV)完全不会 |
| 2018 | **ELMo** | 双向 LSTM 预训练 + 给每个词输出依赖上下文的动态向量;同一个 "bank" 在不同句子里向量不同;contextualized embedding 起源 | Word2Vec/GloVe/FastText 都是静态词向量,无法区分多义词("bank" 河岸 vs 银行) |

## 依赖与延伸

**前置(foundations):**
- [../foundations/](../foundations/) —— softmax、negative sampling、SGD 等训练基础
- [../02-rnn-lstm/02-lstm.md](../02-rnn-lstm/02-lstm.md) —— ELMo 基于双向 LSTM

**通向哪些家族:**
- [../06-bert-family/01-bert.md](../06-bert-family/01-bert.md) —— ELMo 直接催生 BERT,contextualized embedding 路线的真正爆发
- [../05-transformer/01-transformer.md](../05-transformer/01-transformer.md) —— Transformer 的 input embedding layer 概念上继承 Word2Vec
- [../07-gpt-scaling/01-gpt1.md](../07-gpt-scaling/01-gpt1.md) —— GPT 的 token embedding + position embedding 是词嵌入思想的扩展
- [../09-multimodal-clip/01-clip.md](../09-multimodal-clip/01-clip.md) —— CLIP 的 text encoder 把词嵌入思想推到 image-text 联合空间
