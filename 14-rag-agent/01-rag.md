---
name: "RAG"
year: 2020
family: "14-rag-agent"
order: 1
paper: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
authors: ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus", "Fabio Petroni", "Vladimir Karpukhin", "Naman Goyal", "Heinrich Küttler", "Mike Lewis", "Wen-tau Yih", "Tim Rocktäschel", "Sebastian Riedel", "Douwe Kiela"]
key_idea: "把 dense retriever(DPR)和 seq2seq 生成器联合训练,把外部知识库接进 LM 输入侧;开放域 QA 不再依赖参数化知识,可以查"
---

## 前作进展

2018-2020 年大模型路线刚起步,主流知识 QA 任务有两条互相平行的路:

**1. 闭书路线(Closed-book QA)** —— 直接把 GPT / T5 等 LM 在 QA 数据上微调,让模型从参数里"回忆"答案。代表工作:Roberts 2020 "How much knowledge can you pack into the parameters of a language model"。在 Natural Questions 上 T5-11B 达到 35% 准确率。问题:**参数容量有限,长尾事实记不住;知识更新要重训;无法引用来源**

**2. 开书路线(Open-book QA / Reading Comprehension)** —— 给定一段文档让模型读后答题。代表工作:BERT-base SQuAD。但要求人工提供"相关文档",真实开放域 QA 没法用

更接近 RAG 的两个前作:

- **DrQA(Chen 2017)** —— 第一个端到端开放域 QA:先用 TF-IDF 检索 Wikipedia,再用 reader 模型抽答案。但 retriever 和 reader 分离训练,误差累积
- **DPR(Karpukhin 2020,同组)** —— Dense Passage Retrieval,用双塔 BERT 把 query 和 passage 编码到稠密向量空间,内积相似度检索。比 TF-IDF / BM25 在准确率上大幅领先

RAG 论文(2020 年 5 月,NeurIPS 2020)把 DPR + 生成式 LM 端到端联合训练,**第一个让"检索 + 生成"成为统一可微框架**——retriever 和 generator 一起训练,retriever 学到"哪些文档对生成有用"。这是后来所有 RAG 系统的鼻祖。

## 核心思想:检索 + 生成的端到端融合

RAG 的核心思想用一个公式可以概括:

$$
p(y \mid x) = \sum_{z \in \text{top-k}(p(\cdot | x))} p_{\eta}(z \mid x) \cdot p_{\theta}(y \mid x, z)
$$

- $x$ —— 输入 query
- $z$ —— 检索出的文档(latent variable)
- $y$ —— 生成的答案
- $p_{\eta}(z|x)$ —— retriever(DPR,参数 $\eta$)
- $p_{\theta}(y|x, z)$ —— generator(BART,参数 $\theta$)

**关键:边缘化掉 $z$,让 retriever 和 generator 都参与梯度更新**。retriever 知道"我应该检索什么样的文档能让 generator 答对",generator 知道"我应该怎么用检索到的文档"。

### 两个变体

论文提了两个变体:

**RAG-Sequence** —— 整个答案序列共用同一个检索文档 $z$:

$$
p(y \mid x) = \sum_z p_{\eta}(z|x) \cdot \prod_i p_{\theta}(y_i \mid x, z, y_{<i})
$$

**RAG-Token** —— 每个 token 可以"看"不同文档:

$$
p(y \mid x) = \prod_i \sum_z p_{\eta}(z|x) \cdot p_{\theta}(y_i \mid x, z, y_{<i})
$$

RAG-Token 更灵活但训练慢,RAG-Sequence 是大多数任务的默认选择。

### 实现细节

- **Retriever**:DPR(双 BERT-base,query encoder + passage encoder),Wikipedia 21M passage,top-K=5
- **Generator**:BART-large(400M),把 query + 检索文档拼接后做 seq2seq
- **Index**:FAISS HNSW,passage embedding 离线建索引
- **训练**:end-to-end 微调 generator(retriever 的 passage encoder 冻结,只训 query encoder)

为什么 passage encoder 冻结?因为重训需要重新对全 Wikipedia 编码,代价巨大。这一工程妥协影响了后来所有 RAG 系统的设计。

## 关键代码

RAG 的极简实现(用 Hugging Face transformers + sentence-transformers):

```python
from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer
import faiss
import numpy as np

# 1. 离线建索引
embedder = SentenceTransformer("facebook/dpr-question_encoder-single-nq-base")
passage_embedder = SentenceTransformer("facebook/dpr-ctx_encoder-single-nq-base")

corpus = load_wikipedia_passages()  # 21M passages
passage_emb = passage_embedder.encode(corpus)  # 离线计算
index = faiss.IndexFlatIP(768)
index.add(passage_emb.astype("float32"))

# 2. 查询时检索 + 生成
def rag_answer(question, k=5):
    q_emb = embedder.encode([question]).astype("float32")
    scores, ids = index.search(q_emb, k)
    top_passages = [corpus[i] for i in ids[0]]

    # 拼接 query + 文档
    context = " ".join(top_passages)
    input_text = f"question: {question} context: {context}"

    tokenizer = BartTokenizer.from_pretrained("facebook/rag-sequence-nq")
    model = BartForConditionalGeneration.from_pretrained("facebook/rag-sequence-nq")
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    output_ids = model.generate(**inputs, num_beams=4, max_length=64)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(rag_answer("Who wrote the novel 'Brave New World'?"))
# → Aldous Huxley
```

工业级 RAG 系统还要加几个组件:

- **Re-rank**:top-K 检索后用 cross-encoder 重排
- **Chunk strategy**:长文档按段落 / 句子切分,保留 overlap
- **Hybrid search**:dense + sparse(BM25)混合,长尾词覆盖更好
- **Query rewrite**:把口语化 query 改成检索友好的形式

## 性能数据

RAG 在主流知识 QA benchmark 上的成绩(EM = Exact Match):

| 模型 | Natural Questions | TriviaQA | WebQuestions |
|------|------|------|------|
| BART(闭书) | 26.5 | 26.7 | 27.6 |
| T5-11B(闭书) | 34.5 | 50.1 | 37.4 |
| DPR + extractive | 41.5 | 56.8 | 41.1 |
| **RAG-Sequence** | **44.5** | **56.8** | **45.2** |
| **RAG-Token** | 44.1 | 55.2 | **45.5** |

关键观察:

- **RAG 用 400M BART 击败 11B T5(闭书)**——参数少 27×,准确率反超 10 个点。验证"查比记好"
- **比 DPR + extractive(抽取式)略好**——生成式可以综合多个文档,extractive 只能选一段
- **WebQuestions 上提升最大**——这数据集事实多,闭书参数化知识容易过时

RAG 的另一关键优势:**knowledge update 不用重训**——更新 Wikipedia index 即可,模型参数不变。

## 影响 / 后续

RAG 在 LLM 历史的位置:**奠定了"参数 + 检索"的混合知识范式,所有现代 LLM 应用的基础**。具体影响:

**1. 工业 LLM 应用标配** —— ChatGPT / Claude / Gemini 等聊天助手都加了 RAG 模块。Microsoft Bing Chat、Perplexity、Google AI Overview 本质都是大规模 RAG 系统

**2. Vector DB 行业崛起** —— RAG 需要高效向量检索,催生 Pinecone / Weaviate / Chroma / Milvus 等 vector database 创业公司。FAISS / ScaNN 等开源库也成主流

**3. LangChain / LlamaIndex 生态** —— 围绕 RAG 的工具链爆发。LangChain 提供 RAG pipeline 编排,LlamaIndex 专注于"用 LLM 索引文档"的高级 RAG

**4. 长上下文 vs RAG 之争** —— 2024 年 Claude 3 / Gemini 1.5 把上下文扩到 1-10M token,引发"既然能塞全文档进 prompt,还要 RAG 吗"的讨论。结论是:**RAG 仍然便宜 / 可控 / 可引用,长上下文是补充而非替代**

**5. Self-RAG / GraphRAG 等高级变体** —— Self-RAG(2023)让 LLM 自己决定"何时检索";GraphRAG(2024,Microsoft)用知识图谱组织文档,multi-hop 检索更强。RAG 演化没停

**6. RAG 评估方法** —— RAGAS / TruLens 等专门评估 RAG 系统的 framework 出现。指标包括 retrieval recall、answer faithfulness、context relevance 等

RAG 留下的开放问题:

- **检索召回上限** —— 当 query 和 doc 不共享词汇时,dense retrieval 仍有 miss(需要 hybrid search)
- **多跳推理** —— "X 的妻子的母校在哪"这种需要两步检索的问题,单次 retrieval 答不出 → multi-hop RAG / GraphRAG
- **检索-生成不一致** —— LLM 有时忽略检索到的文档,凭参数化知识乱答(hallucination on RAG)→ Self-RAG / RAG with citations

→ [02-react.md](02-react.md) · Agent 范式,RAG 经常作为 Agent 的一个 tool
→ [03-toolformer.md](03-toolformer.md) · 把 retrieve 当 tool 内化为 LLM 能力
→ [../15-reasoning-o1-r1/01-cot.md](../15-reasoning-o1-r1/01-cot.md) · CoT 是参数化推理,RAG 是参数外知识,可以组合
→ [../05-transformer/01-transformer.md](../05-transformer/01-transformer.md) · DPR 和 BART 都基于 Transformer
