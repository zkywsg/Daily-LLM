# RAG 与 Agent

> **让 LLM 突破"参数即知识"和"一问一答"两个根本限制——RAG 把外部知识接进来,Agent 把外部行动接出去。**

## 一句话定位

这家族解决的是 LLM 时代后期的两个根本限制——**(1) 知识全在参数里,过时 / 私有 / 长尾知识无法回答;(2) 一次性输出,无法分步规划 / 调工具 / 自我纠错。** 2020 年 Lewis 等人的 **RAG**(Retrieval-Augmented Generation)给出第一个答案:**把外部知识库接进 LLM 输入侧**——先用 dense retriever(DPR)查相关文档,再让 LLM 基于文档生成答案;模型不再需要"记住所有事实",可以查。2022 年 Yao 等人的 **ReAct** 给出另一半答案:**让 LLM 与外部环境交互**——在 thought(推理)和 action(调工具 / 搜网页 / 执行代码)之间交错,像人一样"想一步 → 做一步 → 看结果 → 再想一步"。2023 年 **Toolformer** 把工具调用变成 LLM 自学能力,模型自己决定何时何处插 `<tool>` 调用;同年 **AutoGPT / BabyAGI** 把 ReAct 思路推到自主 agent 极限,LLM 拿到目标后自己分解任务 / 规划步骤 / 循环执行,引爆 2023 年 "agent 元年"。这家族要回答的问题是:**LLM 怎么从"一个聪明的文本补全器"变成"能查资料、能调工具、能自主完成多步任务的智能体"**。

## 概念本身

RAG 和 Agent 表面看是两条路,本质上都是 **把 LLM 从封闭的"参数函数"打开成"工具使用者"**:

```
封闭 LLM:           x → LLM(θ) → y
RAG:                x → retrieve(KB) → LLM(θ, docs) → y
Agent:              x → LLM ⇄ [search/calc/code/...] → y(经多步)
```

### RAG 的核心

RAG 解决"知识更新 / 私有数据 / 长尾事实"问题。一个最小 RAG 系统三个组件:

```python
# 1. Embed 知识库
docs = chunk(corpus)
embeddings = embed_model(docs)  # 离线
index = vector_db.build(embeddings)

# 2. 查询时检索 top-K
def answer(question):
    q_emb = embed_model(question)
    top_docs = index.search(q_emb, k=5)
    # 3. 拼接上下文 + 生成
    prompt = f"基于以下文档回答问题:\n{top_docs}\n\n问题:{question}"
    return llm(prompt)
```

关键设计点:**chunk 大小 / embedding 模型 / 检索 top-K / 拼接策略 / re-rank**。RAG 看起来简单,工业实现要应对长文档切分、跨段引用、检索召回、上下文窗口限制等问题。

### Agent 的核心

Agent 解决"多步任务 / 工具使用 / 与环境交互"问题。一个最小 ReAct loop:

```
prompt:
  你可以使用工具:[search, calculator, python]
  问题:特斯拉 2023 营收除以同年苹果营收等于多少?

LLM 输出:
  Thought: 我需要先查两家公司 2023 营收
  Action: search("Tesla 2023 revenue")
  Observation: $96.77B
  Thought: 现在查苹果
  Action: search("Apple 2023 revenue")
  Observation: $383.29B
  Thought: 现在计算比值
  Action: calculator(96.77 / 383.29)
  Observation: 0.2524
  Thought: 我已经有答案
  Final Answer: 约 25.2%
```

Agent 的关键是 **LLM 决定下一步动作 → 执行 → 把结果作为新上下文喂回 LLM**,直到任务完成。

### 两条路的融合

2023 年之后 RAG 与 Agent 边界模糊——agent 调的工具里就有"retrieve"(把 RAG 当一个 tool)、RAG 系统也加入 query rewrite / multi-hop 等 agent 行为。今天的"现代 LLM 应用"基本都是 **Agentic RAG**:LLM 同时能查知识 + 调工具 + 多步推理。

围绕这家族的几条主线:

- **从"参数即知识"到"参数 + 检索"**:RAG 起源
- **从"一次输出"到"思考-行动循环"**:ReAct → Toolformer → AutoGPT
- **从"被动 prompt"到"主动规划"**:agent 的任务分解 / 反思 / 多 agent 协作
- **现代演化**:Self-RAG / GraphRAG(自适应检索)、Multi-Agent debate(协作推理)、Computer Use(操作浏览器 / OS)

理解 RAG-Agent 家族 = 理解 LLM 怎么从"chatbot"变成"AI 助手 / copilot / 自主 agent"。今天 90% 的 LLM 生产应用(客服、coding assistant、研究助手、AutoGPT 类应用)都构建在这两个基础范式上。

## 子时间线

| 年份 | 名字 | 关键贡献 | 之前卡在哪 |
|------|------|---------|-----------|
| 2020 | **RAG** | DPR dense retriever + seq2seq 生成,把外部知识库接进 LLM;开放域 QA 不再依赖参数化知识 | 闭书 QA 受限于模型参数容量,过时 / 私有 / 长尾知识无法回答 |
| 2022 | **ReAct** | Thought-Action-Observation 交错框架,LLM 调外部工具 + 推理交替;Agent 范式起源 | CoT 只能纯文本推理,LLM 无法主动查信息 / 调工具 / 验证中间结果 |
| 2023 | **Toolformer** | 让 LLM 自监督学会何时插入工具调用,而不是 prompt 教;tool use 内化为模型能力 | ReAct 依赖 prompt 教模型用工具,稳定性差,新工具要重新调 prompt |
| 2023 | **AutoGPT** | 拿到目标后自动分解 → 规划 → 循环执行,无人干预;自主 agent 范式 | ReAct / Toolformer 仍要人给具体问题,无法做"帮我研究 X 并写报告"这种开放任务 |

## 依赖与延伸

**前置(foundations):**
- [../05-transformer/01-transformer.md](../05-transformer/01-transformer.md) —— LLM backbone
- [../07-gpt-scaling/05-gpt4-llama.md](../07-gpt-scaling/05-gpt4-llama.md) —— Agent 时代靠强 base model 做推理 + 工具调用
- [../15-reasoning-o1-r1/01-cot.md](../15-reasoning-o1-r1/01-cot.md) —— ReAct 是 CoT 的扩展(加 action)
- [../15-reasoning-o1-r1/](../15-reasoning-o1-r1/) —— Reasoning 能力让 agent 更可靠

**通向哪些家族:**
- [../12-rlhf-alignment/](../12-rlhf-alignment/) —— Agent 安全 / 对齐是新研究方向
- [../11-peft-lora/](../11-peft-lora/) —— PEFT 让"在 agent 上做领域微调"成本可控
