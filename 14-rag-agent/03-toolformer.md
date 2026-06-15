---
name: "Toolformer"
year: 2023
family: "14-rag-agent"
order: 3
paper: "Toolformer: Language Models Can Teach Themselves to Use Tools"
authors: ["Timo Schick", "Jane Dwivedi-Yu", "Roberto Dessì", "Roberta Raileanu", "Maria Lomeli", "Luke Zettlemoyer", "Nicola Cantarini", "Edouard Grave", "Thomas Scialom"]
key_idea: "让 LLM 在预训练语料上自监督学习何时何处插入工具调用——给候选位置加 tool call,如果调用后 perplexity 降低就保留;tool use 从 prompt 技巧内化为模型本身能力"
---

## 前作进展

2022 年底,LLM 工具使用主要靠 prompt 教([ReAct](02-react.md) / WebGPT)。这条路有几个工程痛点:

**1. Prompt 教学不稳** —— LLM 偶尔忘记调工具直接答(然后瞎编)、格式错乱、调用参数不对。在 chat 场景里大量精力花在 prompt engineering 上

**2. 新工具要重新调 prompt** —— 加一个工具就要给 LLM 看几个 few-shot 例子,工具多了 prompt 爆炸

**3. 工具调用时机依赖大模型** —— ReAct 在 GPT-3.5+ 能 work,小模型(LLaMA 7B / Mistral 7B)能力不够,prompt 教不会

**4. 调工具会牺牲流畅性** —— 让模型按 "Thought / Action / Observation" 格式输出,自然语言生成质量下降

Schick 等人(Meta AI,2023 年 2 月)的 Toolformer 给出一个完全不同的思路:**不用 prompt 教,让 LLM 自监督学**。核心想法:

> 如果 LLM 在生成 "1969 年人类首次登月" 这句话时,中间插一个 `[Search("first moon landing year")] → 1969` 的调用能让后续 token 预测的 perplexity 降低,那这个工具调用就有用,保留作训练数据。

这一思路把 tool use **从 prompt-level 技巧降到 pretraining-level 能力**。Toolformer 6.7B 模型微调后,在零样本 tool use 上超过 GPT-3 175B——能力被压缩进参数。

## 核心思想:Self-Supervised Tool Learning

Toolformer 的训练 pipeline 三步走:

### Step 1: 采样候选 API 调用位置

给定一段文本 $x = (x_1, ..., x_n)$,对每个位置 $i$ 用 LM 计算"这里插 API 调用的概率"。如果 $P_M(\text{[API call]} | x_1, ..., x_{i-1}) > \tau$,则在位置 $i$ 采样候选 API 调用。

prompt LLM 生成 API 调用 itself:

```
Your task is to add calls to a Question Answering API to a piece of text.

Input: 1969 年人类首次登月。
Output: [QA("When did humans first land on the moon?")] 1969 年人类首次登月。

Input: 法国总统是马克龙。
Output: 法国总统是[QA("Who is the president of France?")] 马克龙。
```

让 LLM 自己生成候选,而不是人工标注——这是 Toolformer 的"自监督"核心。

### Step 2: 执行 API 调用

把每个候选 API call $c_i$ 真的发送给对应工具,得到返回结果 $r_i$。例如 `[QA("When did humans first land on the moon?")] → 1969`。

### Step 3: 过滤——只保留"有用"的调用

关键的 filter 步骤:**只有当插入 API 调用 + 返回结果后,后续 token 的 loss 显著降低,才保留这条样本**。

具体的 loss 比较:

- $L_i^{-} = $ 不带 API 调用时,$x_i, x_{i+1}, ...$ 的加权 cross-entropy
- $L_i^{+} = $ 带 API 调用 + 返回时,$x_i, x_{i+1}, ...$ 的加权 cross-entropy

如果 $L_i^{-} - L_i^{+} > \tau$,保留这条;否则丢弃。

直观理解:**API 调用必须真的帮模型预测后续 token,不然就是"无用的装饰"**。这一 filter 让训练数据自然只保留高质量调用。

### Step 4: 在过滤后数据上微调

把保留的 (text with API call) 数据混合原 pretraining 语料,微调 LM。模型学到:在合适的时机自动插入 `[Tool(args)]` token,然后用返回结果继续生成。

### 工具集

Toolformer 集成 5 个工具:

- **QA** —— 用 Atlas(Meta 的 retrieval QA 模型)
- **Calculator** —— 数学计算
- **Wikipedia Search** —— BM25 检索
- **Machine Translation** —— NLLB 翻译模型
- **Calendar** —— 返回当前日期

每个工具只用 ≤20 个手写例子 prompt 模型生成候选,完全 zero-human-labeling。

## 关键代码

最小 Toolformer 训练循环示意:

```python
def generate_candidate_calls(text, tool_name, prompt_examples, lm):
    """Step 1: 让 LLM 给文本采样候选 API 调用位置"""
    prompt = f"{prompt_examples}\n\nInput: {text}\nOutput:"
    candidates = []
    for pos in range(len(text)):
        # 在位置 pos 看 LM 是否倾向插 API 调用
        prob = lm.token_prob("[", prefix=text[:pos])
        if prob > THRESHOLD:
            call = lm.sample(prompt + text[:pos] + "[")
            candidates.append((pos, call))
    return candidates

def execute_call(call, tool):
    """Step 2: 真的调用工具"""
    return tool(parse_args(call))

def filter_useful(text, pos, call, result, lm):
    """Step 3: 比较带 / 不带 API 调用的 loss"""
    suffix = text[pos:]

    # 不带 API 调用
    loss_minus = lm.cross_entropy(suffix, prefix=text[:pos])

    # 带 API 调用 + 返回值
    augmented_prefix = text[:pos] + f"[{call}→{result}]"
    loss_plus = lm.cross_entropy(suffix, prefix=augmented_prefix)

    return (loss_minus - loss_plus) > FILTER_THRESHOLD

def build_training_data(corpus, tools, lm):
    """Step 1-3 整合"""
    train_data = []
    for text in corpus:
        for tool_name, tool in tools.items():
            for pos, call in generate_candidate_calls(text, tool_name, ...):
                result = execute_call(call, tool)
                if filter_useful(text, pos, call, result, lm):
                    augmented = text[:pos] + f"[{call}→{result}]" + text[pos:]
                    train_data.append(augmented)
    return train_data

# Step 4: 在 train_data 上微调 LM
finetune(lm, train_data)
```

推理时,微调后的 LM 在合适位置自动输出 `[ToolName(args)]`,framework 检测到这个 pattern 就暂停生成、执行工具、把结果填回 prompt、继续生成。**LLM 学到 "调工具" 像学到 "用某个词" 一样自然**。

## 性能数据

Toolformer(基于 GPT-J 6.7B 微调)在零样本任务上的成绩:

| 任务 | GPT-J | GPT-J + CC | GPT-3 175B | OPT 66B | **Toolformer 6.7B** |
|------|------|------|------|------|------|
| LAMA(事实问答) | 17.8 | 19.5 | **31.3** | 22.5 | **29.7**(QA tool) |
| ASDiv(数学) | 7.5 | 9.6 | 14.0 | 6.0 | **40.4**(Calc tool) |
| MathQA | 4.3 | 3.5 | 12.5 | 4.3 | **20.6** |
| SVAMP(数学应用题) | 5.2 | 5.4 | 10.0 | 6.2 | **29.4** |
| MLQA(跨语言 QA) | 8.4 | 7.4 | 25.6 | 14.4 | **20.6**(MT tool) |
| TempLAMA(时序事实) | 13.7 | 12.3 | 23.6 | 14.6 | **16.3** |

关键观察:

- **6.7B Toolformer 击败 175B GPT-3** —— 在数学 / QA 任务上,参数少 25× 但性能反超。**工具使能力比参数容量更重要**
- **数学任务提升最大** —— ASDiv 从 7.5 涨到 40.4(5×),Calculator 工具直接解决了 LLM 算术弱点
- **不会"忘记"基础能力** —— Toolformer 微调后在通用语言建模任务(LAMBADA, Wikitext)上分数不降,工具能力是"加上去"的

## 影响 / 后续

Toolformer 在 LLM 历史的位置:**Tool use 从 prompt-level 升级到 pretraining-level,催生现代 function calling 训练范式**。

**1. 现代 LLM 训练加入 tool use 数据** —— ChatGPT / Claude / Gemini 等闭源模型在 pretraining / SFT 阶段都加入大量 tool use 样本(虽然格式不一定用 Toolformer 论文那套)。LLaMA-3 / Mistral-Large 等开源模型公开承认在训练数据里加 function calling 例子

**2. Function Calling API 工程化** —— 2023 年 6 月 OpenAI GPT-4 推出 `function_call` API,LLM 输出特殊 token 触发函数调用。Anthropic 的 Tool Use API、Google 的 Function Calling 都是同一思路的工程实现

**3. 小模型 tool use 成为可能** —— 之前认为只有 GPT-4 级别才能可靠用工具,Toolformer 证明 6.7B 微调后就能。后续 Gorilla(LLaMA-7B 微调专攻 API 调用)、ToolLLaMA 等小模型 tool 用工作涌现

**4. Function calling benchmark 涌现** —— BFCL(Berkeley Function Calling Leaderboard)、ToolBench、API-Bank 等专门评估模型工具使用能力。成为新一代 LLM 评估维度

**5. Agent + RAG 融合** —— Toolformer 启发"把检索当 tool 调"的设计,现代 Agentic RAG 系统(如 LangGraph)把 retrieval 当作一个工具,LLM 决定何时调

**6. 自监督数据构造范式** —— Toolformer "用 LLM 给自己造训练数据 + perplexity-based filter" 的范式被广泛复用,后续 self-instruct / WizardLM / Magicoder 等指令数据生成都受其启发

Toolformer 留下的开放问题:

- **工具集泛化** —— Toolformer 训练时见过的工具效果好,新工具仍要重训或 prompt。如何做"零样本新工具"是开放问题
- **多工具协同** —— 论文里工具是单调的,真实 agent 需要多工具组合(先 search 再 calculator),需要扩展到序列 tool use → ReAct / AutoGPT 路线
- **错误处理** —— API 调用失败 / 返回错误时 LLM 怎么纠错没解决 → Reflexion / Tree-of-Tools 后续工作

→ [04-autogpt.md](04-autogpt.md) · 自主 agent,把 tool use 推到自动任务分解
→ [02-react.md](02-react.md) · prompt-based tool use 的代表,Toolformer 是其训练化版本
→ [01-rag.md](01-rag.md) · 检索可以作为一个 tool(Toolformer 的 QA tool 就是 RAG)
→ [../11-peft-lora/](../11-peft-lora/) · 微调小模型学 tool use 的实操方法
→ [../07-gpt-scaling/05-gpt4-llama.md](../07-gpt-scaling/05-gpt4-llama.md) · 现代 LLM 训练默认包含 function calling 数据
