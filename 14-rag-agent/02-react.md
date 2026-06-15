---
name: "ReAct"
year: 2022
family: "14-rag-agent"
order: 2
paper: "ReAct: Synergizing Reasoning and Acting in Language Models"
authors: ["Shunyu Yao", "Jeffrey Zhao", "Dian Yu", "Nan Du", "Izhak Shafran", "Karthik Narasimhan", "Yuan Cao"]
key_idea: "把 LLM 的推理(Thought)和行动(Action)交错进行,thought 推理下一步要查什么,action 调外部工具,observation 反馈给 LLM 继续推理;Agent 范式的起源"
---

## 前作进展

2022 年中,LLM reasoning 和 LLM 工具使用是两条独立发展的路线:

**Reasoning 一线** —— [Chain-of-Thought](../15-reasoning-o1-r1/01-cot.md)(Wei 2022)发现 prompt 里展示中间推理步骤能大幅提升数学 / 逻辑能力。但 CoT **完全在 LLM 内部跑**,模型没有访问外部信息的能力——遇到"今天比特币价格是多少"这种需要查实时信息的问题,CoT 只能瞎编

**工具使用一线** —— WebGPT(OpenAI 2021)让 GPT-3 学会用搜索引擎查信息,SayCan(Google 2022)让 LLM 调用机器人 API。但这些工作里 **LLM 调工具是单步的**——给个问题,调一个工具,返回答案。复杂多步任务(需要查多次、计算多次)做不了

两个问题摆在那:**纯推理无法接入外部世界,纯工具无法做长链推理**。

Yao 等人(普林斯顿 + Google,2022 年 10 月)的 ReAct 论文给出答案:**把推理和行动交错(synergizing)**。LLM 在每一步:

1. **Thought**:思考下一步要做什么(纯推理)
2. **Action**:调用一个外部工具(查询 / 计算 / 执行)
3. **Observation**:得到工具返回结果,作为新上下文
4. 回到 Thought,直到任务完成

这一框架看起来简单,但它把 **CoT 从"封闭推理"升级为"开放循环"**,启动了整个 agent 范式。论文发表后 6 个月,LangChain 把 ReAct 实现成 framework,2023 年初 AutoGPT 把这条思路推到极致——今天所有"AI agent"产品都建立在 ReAct 之上。

## 核心思想:Thought-Action-Observation 循环

ReAct 的核心模式可以用一个 trace 直观展示:

**任务**:Aurora Borealis 通常什么颜色?这种颜色是由什么粒子的什么过程产生的?

**ReAct trace**:

```
Thought 1: 我需要先查 Aurora Borealis 的颜色
Action 1: Search[Aurora Borealis]
Observation 1: Aurora Borealis(北极光)是天空中绿色 / 红色为主的极光现象...

Thought 2: 主要是绿色。现在我需要查绿色极光是怎么形成的
Action 2: Search[Aurora green color cause]
Observation 2: 绿色极光由氧原子在 100-300 km 高空被太阳风带电粒子激发后...

Thought 3: 我已经有了答案 —— 绿色 + 氧原子激发
Action 3: Finish[绿色,由氧原子被太阳风带电粒子激发]
```

对比纯 CoT 处理同一问题:

```
Thought: Aurora Borealis 通常是绿色,绿色由氢原子激发产生...
Final Answer: 绿色,氢原子激发(❌ 实际是氧原子)
```

CoT 有可能瞎编(hallucination),ReAct 因为每一步都基于实际查到的信息,**幻觉显著下降**。

### 三种 prompting 范式对比

ReAct 论文清晰对比了三种范式:

| Pattern | 形式 | 长处 | 短处 |
|------|------|------|------|
| **Standard** | x → y | 简单快 | 没推理,准确率低 |
| **CoT** | x → thought → y | 推理强 | 无法查外部信息 / 易幻觉 |
| **Act-only** | x → action → obs → y | 能查 | 没推理,不知何时停 |
| **ReAct** | x → (thought → action → obs)* → y | 推理 + 行动结合 | 推理时多消耗 token |

### Few-shot prompt 例子

ReAct 是 prompt 技巧,关键是给 LLM 几个 trace 例子让它模仿。HotpotQA 上的 prompt 模板:

```
Question: <问题>

Thought 1: <对问题的初步分析>
Action 1: <调用工具,如 Search[xxx] 或 Lookup[xxx]>
Observation 1: <工具返回>

Thought 2: <基于 observation 的下一步思考>
Action 2: <下一个调用>
...

Thought N: <已有足够信息,准备答>
Action N: Finish[<答案>]
```

工具集很简单:`Search[entity]` 查 Wikipedia, `Lookup[keyword]` 在当前页面找,`Finish[answer]` 给最终答案。

## 关键代码

最小 ReAct 实现(用 OpenAI API + 几个 tool):

```python
from openai import OpenAI
import wikipedia

client = OpenAI()

def search(query):
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        return f"Error: {e}"

def calculator(expr):
    try:
        return str(eval(expr))
    except Exception as e:
        return f"Error: {e}"

TOOLS = {"Search": search, "Calculator": calculator}

REACT_PROMPT = """
你可以使用以下工具:
- Search[query]: 搜索维基百科
- Calculator[expression]: 计算数学表达式
- Finish[answer]: 给最终答案

按 Thought / Action / Observation 格式作答,直到 Finish。

Question: {question}
"""

def react_agent(question, max_steps=8):
    history = REACT_PROMPT.format(question=question)
    for step in range(max_steps):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": history}],
            stop=["Observation"],
        )
        text = resp.choices[0].message.content
        history += text

        # parse action
        if "Action" not in text:
            break
        action_line = text.split("Action")[-1].split("\n")[0]
        # 例如 ": Search[Tesla 2023 revenue]"
        tool_name = action_line.split("[")[0].strip(": ")
        tool_arg = action_line.split("[")[1].rstrip("]")

        if tool_name == "Finish":
            return tool_arg

        # execute tool
        obs = TOOLS[tool_name](tool_arg)
        history += f"\nObservation {step+1}: {obs}\n"
    return "Max steps reached"

print(react_agent("特斯拉 2023 营收除以同年苹果营收等于多少?"))
```

工业级 ReAct(LangChain 的 AgentExecutor)还会处理:

- **工具调用解析容错** —— LLM 偶尔输出格式错误,需要重试或修正
- **循环检测** —— 防止 LLM 卡在重复 action 里
- **token budget** —— 长对话超出 context window 时压缩历史
- **并行 action** —— GPT-4 之后支持一次输出多个 tool call

## 性能数据

ReAct 论文在 HotpotQA(多跳 QA)和 Fever(事实验证)上对比:

| Method | HotpotQA EM | Fever Acc |
|------|------|------|
| Standard prompt(无推理) | 28.7 | 57.1 |
| CoT(纯推理) | 30.6 | 56.3 |
| Act-only(纯调工具) | 25.7 | 58.9 |
| **ReAct** | **35.1** | **62.0** |
| CoT + Self-Consistency | 33.4 | 60.4 |

关键观察:

- **ReAct 全面胜过 CoT 和 Act-only** —— 1+1>2,推理和行动的协同有效
- **HotpotQA 上提升大** —— 多跳问题需要多次检索,ReAct 天然适合
- **Self-Consistency 没 ReAct 强** —— CoT 多采样投票仍受限于参数化知识

ReAct 在 ALFWorld(文本游戏环境)和 WebShop(网购模拟)上的表现:

| Task | Standard | Imitation Learning | ReAct |
|------|------|------|------|
| ALFWorld(成功率) | 6% | 37% | **71%** |
| WebShop(成功率) | 9% | 29% | **40%** |

ReAct 不微调,纯 prompt 就把 IL 基线打掉一截。这是 agent 范式力量的早期证明。

## 影响 / 后续

ReAct 在 LLM 历史的位置:**Agent 范式的起源,所有后续 agent 工作的基线**。具体影响:

**1. LangChain / LlamaIndex 等 framework 崛起** —— 2022 年 11 月 LangChain 发布,核心 `AgentExecutor` 就是 ReAct 实现。一年内 LangChain GitHub star 突破 60K,成为 LLM 应用开发事实标准

**2. Function Calling 成为 LLM API 标准** —— 2023 年 6 月 OpenAI GPT-4 加入 `function_call` API,把 ReAct 的"调工具"原生化。后来 Anthropic Tool Use、Google Function Calling 都跟进。Function Calling 本质是 ReAct 的工程化

**3. 自主 agent 浪潮** —— ReAct 是 [AutoGPT](04-autogpt.md) / BabyAGI / AgentGPT 等自主 agent 的核心循环。这些项目把 ReAct 推到"无人干预自动完成长任务"的极限

**4. Tool Use 数据集和评估** —— ToolBench / API-Bank / Gorilla 等专门评估 LLM 工具使用能力的 benchmark 出现。逐渐替代纯 QA benchmark

**5. Reflexion / Self-Refine** —— Shinn 2023 在 ReAct 之上加 "self-reflection"(失败后总结教训重试),Madaan 2023 加 "self-refine"(LLM 评估自己输出并改进)。这些都建立在 ReAct loop 之上

**6. 多模态 agent** —— GPT-4V + ReAct 让 LLM 能"看屏幕 + 调工具",催生 WebVoyager / Computer Use(Anthropic 2024)等 agent 操作图形界面的工作

ReAct 留下的开放问题:

- **错误恢复** —— 一旦中间某步出错(检索 miss / 计算错),LLM 难以发现并纠正 → Reflexion 部分解决
- **长 trajectory 的 token 成本** —— 多步 agent 每步都把全历史喂回 LLM,token 消耗 O(N²)
- **泛化到新工具** —— 加新工具要改 prompt,稳定性差 → [Toolformer](03-toolformer.md) 把 tool use 内化为模型能力

→ [03-toolformer.md](03-toolformer.md) · 让 LLM 自监督学会调工具,不靠 prompt
→ [04-autogpt.md](04-autogpt.md) · 把 ReAct 推到自主 agent 极限
→ [01-rag.md](01-rag.md) · RAG 检索可以作为 ReAct 的一个 tool(Agentic RAG)
→ [../15-reasoning-o1-r1/01-cot.md](../15-reasoning-o1-r1/01-cot.md) · ReAct 在 CoT 基础上加 action 维度
→ [../15-reasoning-o1-r1/03-o1.md](../15-reasoning-o1-r1/03-o1.md) · o1 训练目标里包含 ReAct 风格行为
