---
name: "AutoGPT"
year: 2023
family: "14-rag-agent"
order: 4
paper: "AutoGPT(开源项目,无正式论文);相关综述:A Survey on LLM-based Autonomous Agents (Wang et al. 2023)"
authors: ["Significant Gravitas", "AutoGPT contributors"]
key_idea: "把 ReAct 推到极限——LLM 拿到高级目标后自己分解为子任务、规划执行步骤、循环调工具直到完成,无人干预;启动自主 agent 范式"
---

## 前作进展

2023 年初,LLM 工具使用 / agent 路线发展到一个特殊节点:

- **[ReAct](02-react.md)(2022.10)** 验证了 thought-action-observation 循环
- **GPT-4(2023.3)** 推理能力质变,能可靠地按 ReAct 格式输出几十步
- **LangChain(2022.11)** 把 ReAct 实现成 framework,开发者可以快速搭 agent
- **[Toolformer](03-toolformer.md)(2023.2)** 证明 tool use 可以内化为模型能力

但所有这些工作有一个共同前提:**人给 LLM 一个具体问题**。"特斯拉 2023 营收除以苹果营收"是一个明确问题,LLM 在 ReAct 循环里能答。

如果给一个**模糊高级目标**呢?比如:

> "帮我研究一下电动汽车市场,找出 2024 年三大趋势,写一份 1000 字报告并保存为 markdown 文件。"

ReAct 框架本身处理不了——LLM 不知道"研究"具体指什么、不会自己分解成子任务、不会主动规划"先 search → 再分析 → 再写作 → 再保存"这种长 trajectory。

2023 年 3 月 30 日,一个叫 Toran Bruce Richards(GitHub 名 "Significant Gravitas")的开发者在 GitHub 发布 **Auto-GPT**(后改名 AutoGPT)。核心想法极简:**把 GPT-4 当成一个无限循环的 agent,给它一个高级目标 + 工具集,让它自己想下一步**。

发布后 14 天 GitHub star 突破 100K(GitHub 历史上最快增长项目之一),Twitter 爆发"自主 agent"讨论。**AutoGPT 不是论文,是一个事件**——它把"LLM agent"从研究领域推到主流认知。

同期 BabyAGI(Yohei Nakajima)、AgentGPT、SuperAGI、Microsoft Jarvis(HuggingGPT)等类似项目井喷,2023 年成为 "agent 元年"。

后续学术界出了大量综述(Wang 2023, Xi 2023, Park 2023)系统化这一范式,核心范式被命名为 **autonomous LLM agent**。

## 核心思想:Goal → Plan → Loop

AutoGPT 与 ReAct 的关键区别在两点:**自动任务分解** + **无限循环执行**。

### 整体架构

```
用户输入: 一个高级目标 + 一组 constraint
    ↓
LLM 把目标分解为子任务列表(task queue)
    ↓
循环:
    1. 从 queue 取下一个子任务
    2. LLM 决定要调什么工具(search / browse / file / code / ...)
    3. 执行工具,得到 observation
    4. 把 observation 加入 memory
    5. LLM 评估:这步是否完成?需要新子任务吗?
    6. 更新 task queue(可能加新任务、删完成任务、重排序)
    ↓
直到 task queue 空 / 达到目标 / 用户中止
    ↓
输出最终结果
```

### 关键组件

**1. Task Decomposition**

LLM 接到目标后用类似 prompt:

```
You are AutoGPT. Your goal: <用户目标>

Decompose this goal into a numbered list of concrete sub-tasks.
Output JSON: {"tasks": ["task 1", "task 2", ...]}
```

例如目标"研究电动汽车市场"会被分解为:

```json
{"tasks": [
  "search latest EV market reports 2024",
  "identify top 3 trends from reports",
  "analyze growth data from Q1-Q3 2024",
  "draft 1000-word report",
  "save report as markdown file"
]}
```

**2. Tool Loop(类 ReAct,但更长)**

对每个子任务执行 ReAct 风格 thought-action-observation,但允许嵌套子任务、回溯、任务重排。

**3. Memory**

ReAct 一次会话所有上下文都在 prompt 里,token 很快爆。AutoGPT 引入 **persistent memory**——把执行历史存到 vector DB(Pinecone / Chroma),需要时检索相关记忆喂回 prompt。这一组件让 agent 能跑几小时 / 上千步。

**4. Self-Critique**

每步执行后让 LLM 评估:"这一步做得对吗?需要重做吗?子任务列表要调整吗?" 这是 [Reflexion](https://arxiv.org/abs/2303.11366)(Shinn 2023)正式化的能力,AutoGPT 早期版本已经在用。

**5. Tool Set**

AutoGPT 默认带的工具:

- `web_search` —— Google / DuckDuckGo 搜索
- `web_browse` —— 用 Selenium 打开网页读内容
- `read_file` / `write_file` —— 文件读写
- `execute_python` —— 在 sandbox 执行代码
- `delegate_task` —— 起一个 sub-agent 处理某子任务

可以通过 plugin 加新工具(Wolfram Alpha, Twitter API, ...)。

## 关键代码

最小 AutoGPT-like agent loop:

```python
from openai import OpenAI
client = OpenAI()

class AutoAgent:
    def __init__(self, goal, tools):
        self.goal = goal
        self.tools = tools
        self.task_queue = []
        self.memory = []  # 简化:用 list,真实版用 vector DB

    def decompose(self):
        prompt = f"""
        Goal: {self.goal}
        Decompose into JSON list of sub-tasks.
        Output: {{"tasks": [...]}}
        """
        resp = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        self.task_queue = json.loads(resp.choices[0].message.content)["tasks"]

    def execute_step(self, task):
        # 类 ReAct 单步:LLM 看任务 + memory,决定调哪个工具
        context = "\n".join(self.memory[-10:])  # 最近 10 条记忆
        tool_list = ", ".join(self.tools.keys())
        prompt = f"""
        Goal: {self.goal}
        Current task: {task}
        Recent memory: {context}
        Available tools: {tool_list}

        Output JSON: {{"tool": "<name>", "args": <object>, "done": <bool>}}
        """
        resp = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        decision = json.loads(resp.choices[0].message.content)

        if decision["tool"] in self.tools:
            obs = self.tools[decision["tool"]](**decision["args"])
            self.memory.append(f"{task} → {decision['tool']}({decision['args']}) → {obs[:200]}")
        return decision["done"]

    def critique_and_replan(self):
        # 每隔几步 LLM 检查 task queue 是否合理
        prompt = f"""
        Goal: {self.goal}
        Remaining tasks: {self.task_queue}
        Recent memory: {self.memory[-5:]}

        Are remaining tasks still relevant? Add/remove tasks if needed.
        Output JSON: {{"tasks": [...]}}
        """
        resp = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        self.task_queue = json.loads(resp.choices[0].message.content)["tasks"]

    def run(self, max_steps=50):
        self.decompose()
        for step in range(max_steps):
            if not self.task_queue:
                break
            task = self.task_queue.pop(0)
            done = self.execute_step(task)
            if step % 5 == 4:
                self.critique_and_replan()
        return self.memory  # 最终所有记忆 = 工作结果
```

完整 AutoGPT 实现还要处理 token budget、tool 错误重试、安全检查(不允许 LLM 调危险命令)、UI 等几千行代码。

## 性能数据

AutoGPT 没有正式 benchmark(项目早期完全开源 hack 风格)。后续学术界提出几个 agent benchmark:

| Benchmark | 任务类型 | GPT-4 + ReAct | AutoGPT-style | Human |
|------|------|------|------|------|
| **AgentBench(2023)** | 多领域 agent 任务 | 4.01/10 | 4.42/10 | 7.50/10 |
| **GAIA(2023)** | 通用 AI 助手 | 15% | 30% | 92% |
| **WebArena(2023)** | 网页操作 | 14% | 22% | 78% |
| **SWE-bench(2024)** | 真实代码 issue 修复 | 1.7% | 12.5%(SWE-agent)| - |
| **OSWorld(2024)** | 操作系统任务 | 12% | 18% | 72% |

关键观察:

- **AutoGPT-style 普遍胜过单步 ReAct** —— 长任务里"自动分解 + 反思"有效
- **离 human baseline 还很远** —— GAIA 92% vs AutoGPT 30%。真实开放任务 agent 仍弱
- **特化 agent 突破巨大** —— SWE-agent(Princeton 2024)针对代码修复设计,SWE-bench 从 1.7% 推到 12.5%。**通用 agent 弱,专用 agent 强**

AutoGPT 实际使用的几个观察:

- **token 成本高** —— 一个 1 小时任务可能烧 $5-50 GPT-4 API 费用
- **容易卡死** —— LLM 在子任务里循环失败,task queue 越加越大
- **结果质量参差** —— 简单任务(搜索+总结)效果好,复杂任务(研究+写代码+部署)质量不稳

## 影响 / 后续

AutoGPT 在 LLM 历史的位置:**自主 agent 范式起源,把 "LLM 当 OS 写程序" 的思路推到主流**。具体影响:

**1. Agent 工具链全面爆发** —— 2023 年 4 月之后:LangGraph(LangChain 的状态机 agent 版)、LlamaIndex Agent、CrewAI(多 agent 协作)、Microsoft AutoGen(多 agent 对话)、Microsoft Semantic Kernel 等百花齐放

**2. Multi-Agent 范式** —— AutoGPT 的 `delegate_task` 启发 multi-agent 工作。MetaGPT(2023)用多 agent 模拟软件公司、AutoGen(2023)让 agent 互相对话求解问题、ChatDev(2023)让 agent 角色扮演开发流程

**3. Browser / Computer Use 落地** —— ReAct + AutoGPT 思路结合视觉模型催生"操作 GUI"的 agent。WebVoyager(2024)、Anthropic Computer Use(2024.10)、OpenAI Operator(2025.1)都基于这一谱系——agent 看屏幕 + 鼠标键盘操作

**4. Specialized Agent 成熟** —— SWE-agent / Devin(coding)、ChemCrow(化学)、AI Scientist(自动写论文)等领域专家 agent 出现。比通用 AutoGPT 实用得多

**5. Agent 评估方法学** —— AgentBench / GAIA / WebArena / OSWorld 等 benchmark 标准化 agent 评估。2024-2025 成为 LLM 评估的核心维度

**6. Agent 安全 / 对齐成新问题** —— 自主 agent 在网络上操作真实账户 / 调真实 API 时,prompt injection、misalignment、unauthorized action 等风险被认真讨论。OpenAI / Anthropic 等都建立专门的 agent safety team

**7. 反思路线引爆** —— Reflexion(Shinn 2023)、Self-Refine(Madaan 2023)、Tree of Thoughts(Yao 2023)系统化 AutoGPT 已经在用的"反思"能力。今天先进 agent(o1, Devin)都内置反思

AutoGPT 留下的开放问题:

- **长任务可靠性** —— 越长越容易卡死或走偏,需要更强 planning / verification
- **学习能力** —— 当前 agent 一次任务结束就忘,做不到 "从错误中学" → 持久学习 agent 是开放方向
- **多 agent 协调** —— 多 agent debate 在某些任务上反而比单 agent 差,沟通开销 + miscommunication 是新问题
- **agent + reasoning 模型融合** —— [o1 / R1](../15-reasoning-o1-r1/) 的推理能力 + AutoGPT 的工具能力如何高效结合,是 2025 年的核心研究方向

→ [02-react.md](02-react.md) · 父思想,AutoGPT 是 ReAct 的"自主化"扩展
→ [03-toolformer.md](03-toolformer.md) · 工具使用的训练化路线,agent 工具的底层支撑
→ [01-rag.md](01-rag.md) · agent 的 memory 组件通常用 RAG 实现
→ [../15-reasoning-o1-r1/03-o1.md](../15-reasoning-o1-r1/03-o1.md) · 推理模型 + agent 工具能力是 2025 方向
→ [../12-rlhf-alignment/](../12-rlhf-alignment/) · agent 对齐 / 安全是新挑战
