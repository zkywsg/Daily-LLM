# 推理模型(Test-time Compute)

> **把 LLM 的算力从训练阶段挪到推理阶段:让模型"想得更久"而不是"训得更大",开启 LLM 能力的第二条扩展轴。**

## 一句话定位

这家族解决的是 LLM 时代后期的一个新问题——**纯靠加规模训练已经接近物理 / 经济极限,如何继续提升 LLM 的能力?** 2020-2023 年 GPT-3 → GPT-4 时代的 LLM scaling 主要靠 "加参数 + 加数据 + 加训练算力",[Chinchilla 公式](../07-gpt-scaling/04-scaling-laws.md) 给出了定量预测,但这条路在 2023 年开始遇到天花板——GPT-4 训练耗费 $1 亿,继续 10× 规模代价不可承受。2022 年 Wei 等人的 **Chain-of-Thought(CoT)** 论文给出了完全不同的思路:**让模型在 prompt 里输出推理过程**,数学题准确率从 17% 涨到 78%,**几乎免费就能让现有模型变强**。这一观察启动了 "test-time compute"(测试时算力)路线——能力可以通过"想得更久"而不是"训得更大"获得。2022 年的 **Self-Consistency** 把它推到 sampling 维度;2024 年 OpenAI 的 **o1** 把"长链推理"从 prompt 技巧变成训练目标,**用 RL 让 LLM 自己学到分步思考**,在 IMO 数学竞赛、PhD 级科学问答上击败 GPT-4 多倍;2025 年 1 月 **DeepSeek-R1** 开源了 o1 风格的推理模型,证明纯 RL(GRPO)训练无需 SFT cold start 也能涌现"反思 / 回溯 / 自验证"等推理行为。这家族要回答的问题是:**从 CoT prompt 到 o1 / R1 时代,LLM 推理能力是怎么从"涌现现象"变成"可训练能力"的**。

## 概念本身

推理(reasoning)在 LLM 语境里特指**多步逻辑链** —— 不是一步从 prompt 到答案,而是"问题 → 中间步骤 1 → 中间步骤 2 → ... → 答案"。这一能力的几个层次:

**Level 0: 直接回答** —— `Q: 23 × 47 = ? A: 1081`。GPT-3 时代 LLM 默认行为,数学 / 推理类问题准确率低

**Level 1: Few-shot CoT** —— Prompt 里展示几个"问题 + 推理过程 + 答案"的例子,模型模仿格式生成自己的推理。Wei 2022 发现 8-shot 例子让 GSM8K 数学题准确率从 17% 涨到 60%+

**Level 2: Zero-shot CoT** —— 仅加 "Let's think step by step" 触发模型自己生成推理。Kojima 2022 发现这一 trick 在 SOTA LLM 上效果接近 few-shot

**Level 3: Self-Consistency** —— 采样多条推理路径,投票选最一致答案。Wang 2022 把 GSM8K 从 60% 推到 75%+

**Level 4: o1-style RL 训练** —— 直接训练 LLM 在内部输出长链推理,不需要 prompt 触发。模型学到"反思 / 回溯 / 自验证 / 路径搜索"等高级推理行为。OpenAI o1(2024)、DeepSeek-R1(2025)代表这一层次

这家族的核心 insight 是 **test-time compute = 新的 scaling 轴**:

- **传统轴**:训练时 N(参数)× D(数据)× C(算力) → 训出更强 base model
- **新轴**:推理时让模型"输出更多 token"(thinking tokens)→ 更准确答案

OpenAI o1 论文里有一张关键图:**模型在测试时的"思考 token 数"按对数线性关系决定最终准确率**。这意味着——**给同一个 base model 5 倍推理算力,准确率可以提升等同于训练 10 倍参数**。这是 LLM 时代的根本范式转变。

围绕这一思想的几条演化主线:

- **从 prompt 到训练**:CoT(prompt 技巧)→ o1(训练目标)
- **思考链的形式**:线性 CoT → tree search(ToT)→ 反思与回溯(o1 风格)
- **奖励模型设计**:从 outcome reward 到 process reward(每步打分)
- **训练算法**:从 SFT 到 RLHF 到纯 RL(GRPO)

理解推理家族 = 理解今天 LLM 能力的天花板从何而来。在 2024-2025 年,推理能力成为 LLM 评估的新维度——MMLU / HumanEval 已经饱和,GPQA / AIME / Math500 / Codeforces 成为新基准,这些 benchmark 上 o1 / R1 系列大幅领先 GPT-4 / Claude 3.5。

## 子时间线

| 年份 | 名字 | 关键贡献 | 之前卡在哪 |
|------|------|---------|-----------|
| 2022 | **Chain-of-Thought** | 在 prompt 里展示"问题 + 推理步骤 + 答案"few-shot 例子,GSM8K 数学题准确率 17% → 60%+;开启 prompt-level reasoning | LLM 直接回答数学/逻辑题准确率低,缺少显式推理过程 |
| 2022 | **Self-Consistency** | 对同 prompt 采样多条推理路径,投票选最一致答案;GSM8K 60% → 75%;早期 test-time scaling | CoT 单次采样有噪,某些步骤错就全错 |
| 2024 | **OpenAI o1** | 把长链推理作为训练目标,RL 让 LLM 自己学反思/回溯/自验证;IMO 数学、PhD 级科学问答击败 GPT-4 多倍;test-time compute 成新 scaling 轴 | CoT 仍是 prompt 技巧,模型本身没学会"长时思考" |
| 2025 | **DeepSeek-R1** | 开源 o1 风格推理模型,GRPO 纯 RL 训练无需 SFT cold start 也能涌现推理行为;reasoning trace 全公开 | o1 闭源 + 训练细节不公开,开源社区无法复现推理能力 |

## 依赖与延伸

**前置(foundations):**
- [../07-gpt-scaling/05-gpt4-llama.md](../07-gpt-scaling/05-gpt4-llama.md) —— LLM backbone,o1/R1 在 GPT-4 / DeepSeek-V3 等基础上做
- [../07-gpt-scaling/04-scaling-laws.md](../07-gpt-scaling/04-scaling-laws.md) —— test-time compute 是 scaling law 的第二条轴
- [../12-rlhf-alignment/](../12-rlhf-alignment/) —— RLHF 是 o1/R1 训练的基础;GRPO 是 PPO 变体
- [../07-gpt-scaling/03-gpt3.md](../07-gpt-scaling/03-gpt3.md) —— CoT 是 in-context learning 的扩展

**通向哪些家族:**
- [../14-rag-agent/](../14-rag-agent/) —— agent 的多步规划本质是推理;推理模型让 agent 更可靠
- [../11-peft-lora/](../11-peft-lora/) —— PEFT 应用到推理模型可降低 RL 训练成本
