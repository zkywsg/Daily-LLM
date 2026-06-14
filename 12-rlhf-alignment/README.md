# 对齐与 RLHF

> **基础 LLM 学到的是"接下来最像人类写的下一个 token";对齐做的是让它"按用户希望的方式回答",这是从语言模型到对话助手的最后一公里。**

## 一句话定位

这家族解决的是 LLM 时代最重要的一个工程问题——**预训练模型能力很强但行为不可控**。GPT-3 在 prompt 工程下能写小说、写代码、做翻译,但同样的 prompt 也能让它生成假新闻、辱骂用户、给出危险建议。直接部署的 LLM 不是"助手",它只是一个"高保真文本分布采样器"——你想要好的输出,它能产出;你不小心引向坏的方向,它一样配合。**对齐**(alignment)这个概念被引入,目标是让模型行为符合人类意图,而不只是模仿训练数据。这条路在 2020 年由 OpenAI 的 *Learning to Summarize* 给出了第一个完整方案——**用人类对模型输出的偏好训练一个 reward model,再用强化学习让模型最大化 reward**(后称 RLHF);2022 年 InstructGPT 把这一流程定型成 SFT → RM → PPO 三阶段,2022 年 11 月发布的 ChatGPT 就是 InstructGPT 在 GPT-3.5 上的部署版,直接引爆了 LLM 进入消费市场。2023 年这家族出现两个重要分支——**Constitutional AI** 用 AI 自动生成偏好数据替代人工(RLAIF),**DPO** 跳过 RL 用监督学习直接优化偏好,大幅简化工程。这家族要回答的问题是:**怎么把一个基础 LLM 变成一个有用、无害、诚实的对话助手**。

## 概念本身

对齐的核心是一个语义层面的目标:**让模型行为与人类意图一致**。但"人类意图"是个模糊概念,工程上要把它转化成可优化的损失函数。RLHF 的解法分三步:

**1. SFT(Supervised Fine-Tuning)**——用人工写的 (prompt, 高质量回答) 数据集对预训练模型做监督微调。这是把"基础 LLM 的随意生成"拉到"接近合格回答"的第一步,但 SFT 只能让模型模仿示例,无法理解"为什么这个回答比那个好"

**2. RM(Reward Model)**——招募标注员对**同一 prompt 的多个模型输出**做两两比较,标注哪个更好。用这些偏好数据训练一个 reward model:`r(x, y)` 给定 prompt `x` 和回答 `y` 输出一个标量分数,**人偏好的回答得分高**。这一步把"人类偏好"这件难表述的事编码成了一个神经网络可计算的标量函数

**3. RL 优化**——把 reward model 当成 RL 环境的 reward 函数,用 PPO 等算法让 LLM 在生成时最大化 `r(x, y)`,同时加 KL 散度惩罚防止偏离原 SFT 模型太远。这一步让模型**主动追求"高分回答"**而不仅模仿示例

这三步的核心创新是**第二步把人类偏好量化**——这是机器学习史上第一次系统化用人类比较数据训练大规模 reward model。它的代价也很高:OpenAI 在 InstructGPT 上雇佣了约 40 名全职标注员,产出 33K 条偏好比较,成本几百万美元。

后续两条简化路线:

**Constitutional AI / RLAIF**——既然人类偏好这件事的本质是"按某些原则评价",那能不能让 AI 自己按一套原则(constitution)给输出打分?Anthropic 2022 年提出 Constitutional AI,用一个 LLM 自评自身输出是否符合"helpful, harmless, honest"原则,生成 AI feedback 替代人类反馈。这把对齐成本从"人力密集"压到"算力密集"

**DPO(Direct Preference Optimization)**——Rafailov 2023 提出的更激进简化:**完全跳过 reward model 和 RL**,把 RLHF 的数学目标推导成一个监督学习损失。直接用 (preferred, rejected) 偏好对训练 LLM,工程上和 SFT 一样简单,效果在多个 benchmark 上接近 PPO。DPO 让 RLHF 从"几个团队才能做"变成了"任何人都能跑"——2024 年开源 LLM 微调几乎全转向 DPO

这家族的本质洞察是:**LLM 的能力来自预训练,行为来自后训练**。预训练教会模型"语言是什么样的",后训练教会模型"应该用什么样的语言"。今天所有部署的 LLM(GPT-4、Claude、Gemini、LLaMA-Chat)都经过对齐——基础模型不会直接给用户用。

## 子时间线

| 年份 | 名字 | 关键贡献 | 之前卡在哪 |
|------|------|---------|-----------|
| 2020 | **Learning to Summarize from Human Feedback** | RLHF 在 NLP 上的第一个完整方案:reward model + PPO 让模型摘要质量超过监督学习 baseline | 监督学习只能学到"人会写什么",学不到"人偏好哪个版本" |
| 2022 | **InstructGPT** | SFT → RM → PPO 三阶段定型;1.3B 对齐版超过 175B 原版,证明对齐胜过 100× 规模;ChatGPT 的直接前身 | GPT-3 能力强但行为不可控,完成任务的可靠性差 |
| 2023 | **Constitutional AI** | 用 AI 按 constitution 自评自身输出(RLAIF),把人工偏好数据成本压到 0,Claude 1/2 的核心方法 | RLHF 依赖大量人工标注,成本和速度都成为瓶颈 |
| 2023 | **DPO** | 跳过 reward model 和 RL,把 RLHF 推导成监督学习损失;工程简单且效果接近 PPO,2024 开源 LLM 默认对齐方法 | PPO 训练不稳、调参困难、需要专门工程团队 |

## 依赖与延伸

**前置(foundations):**
- [../07-gpt-scaling/03-gpt3.md](../07-gpt-scaling/03-gpt3.md) —— GPT-3 是 RLHF 的对齐对象;in-context learning 是 SFT 数据格式的灵感来源
- [../07-gpt-scaling/01-gpt1.md](../07-gpt-scaling/01-gpt1.md) —— 预训练 + 微调范式;RLHF 是这套范式的"对齐版微调"
- `../foundations/03-optimizers/` —— PPO 的优化器细节(advantage 估计、importance sampling 等)

**通向哪些家族:**
- [../15-reasoning-o1-r1/](../15-reasoning-o1-r1/) —— o1/R1 用 RLHF 思路把推理过程作为优化目标(过程奖励)
- [../14-rag-agent/](../14-rag-agent/) —— agent 的工具调用能力依赖对齐后的 LLM(基础模型乱用工具不可控)
- [../11-peft-lora/](../11-peft-lora/) —— LoRA 等 PEFT 方法可以应用在 RLHF / DPO 上降低成本
