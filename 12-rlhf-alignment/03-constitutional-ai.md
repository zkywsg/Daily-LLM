---
name: "Constitutional AI"
year: 2022
family: "12-rlhf-alignment"
order: 3
paper: "Constitutional AI: Harmlessness from AI Feedback"
authors: ["Yuntao Bai", "Saurav Kadavath", "Sandipan Kundu", "Amanda Askell", "Jackson Kernion", "Andy Jones", "et al."]
key_idea: "用一套书面原则(constitution)让 AI 自评自身输出,生成 AI feedback 替代人类偏好标注 — 把对齐从'人力密集'压成'算力密集',是 Claude 系列的核心方法"
---

## 前作进展

[InstructGPT](02-instructgpt.md) 证明了 RLHF 是把基础 LLM 对齐到对话助手的可行路径,但暴露了一个明显的可扩展性问题——**对齐高度依赖人工标注**:

- InstructGPT 用了 40 名标注员 6 个月,产出 33K 偏好对,成本 $2-5M
- 训练一个新版本(model update)需要重新标注,因为旧模型的偏好数据可能不再代表新模型的输出分布
- 安全 / 有害性这类**高风险标注**对标注员心理伤害大(看大量有害内容)
- 标注员的偏见会被编码进 reward model 进而进 LLM——这一不透明性让对齐质量难以审计

Anthropic 团队(从 OpenAI 出来的对齐研究核心成员,包括 Dario Amodei、Tom Brown、Jared Kaplan 等)2022 年 12 月发表 *Constitutional AI: Harmlessness from AI Feedback*,提出一个激进想法:**既然 LLM 已经能理解"helpful、harmless、honest"这些原则,为什么不让 LLM 自己按原则评价自己的输出?**

具体方案是用一份**书面 constitution**(一套约 16 条原则)替代人工标注员,让一个 LLM 按 constitution 评价另一个 LLM 的输出,生成 AI 偏好数据用于训练 reward model 和 RL。这一方法叫 **RLAIF(Reinforcement Learning from AI Feedback)**,是 RLHF 的"AI 替代"版本。

Constitutional AI 是 **Claude(2023 年 3 月发布)** 的核心对齐方法。Anthropic 把这套方法论持续推进到 Claude 2 / Claude 3 / Claude 3.5,2024 年的 Claude 3.5 Sonnet 仍在使用 Constitutional AI 的变种。这家族里的 Constitutional AI 节点不仅是一个具体方法,更代表了"AI 替代人类反馈"这条对齐路线。

## 核心思想:两阶段 AI 反馈

Constitutional AI 把 InstructGPT 的"人工 RLHF"流程改造成"AI RLAIF"流程,分两阶段:

```mermaid
graph LR
    sft["SFT 后的 LLM"]:::input --> gen1["生成对有害 prompt 的回答"]:::compute
    gen1 --> critique["AI 按 constitution<br/>批评 + 重写"]:::compute
    critique --> sft_revised["SL-CAI 阶段:<br/>用 revised 回答 SFT"]:::output
    sft_revised --> gen2["生成同一 prompt 的多个回答"]:::compute
    gen2 --> rank["AI 按 constitution<br/>排序回答"]:::compute
    rank --> rlaif["RLAIF 阶段:<br/>RM + PPO 训练"]:::output

    classDef input fill:#fef3c7,stroke:#d97706,color:#92400e;
    classDef compute fill:#fce7f3,stroke:#db2777,color:#9d174d;
    classDef output fill:#ecfdf5,stroke:#059669,color:#065f46;
```

*图 1:Constitutional AI 两阶段——SL-CAI 用 AI 改写有害回答做 SFT,RLAIF 用 AI 排序生成偏好数据做 RM + PPO。两阶段都不需要人类标注有害性。*

**Stage 1: Supervised Learning from AI Critiques(SL-CAI)**——把 SFT 模型从"会回答任何问题"调整到"会拒绝有害请求 + 给出无害替代",流程:

1. 给 SFT 模型一组**有害 prompt**(由 Anthropic red team 收集,典型 ~16K 条),让它生成回答 `y_0`
2. 给同一个模型一个**critique prompt**:"按下面这条 constitution 原则,这个回答有什么问题?" 模型生成批评 `c`
3. 给一个 **revise prompt**:"按你的批评,把回答改成无害版本" 模型生成 `y_1`
4. 用 `(prompt, y_1)` 对原模型做 SFT,得到 SL-CAI 模型

每个 prompt 可以反复 critique-revise 多轮(通常 4 次),得到逐步改进的无害回答。**整个过程不需要人类介入**——constitution 是一次性写好的 ~16 条原则,critique 和 revise 都由 LLM 自己完成。

**Stage 2: Reinforcement Learning from AI Feedback(RLAIF)**——做标准 RLHF,但用 AI 替代人写偏好:

1. 给 SL-CAI 模型一个 prompt,让它生成 2 个候选回答 `(y_A, y_B)`
2. 给 LLM 一个**ranking prompt**:"按 constitution 原则,哪个回答更好?A 还是 B?"
3. LLM 输出 A 或 B,这就是 AI 偏好标注
4. 收集大量 (prompt, y_A, y_B, AI_preference) 三元组训练 reward model
5. 用 PPO + RM 微调 SL-CAI 模型,得到 RL-CAI(最终的 Constitutional AI 模型)

这两阶段完全消除了 InstructGPT 中"40 标注员 6 个月" 的工作量,把对齐成本从**人力 $M 级**压到**算力 $K 级**。

## Constitution 长什么样

Anthropic 在论文里公开了 Constitutional AI 用的 constitution——大约 16 条原则,每条是一句自然语言指南。摘录几条(原版英文):

- *"Choose the response that is most supportive and encouraging of life, liberty, and personal security."*
- *"Choose the response that is least likely to be viewed as harmful or offensive to a non-Western audience."*
- *"Choose the response that is most thoughtful, respectful and cordial."*
- *"Choose the response that has the least objectionable, offensive, unlawful, deceptive, inaccurate, or harmful content."*
- *"Choose the response that least gives the impression of medical authority or expertise, and does not offer medical advice."*

这些原则**故意写得模糊**——没有列举具体的有害类型(暴力、性、毒品等),而是用"non-Western audience"、"medical authority"这种**情境化高层描述**。这有两个考虑:

**1. 让 LLM 自己消化和应用**——具体规则容易被"绕过"(用户找出未列举的边缘 case);高层原则迫使 LLM 学到判断准则而不是查表
**2. 可审计性**——原则用自然语言写,任何人能读、能讨论、能修改。Anthropic 后续多次更新 constitution,过程公开

constitution 的设计本身是一门学问。Anthropic 后来公开 *Collective Constitutional AI*(2024),用公众参与的方式制定 constitution——把"AI 该按什么原则行事"这一决策权从公司内部专家组扩大到更广泛的社会群体。

## RLAIF vs RLHF 性能对比

Constitutional AI 的核心实证问题是:**AI 偏好够不够准确,能不能替代人类偏好?** Anthropic 的对照实验(论文 Table 1)给出明确答案:

| 方法 | 人工评分:helpfulness | 人工评分:harmlessness |
|------|------|------|
| Helpful-only RLHF baseline | **51%**(胜率,vs SFT-only) | -23%(更有害) |
| Standard RLHF(人类反馈,helpful + harmless) | 50% | 0%(基准) |
| **Constitutional AI(RLAIF)** | **51%**(基本持平) | **+9%(更无害)** |

关键观察:**Constitutional AI 在 harmlessness 上超过人类反馈 RLHF,helpfulness 持平**。这意味着 AI 偏好在"安全性评估"上甚至比人类标注员更一致——可能因为:

- AI 不会疲劳、不会有情绪、不会在大量有害内容下表现退化
- AI 应用 constitution 比人类应用心中模糊标准更一致
- 红队 + AI 反馈的迭代可以覆盖更多边缘 case

不过 Constitutional AI 也有几个局限:

**1. helpfulness 上限受 constitution 设计影响**——constitution 没说"要详细回答",AI 就可能给简短回答。OpenAI 的 RLHF helpfulness 更高部分是因为标注员有"详细+有用"的隐式倾向
**2. 自评质量依赖 LLM 本身能力**——constitution 让小模型自评效果差(小模型对 constitution 的理解就有问题)。这一方法需要 base model 至少达到 ~10B 参数才 work 良好
**3. 文化偏见编码进 constitution**——constitution 是人写的,作者价值观会被编码。Anthropic 后来的 Collective Constitutional AI 部分缓解这一问题

## Anthropic 体系:Claude 系列

Constitutional AI 是 Anthropic Claude 系列的核心对齐方法,但 Claude 1/2/3 的实际对齐做了大量工程化扩展:

**Claude 1**(2023 年 3 月)——首次商用部署 Constitutional AI;能力大约对应 GPT-3.5
**Claude 2**(2023 年 7 月)——上下文窗口扩到 100K(后 200K),Constitutional 原则细化,加入更精细的拒绝行为
**Claude 3 / 3.5**(2024)——多模态 + 进一步对齐细化;模型规模和 RLAIF 数据量都显著扩大

Anthropic 的对齐研究也产出了几个独立技术贡献:

- **Helpful, Harmless, Honest(HHH)** 三原则框架——后被整个对齐社区采用
- **Red teaming** 系统化——开发了大规模 red-teaming 流程,主动发现模型失败模式
- **Activation steering / mechanistic interpretability**——理解模型内部如何"决定"对齐行为
- **AI safety research culture** 内部化——Anthropic 把对齐研究放在和能力研究同等重要的位置

## 训练细节

| 维度 | Constitutional AI(论文中 52B 模型) |
|------|------|
| Backbone | Anthropic 内部 LM(规模约 52B,未公开细节) |
| Constitution | 16 条原则,大约 1500 字 |
| 有害 prompt 数据 | ~16K 条 red team prompts |
| SL-CAI 迭代轮数 | 通常 4 轮 critique-revise |
| RLAIF 偏好数据 | ~30K AI 偏好对(对照 InstructGPT 33K 人类偏好) |
| RM 训练 | 同 RLHF 流程,初始化为 SFT 权重 |
| PPO | 同 RLHF 流程,KL + pretrain loss |
| 总成本 | 主要是算力 — 估计 InstructGPT 的 1/10 到 1/5 |

注意 **AI 反馈用的 LLM 和被对齐的 LLM 是同一个模型**(或同一个 base model 的不同 SFT 版本)。这是 RLAIF 的关键设计——**让模型自己当裁判**。听起来递归,但实证上 work,因为 critique/ranking 任务比生成任务对模型能力要求低。

## 关键代码

Constitutional AI 的核心是 critique-revise prompt template:

```python
CRITIQUE_PROMPT = """Below is a harmful or potentially harmful response from an AI assistant.

Conversation:
{conversation}
Response: {response}

Constitutional principle: {principle}

Critique the response according to the principle above. Identify any ways
the response violates or fails to align with this principle.

Critique:"""

REVISE_PROMPT = """Conversation:
{conversation}
Original response: {response}
Critique: {critique}

Rewrite the response to address the critique and better align with the principle.

Revised response:"""

def constitutional_revise(model, conversation, response, constitution):
    """对一个回答做 critique-revise 循环"""
    current = response
    for principle in constitution:  # 遍历 16 条原则
        # Stage A: 让 LLM 批评当前回答
        critique = model.generate(CRITIQUE_PROMPT.format(
            conversation=conversation,
            response=current,
            principle=principle,
        ))
        # Stage B: 让 LLM 基于批评重写
        current = model.generate(REVISE_PROMPT.format(
            conversation=conversation,
            response=current,
            critique=critique,
        ))
    return current  # 返回经过 16 轮原则改进的回答

# RLAIF 偏好生成
def ai_preference_label(model, prompt, response_a, response_b, constitution):
    """让 LLM 在两个回答间选偏好"""
    rank_prompt = f"""Conversation: {prompt}
Response A: {response_a}
Response B: {response_b}

Constitutional principles:
{constitution}

Which response better aligns with the principles?
Answer with just 'A' or 'B'."""
    answer = model.generate(rank_prompt, max_tokens=1)
    return 'A' if 'A' in answer else 'B'
```

工程上的实践要点:

- **constitution 顺序会影响结果**——LLM 对早期原则的关注更多。Anthropic 用随机顺序训练以平均化
- **使用 chain-of-thought**——让 LLM 在选 A/B 之前先推理,准确率显著提升
- **多模型集成可选**——用多个 LLM 投票(比如 Claude-1 + Claude-2 + GPT-4)给偏好数据,质量更高

## 影响 / 后续

Constitutional AI 在 RLHF 历史上的位置:**把对齐从"人力密集"转向"算力密集"的范式转换**。具体影响:

**1. RLAIF 成为对齐研究的活跃方向**——Google 的 *RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback*(2023)系统比较 RLHF vs RLAIF,确认在多个任务上 RLAIF 接近或超过 RLHF;UC Berkeley、CMU 等多个团队跟进发布 RLAIF 改进算法

**2. AI 自评成为对齐工具**——超越 Constitutional AI 本身,"用 LLM 评估 LLM 输出"成为评测的标准做法。MT-Bench、AlpacaEval、Arena 等评测体系都用 GPT-4 当 judge,这是 Constitutional AI 思路的直接延伸

**3. Helpful-Harmless-Honest 框架被广泛采用**——HHH 三原则成为对齐工作的事实标准,后续所有 alignment paper 都按这个分类讨论

**4. Constitution 设计成为研究子方向**——*Collective Constitutional AI*(2024)、*Specific vs Broad Principles*(2024)等论文研究 constitution 该怎么写。这是把"AI 该按什么行事"从工程问题升级为社会-政治问题

**5. 部分混合形态成为商业最佳实践**——大公司通常用 RLHF + RLAIF 混合:核心安全场景用人工标注保证质量,扩展性场景用 AI 反馈降低成本。LLaMA-2-Chat、GPT-4 都有 AI 反馈的成分

**6. 推动 alignment-capability tradeoff 讨论**——Constitutional AI 显示对齐成本可以大幅降低,加速了"对齐是阻碍能力部署的瓶颈"这一争论。Anthropic 的立场是"对齐和能力同等重要、必须同步发展",这一文化深刻影响了 2023+ 的对齐研究方向

Constitutional AI 留下的开放问题:

- **AI 反馈的偏见来源**——base model 的偏见会被放大还是衰减?
- **Constitution 的对抗鲁棒性**——精心设计的 prompt 能否绕过 constitution 判断?
- **多模型 vs 单模型自评**——是否应该用不同 base 的 LLM 互相评估?

→ [04-dpo.md](04-dpo.md) · 算法上的简化,RLAIF + DPO 是当前主流组合
→ [02-instructgpt.md](02-instructgpt.md) · 父方法,Constitutional AI 替代其中的人工标注环节
→ [01-learning-to-summarize.md](01-learning-to-summarize.md) · RLHF 奠基,Constitutional 是其 AI 反馈分支
→ [../07-gpt-scaling/05-gpt4-llama.md](../07-gpt-scaling/05-gpt4-llama.md) · GPT-4 / Claude 3 等前沿模型都用 RLAIF 元素
→ [../15-reasoning-o1-r1/](../15-reasoning-o1-r1/) · process reward model 也是 AI 反馈的一种形式
