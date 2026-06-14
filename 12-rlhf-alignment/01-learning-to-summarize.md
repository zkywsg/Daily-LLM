---
name: "Learning to Summarize from Human Feedback"
year: 2020
family: "12-rlhf-alignment"
order: 1
paper: "Learning to Summarize from Human Feedback"
authors: ["Nisan Stiennon", "Long Ouyang", "Jeff Wu", "Daniel Ziegler", "Ryan Lowe", "Chelsea Voss", "Alec Radford", "Dario Amodei", "Paul Christiano"]
key_idea: "用人工偏好比较训练 reward model + PPO 微调 LLM,摘要质量超过监督学习 baseline 和参考摘要,确立 RLHF 在 NLP 上的完整方案"
---

## 前作进展

到 2020 年初,NLP 文本生成任务(摘要、翻译、对话)的标准做法是 **监督学习 + ROUGE/BLEU 自动指标**——拿 (输入, 参考输出) 数据对训练 Seq2Seq 模型,用 ROUGE 等基于 n-gram 重合度的指标评估。但这条路有几个一直没解决的问题:

**1. 自动指标和人类判断的相关性差**——ROUGE 高的摘要人未必觉得好,反之亦然。多篇论文(Paulus 2017, Schluter 2017)指出 ROUGE 主要测词汇重合,完全忽略事实准确性、流畅度、信息覆盖等真正的质量维度

**2. 监督学习只能模仿,无法超越参考**——训练数据里的"参考摘要"(通常是新闻编辑写的)质量参差不齐,且只代表一种风格。模型学到的是"像这些参考的输出",拿不到比参考更好的能力

**3. 长尾错误难以控制**——监督学习对所有训练样本同等对待,模型仍会犯一些低级错误(事实捏造、重复、跑题),因为这些错误在 NLL loss 上代价不大

这条线的早期 RLHF 尝试:

- **2017 Christiano** *Deep Reinforcement Learning from Human Preferences*——在 Atari 和 MuJoCo 上验证了 RLHF 的可行性:用人类偏好训练 reward model,RL 在 reward model 上优化能让 agent 学到难以用奖励函数表达的复杂行为(如做后空翻)。但这是游戏控制场景,没人证明 NLP 上能用
- **2019 Ziegler** *Fine-Tuning Language Models from Human Preferences*——OpenAI 在情感生成和续写任务上做了 RLHF 早期实验,效果有但任务太小

Stiennon 等人 2020 年 9 月的 *Learning to Summarize from Human Feedback* 是第一个**在标准 NLP benchmark 上系统化 RLHF + 击败监督 SOTA** 的工作。论文核心论点:**用人类偏好直接优化,可以让模型生成质量超过参考摘要本身**——这在监督学习范式下是不可能的。

## 核心思想:三阶段 RLHF 流程

Stiennon 论文把 RLHF 整理成今天 RLHF 的标准三阶段:

```mermaid
graph LR
    pre["预训练 LLM<br/>(GPT-3 1.3B/6.7B)"]:::input --> sft["Stage 1: SFT<br/>(在参考摘要上微调)"]:::compute
    sft --> gen["生成多个候选摘要"]:::compute
    gen --> label["人标注偏好<br/>(64K 对比较)"]:::compute
    label --> rm["Stage 2: 训练 RM<br/>r(x, y) = 标量分数"]:::compute
    rm --> ppo["Stage 3: PPO<br/>max E[r(x,y)] - β KL(π‖π_SFT)"]:::compute
    ppo --> aligned["对齐后的 LLM"]:::output

    classDef input fill:#fef3c7,stroke:#d97706,color:#92400e;
    classDef compute fill:#fce7f3,stroke:#db2777,color:#9d174d;
    classDef output fill:#ecfdf5,stroke:#059669,color:#065f46;
```

*图 1:RLHF 完整三阶段——SFT 拉到合格区、RM 把人类偏好编码成标量、PPO 让 LLM 优化这个标量同时受 KL 约束不偏离 SFT 模型。*

**Stage 1: SFT(Supervised Fine-Tuning)** —— 在标准摘要数据集(Reddit TL;DR、CNN/DailyMail)上微调预训练 LLM。这一步让模型学会"摘要应该是什么形式":短、覆盖关键信息、自然语言。SFT 后模型已经能生成像样的摘要,但质量参差不齐。

**Stage 2: 训练 Reward Model** —— 给定 prompt `x`,让 SFT 模型生成 4 个候选摘要 `(y_1, y_2, y_3, y_4)`。标注员从中两两比较,标注哪个更好。用这些偏好对训练 reward model `r_φ(x, y)`,目标是让 RM 的打分与人类偏好一致。损失用 Bradley-Terry 模型的负对数似然:

$$
\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]
$$

`y_w` 是 winning(被偏好的),`y_l` 是 losing。`σ` 是 sigmoid。意思是:**RM 给 winner 的分要明显高于 loser**。Stiennon 团队收集了 64K 对比较,训练了一个和 SFT 模型同规模(1.3B 或 6.7B)的 reward model。

**Stage 3: PPO 强化学习** —— 把 RM 当 reward 函数,让 LLM 用 PPO 算法优化:

$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta \cdot \text{KL}(\pi_\theta(\cdot|x) \,\|\, \pi_{\text{SFT}}(\cdot|x)) \right]
$$

第一项是 reward,模型想要高分;第二项是 **KL 惩罚**,防止 `\pi_\theta` 偏离 SFT 模型太远。`β` 控制偏离程度,典型 `β = 0.01–0.1`。这一项极其关键——**没有它,模型会学到"hack reward model"**,生成对 RM 评分高但实际乱码的输出。

## 为什么需要 PPO?

PPO(Proximal Policy Optimization, Schulman 2017)是 OpenAI 自家的 RL 算法,在 RLHF 里被选用有具体理由:

**1. 离线训练 + 多次更新**——PPO 用 importance sampling 让"采样一次数据可以更新多次",对 LLM 这种 forward 极贵的场景非常友好

**2. clip 防止 policy 跳变过大**——PPO 的核心 trick 是 `clip(r, 1-ε, 1+ε)` 把策略更新比例限制在 `[1-ε, 1+ε]`(典型 `ε = 0.2`)。这避免了 LLM 一步更新就崩(catastrophic forgetting),配合 KL 惩罚双重保险

**3. 简单 + 稳定**——相比 TRPO 等更严格的策略梯度方法,PPO 实现简单、调参少、在大多数任务上稳定

但 PPO 在 LLM 上仍是工程恶梦:**需要 4 个模型同时在显存里**(actor / critic / reward model / reference SFT model),训练慢、显存爆、调参困难。这是 2023 年 DPO 出来后社区集体转向 DPO 的根本原因(详见 [04-dpo.md](04-dpo.md))。

## 性能:超过参考摘要

Stiennon 论文最震撼的结果是 **RLHF 后的模型质量在人工评测中超过了参考摘要**(论文 Figure 1):

| 模型 | 人工评分(% 时间被偏好 vs 参考摘要) |
|------|------|
| Reddit TL;DR 数据集自带参考 | 50%(基准) |
| 监督微调(1.3B,SFT-only) | 35% |
| 监督微调(6.7B,SFT-only) | 41% |
| 人类写的高质量摘要 | 70% |
| **RLHF(1.3B)** | **62%** |
| **RLHF(6.7B)** | **74%** |

关键观察:

- **1.3B RLHF(62%)超过 6.7B SFT(41%)**——对齐胜过 5× 参数。这是后来 InstructGPT "1.3B 对齐版超过 175B 原版" 的预告
- **6.7B RLHF(74%)甚至超过人类写的摘要(70%)**——这是 LLM 第一次在文本生成任务上明确"超人"。当然评测有偏(人工评分员可能偏好流畅多于准确),但 trend 是清晰的
- **RLHF 在 ROUGE 上反而比 SFT 略低**——再次确认 ROUGE 不反映真实质量

## 训练细节

| 维度 | Learning to Summarize(6.7B 版) |
|------|------|
| Backbone | GPT-3 1.3B / 6.7B,decoder-only Transformer |
| Stage 1 SFT | 在 Reddit TL;DR + CNN/DM 上微调 |
| Stage 2 RM | 收集 64K 对比较;RM 初始化为 SFT 模型权重 + 一个 scalar head |
| RM 训练 | 1 epoch,lr 1e-5,batch 64 |
| Stage 3 PPO | KL 惩罚系数 β 自适应调整(若 KL > 目标值则增大 β) |
| PPO 训练步 | ~10K updates,每 update 用 ~512 prompt 采样 |
| 硬件 | 8 × V100 GPU(SFT 和 RM 是天级别,PPO 是周级别) |
| 标注成本 | 64K 对 × ~$0.20 / 对 ≈ $13K(论文报告) |

注意 **RM 初始化为 SFT 模型权重 + 加一个 scalar head**——这是后来标准做法。RM 不从随机初始化,因为 SFT 已经学到"摘要长什么样"的表征,RM 只需要在上面学"哪个更好"的标量映射。

## 关键代码

RM 训练的核心:Bradley-Terry loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    def __init__(self, base_lm):
        super().__init__()
        self.lm = base_lm  # 预训练 LM(SFT 后的权重)
        # 一个 scalar head:把最后 token 的 hidden state 映射到 reward 值
        self.head = nn.Linear(base_lm.config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        # 用 LM 抽特征,取最后非 padding token 的 hidden state
        hidden = self.lm(input_ids, attention_mask=attention_mask).last_hidden_state
        last_idx = attention_mask.sum(-1) - 1
        last_hidden = hidden[torch.arange(hidden.size(0)), last_idx]
        return self.head(last_hidden).squeeze(-1)  # [B]

def rm_loss(rm, prompt, win_resp, lose_resp):
    """Bradley-Terry: 让 winner 的 reward 显著高于 loser"""
    r_win = rm(*encode(prompt, win_resp))
    r_lose = rm(*encode(prompt, lose_resp))
    # 相当于二分类:p(win > lose) = sigmoid(r_win - r_lose)
    return -F.logsigmoid(r_win - r_lose).mean()
```

PPO 阶段的核心思路(简化):

```python
def ppo_step(actor, critic, ref_model, reward_model, prompts):
    """单步 PPO 更新"""
    # 1. 当前 policy 生成回答
    responses = actor.generate(prompts)
    log_probs_old = actor.log_probs(prompts, responses).detach()

    # 2. 算 reward:RM 给的分 - KL 惩罚
    rewards = reward_model(prompts, responses)
    log_probs_ref = ref_model.log_probs(prompts, responses)
    kl = (log_probs_old - log_probs_ref)  # 当前 vs SFT
    reward_adjusted = rewards - beta * kl  # KL 惩罚

    # 3. 优势估计
    values = critic(prompts, responses)
    advantages = compute_gae(reward_adjusted, values)

    # 4. 多次 PPO 更新(importance sampling + clip)
    for _ in range(ppo_epochs):
        log_probs_new = actor.log_probs(prompts, responses)
        ratio = (log_probs_new - log_probs_old).exp()
        # clip 防止 policy 跳变太大
        loss_actor = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 1 - eps, 1 + eps) * advantages,
        ).mean()
        loss_critic = F.mse_loss(values, reward_adjusted)
        (loss_actor + 0.5 * loss_critic).backward()
        actor.optimizer.step()
        critic.optimizer.step()
```

注意几个工程要点:

- **4 个模型同时常驻显存**:actor(被优化)、critic(估计 value)、reward_model、reference SFT model(算 KL)
- **`log_probs_old.detach()`**:用旧 policy 的概率作为 importance sampling 的基准,新 policy 的更新不影响这个 baseline
- **KL 惩罚是双重的**——既减在 reward 里(避免 reward hacking),也可以加在 loss 上(防止跳得太远)

这套代码实际上几百行,加上分布式训练、checkpoint、生成 batching,工程复杂度极高。这就是 OpenAI 的 RLHF 团队需要几十个工程师的原因——也是 DPO 在 2023 年彻底简化它后被开源社区一致接受的原因。

## 影响 / 后续

Stiennon 2020 在 RLHF 历史上的位置:**第一个完整 + 标准化的方案**。它的具体影响:

**1. 三阶段 SFT → RM → PPO 流程被 InstructGPT 直接继承**——[InstructGPT](02-instructgpt.md) 把这套方案从摘要单一任务推广到所有任务(QA、写作、代码、推理),并加大规模(175B + 33K 偏好),最终产物就是 ChatGPT。可以说 ChatGPT 的方法论核心是 2020 年这篇论文给出的

**2. "RM + KL 惩罚 + PPO" 的组合定型**——这套配方在 2020–2023 期间几乎所有 RLHF 工作都沿用。Anthropic 的 Claude、Google 的 Bard、Meta 的 LLaMA-Chat 都是同款流程,只是数据规模和具体 hyperparameter 略有差异

**3. 揭示了"对齐胜过规模"的规律**——1.3B RLHF 超过 6.7B SFT,这一发现直接催生了 2022 InstructGPT 用 1.3B 击败 175B 的标志性结果。对工业意义巨大——**部署成本低 100× 的对齐小模型可以替代未对齐大模型**

**4. 暴露 RLHF 的工程痛点**——4 个模型同时在显存、PPO 训练慢且不稳、reward hacking 频繁出现。这些痛点推动了后续两条简化路线:

- **算力简化**:[Constitutional AI](03-constitutional-ai.md) 用 AI 自评省掉人工标注
- **算法简化**:[DPO](04-dpo.md) 推导出监督学习等价形式,完全去掉 PPO

→ [02-instructgpt.md](02-instructgpt.md) · 把三阶段流程从摘要推广到通用任务,ChatGPT 直接前身
→ [03-constitutional-ai.md](03-constitutional-ai.md) · 用 AI feedback 替代人类反馈,Anthropic Claude 的核心方法
→ [04-dpo.md](04-dpo.md) · 跳过 RM 和 RL,工程上和 SFT 一样简单
→ [../07-gpt-scaling/03-gpt3.md](../07-gpt-scaling/03-gpt3.md) · 父结构,RLHF 的对齐对象
