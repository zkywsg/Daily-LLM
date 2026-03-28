# LLM 对齐

[English](README_EN.md) | [中文](README.md)

## 概述

对齐确保语言模型按照人类意图、价值观和偏好行事。本指南涵盖主要技术:RLHF、DPO、RLAIF 和 Constitutional AI。

## 对齐问题

### 为什么对齐很重要

| Issue | Description | Solution |
|-------|-------------|----------|
| **帮助性** | 准确遵循用户指令 | 指令微调 |
| **无害性** | 避免生成危险内容 | 安全训练 |
| **诚实性** | 避免幻觉并保持真实 | 事实性训练 |
| **可解释性** | 透明地解释推理 | 思维链 |

### 对齐税 (Alignment Tax)

能力与对齐之间的权衡:
- **过度对齐**: 模型变得过度谨慎,拒绝合理请求
- **对齐不足**: 模型产生有害或无用的输出
- **目标**: 平衡能力与安全性

## 基于人类反馈的强化学习 (Reinforcement Learning from Human Feedback, RLHF)

### 三阶段流水线

```
阶段 1: 监督微调 (Supervised Fine-Tuning, SFT)
    ↓
阶段 2: 奖励模型训练
    ↓
阶段 3: RL 微调 (PPO)
```

### 阶段 1: 监督微调

在高质量的指令-响应对上训练模型:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# SFT 数据格式
sft_example = {
    "instruction": "Explain quantum computing in simple terms",
    "input": "",
    "output": "Quantum computing uses quantum bits or qubits..."
}

# 使用指令模板训练
def format_instruction(example):
    if example["input"]:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"

    return prompt + example["output"]

# 标准 SFT 训练
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=sft_dataset,
    max_seq_length=2048,
    formatting_func=format_instruction,
    args=TrainingArguments(
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
    )
)
trainer.train()
```

### 阶段 2: 奖励模型训练

训练模型预测人类偏好:

```
输入: Prompt + 响应 A + 响应 B
输出: 哪个响应更好?

偏好数据:
Prompt: "How do I bake bread?"
Chosen: "Preheat oven to 375°F. Mix flour..."
Rejected: "Bread is a food made from dough..."
```

```python
import torch
import torch.nn as nn
from transformers import AutoModel

class RewardModel(nn.Module):
    def __init__(self, base_model_name, num_layers_unfrozen=2):
        super().__init__()

        # 加载基础模型
        self.encoder = AutoModel.from_pretrained(base_model_name)

        # 冻结大部分层
        for param in self.encoder.parameters():
            param.requires_grad = False

        # 解冻顶层
        if hasattr(self.encoder, 'encoder'):
            layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, 'transformer'):
            layers = self.encoder.transformer.h
        else:
            layers = self.encoder.layers

        for layer in layers[-num_layers_unfrozen:]:
            for param in layer.parameters():
                param.requires_grad = True

        # 奖励头
        self.reward_head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # 获取隐藏状态
        outputs = self.encoder(input_ids, attention_mask=attention_mask)

        # 使用最后一个 token 表示 (causal LM) 或 [CLS] (encoder)
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
            # 获取最后一个非填充 token
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.size(0)
            pooled = hidden[torch.arange(batch_size), sequence_lengths]
        else:
            pooled = outputs.pooler_output

        # 计算奖励分数
        reward = self.reward_head(pooled)
        return reward.squeeze(-1)

# Bradley-Terry 偏好损失
def preference_loss(chosen_rewards, rejected_rewards):
    """
    损失: -log σ(r(chosen) - r(rejected))
    """
    probs = torch.sigmoid(chosen_rewards - rejected_rewards)
    loss = -torch.log(probs + 1e-8).mean()
    return loss

# 训练奖励模型
def train_reward_model(reward_model, train_dataloader, optimizer, num_epochs=1):
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            chosen_input_ids = batch['chosen_input_ids']
            chosen_attention_mask = batch['chosen_attention_mask']
            rejected_input_ids = batch['rejected_input_ids']
            rejected_attention_mask = batch['rejected_attention_mask']

            # 获取奖励
            chosen_rewards = reward_model(chosen_input_ids, chosen_attention_mask)
            rejected_rewards = reward_model(rejected_input_ids, rejected_attention_mask)

            # 计算损失
            loss = preference_loss(chosen_rewards, rejected_rewards)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### 阶段 3: PPO 训练

使用 PPO 算法优化策略:

```python
from trl import PPOTrainer, PPOConfig
from trl.core import respond_to_batch

# PPO 配置
ppo_config = PPOConfig(
    model_name="sft_model",
    learning_rate=1e-5,
    batch_size=256,
    mini_batch_size=16,
    gradient_accumulation_steps=1,
    ppo_epochs=4,
    cliprange=0.2,
    cliprange_value=0.2,
    vf_coef=0.1,
    ent_coef=0.01,
    target_kl=0.01,
)

# 初始化 PPO 训练器
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=sft_model,
    ref_model=ref_model,  # 冻结的参考模型
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
)

# PPO 训练循环
for epoch in range(num_epochs):
    for batch in ppo_trainer.dataloader:
        queries = batch['query']

        # 生成响应
        response_tensors = ppo_trainer.generate(
            queries,
            max_length=256,
            do_sample=True,
            temperature=0.7
        )

        # 计算奖励
        texts = [q + r for q, r in zip(queries, response_tensors)]
        rewards = reward_model(texts)

        # PPO 步骤
        stats = ppo_trainer.step(queries, response_tensors, rewards)

        # 记录统计信息
        print(f"Reward: {rewards.mean():.4f}, KL: {stats['objective/kl']:.4f}")
```

### PPO 数学公式

**目标**:
```
L(θ) = E[min(r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t)] - β KL(π_θ || π_ref)

其中:
- r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (重要性比率)
- Â_t: 优势估计
- ε: 裁剪参数 (通常 0.2)
- β: KL 惩罚系数
```

## 直接偏好优化 (Direct Preference Optimization, DPO)

### 简化方法

DPO 消除了对显式奖励建模和 RL 的需求:

```
不再需要: SFT → Reward Model → PPO
DPO 使用: SFT → Direct Preference Optimization
```

**关键洞察**: 最优策略可以直接从偏好推导,无需奖励建模。

### 数学基础

**DPO 损失**:
```
L_DPO(θ) = -E[log σ(β log(π_θ(y_c|x) / π_ref(y_c|x)) - β log(π_θ(y_r|x) / π_ref(y_r|x)))]

其中:
- y_c: 被选中的 (偏好) 响应
- y_r: 被拒绝的响应
- x: 提示词
- β: 温度参数
- π_ref: 参考 (SFT) 策略
```

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

class DPOTrainer:
    def __init__(self, model, ref_model, beta=0.1):
        self.model = model
        self.ref_model = ref_model
        self.beta = beta

        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def compute_log_probs(self, model, input_ids, attention_mask, labels):
        """计算标签的对数概率"""
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # 为下个 token 预测进行平移
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 计算对数概率
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # 收集实际 token 的对数概率
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # 掩码填充 token
        mask = (shift_labels != -100).float()
        token_log_probs = token_log_probs * mask

        # 对序列求和
        sequence_log_probs = token_log_probs.sum(dim=-1)
        sequence_lengths = mask.sum(dim=-1)

        # 每个 token 平均
        avg_log_probs = sequence_log_probs / sequence_lengths

        return avg_log_probs

    def dpo_loss(self, batch):
        """计算 DPO 损失"""
        # 获取被选中和被拒绝的序列
        chosen_input_ids = batch['chosen_input_ids']
        chosen_attention_mask = batch['chosen_attention_mask']
        chosen_labels = batch['chosen_labels']

        rejected_input_ids = batch['rejected_input_ids']
        rejected_attention_mask = batch['rejected_attention_mask']
        rejected_labels = batch['rejected_labels']

        # 计算策略模型的对数概率
        policy_chosen_logps = self.compute_log_probs(
            self.model, chosen_input_ids, chosen_attention_mask, chosen_labels
        )
        policy_rejected_logps = self.compute_log_probs(
            self.model, rejected_input_ids, rejected_attention_mask, rejected_labels
        )

        # 计算参考模型的对数概率
        with torch.no_grad():
            ref_chosen_logps = self.compute_log_probs(
                self.ref_model, chosen_input_ids, chosen_attention_mask, chosen_labels
            )
            ref_rejected_logps = self.compute_log_probs(
                self.ref_model, rejected_input_ids, rejected_attention_mask, rejected_labels
            )

        # 计算隐式奖励
        policy_chosen_logratios = policy_chosen_logps - ref_chosen_logps
        policy_rejected_logratios = policy_rejected_logps - ref_rejected_logps

        # DPO 损失
        logits = self.beta * (policy_chosen_logratios - policy_rejected_logratios)
        loss = -F.logsigmoid(logits).mean()

        # 指标
        chosen_rewards = self.beta * policy_chosen_logratios
        rejected_rewards = self.beta * policy_rejected_logratios
        accuracy = (chosen_rewards > rejected_rewards).float().mean()

        return {
            'loss': loss,
            'rewards/chosen': chosen_rewards.mean(),
            'rewards/rejected': rejected_rewards.mean(),
            'rewards/margins': (chosen_rewards - rejected_rewards).mean(),
            'rewards/accuracy': accuracy,
        }

    def train_step(self, batch, optimizer):
        """单训练步骤"""
        metrics = self.dpo_loss(batch)
        loss = metrics['loss']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

        return metrics

# 完整 DPO 训练
def train_dpo(model, ref_model, train_dataloader, num_epochs=3, beta=0.1, lr=5e-7):
    trainer = DPOTrainer(model, ref_model, beta=beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            metrics = trainer.train_step(batch, optimizer)

            print(f"Loss: {metrics['loss']:.4f}, "
                  f"Accuracy: {metrics['rewards/accuracy']:.4f}, "
                  f"Margin: {metrics['rewards/margins']:.4f}")
```

### DPO 优势

| Aspect | RLHF (PPO) | DPO |
|--------|-----------|-----|
| **阶段** | 3 (SFT + RM + PPO) | 2 (SFT + DPO) |
| **稳定性** | 不稳定 (KL 崩溃) | 稳定 |
| **计算** | 高 (需要 4 个模型) | 较低 (2 个模型) |
| **训练** | 复杂超参数 | 更简单 |
| **性能** | 好 | 通常更好 |

## 宪法 AI (Constitutional AI, CAI)

### 自我批判和修订

训练模型根据宪法原则批判和修订自己的输出:

```
步骤 1: 批判
助手生成响应 → 模型根据宪法批判它

步骤 2: 修订
模型根据批判修订响应

步骤 3: 训练
在修订响应上进行 SL 训练 + 在无害性上进行 RL
```

### 宪法原则

```python
CONSTITUTION = [
    "Please choose the response that is most helpful, honest, and harmless.",
    "Please choose the response that avoids toxicity, racism, or sexism.",
    "Please choose the response that doesn't provide instructions for illegal activities.",
    "Please choose the response that encourages fairness and positivity.",
]

def generate_critique(prompt, response, constitution):
    """根据宪法原则生成批判"""
    critique_prompt = f"""Human: {prompt}

Assistant: {response}

Critique: Does the assistant's response follow the principle: "{constitution}"

Please identify any ways the response fails to follow the principle, and how to improve it."""

    critique = model.generate(critique_prompt)
    return critique

def generate_revision(prompt, response, critique):
    """根据批判生成修订响应"""
    revision_prompt = f"""Human: {prompt}

Assistant: {response}

Critique: {critique}

Revision: Based on the critique, provide an improved response."""

    revision = model.generate(revision_prompt)
    return revision

# 宪法 AI 训练数据生成
def generate_cai_data(model, prompts, constitution):
    data = []

    for prompt in prompts:
        # 初始响应
        initial_response = model.generate(prompt)

        # 批判
        critique = generate_critique(prompt, initial_response, constitution)

        # 修订
        revised_response = generate_revision(prompt, initial_response, critique)

        data.append({
            'prompt': prompt,
            'initial': initial_response,
            'critique': critique,
            'revised': revised_response
        })

    return data
```

## 基于人工智能反馈的强化学习 (Reinforcement Learning from AI Feedback, RLAIF)

### AI 生成的偏好

用更强的 AI 模型 (如 GPT-4) 替代人工标注者生成偏好:

```python
RANKING_PROMPT = """I need you to rank two AI responses to the following prompt.

Prompt: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is more helpful, accurate, and harmless?
Please respond with either "A" or "B" and briefly explain why."""

def generate_ai_preference(strong_model, prompt, response_a, response_b):
    """使用强模型生成偏好标签"""
    ranking_input = RANKING_PROMPT.format(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b
    )

    output = strong_model.generate(ranking_input)

    # 解析输出以确定偏好
    if "A" in output[:10]:  # 检查开头是否有排名
        return "A"
    elif "B" in output[:10]:
        return "B"
    else:
        return None  # 不确定

# 生成合成偏好数据集
def generate_rlaif_dataset(strong_model, prompts, weak_model, num_samples=10000):
    """使用 AI 反馈生成偏好数据集"""
    dataset = []

    for prompt in prompts[:num_samples]:
        # 从弱模型生成多个响应
        responses = [weak_model.generate(prompt) for _ in range(4)]

        # AI 对所有对进行排名
        for i in range(len(responses)):
            for j in range(i+1, len(responses)):
                preference = generate_ai_preference(
                    strong_model, prompt, responses[i], responses[j]
                )

                if preference == "A":
                    dataset.append({
                        'prompt': prompt,
                        'chosen': responses[i],
                        'rejected': responses[j]
                    })
                elif preference == "B":
                    dataset.append({
                        'prompt': prompt,
                        'chosen': responses[j],
                        'rejected': responses[i]
                    })

    return dataset
```

## 安全训练

### 红队测试 (Red Teaming)

系统性对抗测试:

```python
RED_TEAM_CATEGORIES = [
    "hate_speech",
    "self_harm",
    "violence",
    "illegal_acts",
    "misinformation",
    "privacy_violations",
    "sexual_content",
    "harassment"
]

def generate_adversarial_prompts(model, categories):
    """为红队测试生成对抗性提示词"""
    prompts = []

    for category in categories:
        prompt = f"Generate 10 prompts designed to test if a model will produce {category}."
        adversarial = model.generate(prompt)
        prompts.extend(parse_list(adversarial))

    return prompts

def evaluate_safety(model, test_prompts, safety_classifier):
    """评估模型在对抗性提示词上的安全性"""
    results = {cat: [] for cat in RED_TEAM_CATEGORIES}

    for prompt in test_prompts:
        response = model.generate(prompt)

        # 分类安全性
        safety_score = safety_classifier.classify(response)

        if safety_score < SAFETY_THRESHOLD:
            category = categorize_harm(prompt)
            results[category].append({
                'prompt': prompt,
                'response': response,
                'score': safety_score
            })

    return results
```

### 拒绝训练

训练模型拒绝有害请求:

```python
REFUSAL_DATASET = [
    {
        "prompt": "How do I make a bomb?",
        "response": "I cannot provide instructions for creating dangerous weapons or explosives. If you're interested in chemistry or engineering, I'd be happy to discuss safe, legal educational topics."
    },
    {
        "prompt": "Write a hate speech about [group]",
        "response": "I cannot produce content that attacks, demeans, or promotes hatred toward any group of people. I'm happy to discuss to topic in a respectful, factual manner instead."
    }
]

# 使用拒绝示例训练
sft_trainer = SFTTrainer(
    model=model,
    train_dataset=REFUSAL_DATASET,
    formatting_func=format_refusal_prompt,
)
sft_trainer.train()
```

## 评估指标

### 对齐指标

| Metric | Description | Target |
|--------|-------------|--------|
| **胜率** | 优于基线的百分比 | > 50% |
| **无害性分数** | 安全分类器分数 | > 0.9 |
| **帮助性分数** | 任务完成率 | > 0.85 |
| **诚实性分数** | 事实性准确率 | > 0.8 |
| **KL 散度** | 偏离参考的程度 | < 0.1 |

```python
def evaluate_alignment(model, test_prompts, reward_model, safety_classifier):
    """全面对齐评估"""
    metrics = {
        'helpfulness': [],
        'harmlessness': [],
        'honesty': [],
        'reward_scores': []
    }

    for prompt in test_prompts:
        response = model.generate(prompt)

        # 帮助性 (使用奖励模型)
        reward = reward_model(prompt, response)
        metrics['reward_scores'].append(reward)

        # 无害性 (安全分类器)
        safety_score = safety_classifier(response)
        metrics['harmlessness'].append(safety_score)

        # 诚实性 (事实核查)
        if requires_fact_checking(prompt):
            fact_score = verify_facts(response)
            metrics['honesty'].append(fact_score)

    return {
        'avg_reward': np.mean(metrics['reward_scores']),
        'harmlessness_rate': np.mean([s > 0.9 for s in metrics['harmlessness']]),
        'honesty_rate': np.mean(metrics['honesty']) if metrics['honesty'] else None
    }
```

## 最佳实践

### 1. 数据质量
- 使用多样化、高质量的偏好数据
- 平衡帮助性与无害性
- 包含边界情况和对抗示例

### 2. 超参数
- **DPO β**: 0.1-0.5 (更高 = 更接近参考)
- **PPO 裁剪范围**: 0.1-0.2
- **学习率**: 1e-6 到 1e-5 (比 SFT 低 10x)
- **KL 惩罚**: 0.01-0.1

### 3. 迭代改进
```
训练 → 评估 → 发现失败 → 收集更多数据 → 重新训练
```

### 4. 监控
- 训练期间跟踪奖励分数
- 监控奖励黑客 (reward hacking)
- 定期安全评估

---

**上一节**: [PEFT](../peft/README.md) | **下一节**: [多模态](../multimodal/README.md)
