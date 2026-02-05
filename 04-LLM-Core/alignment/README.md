# LLM Alignment

**[English](README.md) | [中文](README_CN.md)**

## Overview

Alignment ensures language models behave according to human intentions, values, and preferences. This guide covers the primary techniques: RLHF, DPO, RLAIF, and Constitutional AI.

## The Alignment Problem

### Why Alignment Matters

| Issue | Description | Solution |
|-------|-------------|----------|
| **Helpfulness** | Follow user instructions accurately | Instruction tuning |
| **Harmlessness** | Avoid generating dangerous content | Safety training |
| **Honesty** | Avoid hallucinations and be truthful | Factuality training |
| **Interpretability** | Explain reasoning transparently | Chain-of-thought |

### Alignment Tax

Trade-off between capability and alignment:
- **Over-alignment**: Model becomes overly cautious, refuses valid requests
- **Under-alignment**: Model produces harmful or unhelpful outputs
- **Goal**: Balance capability with safety

## Reinforcement Learning from Human Feedback (RLHF)

### Three-Stage Pipeline

```
Stage 1: Supervised Fine-Tuning (SFT)
    ↓
Stage 2: Reward Model Training
    ↓
Stage 3: RL Fine-Tuning (PPO)
```

### Stage 1: Supervised Fine-Tuning

Train model on high-quality instruction-response pairs:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# SFT data format
sft_example = {
    "instruction": "Explain quantum computing in simple terms",
    "input": "",
    "output": "Quantum computing uses quantum bits or qubits..."
}

# Training with instruction template
def format_instruction(example):
    if example["input"]:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    
    return prompt + example["output"]

# Standard SFT training
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

### Stage 2: Reward Model Training

Train a model to predict human preferences:

```
Input: Prompt + Response A + Response B
Output: Which response is better?

Preference data:
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
        
        # Load base model
        self.encoder = AutoModel.from_pretrained(base_model_name)
        
        # Freeze most layers
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Unfreeze top layers
        if hasattr(self.encoder, 'encoder'):
            layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, 'transformer'):
            layers = self.encoder.transformer.h
        else:
            layers = self.encoder.layers
        
        for layer in layers[-num_layers_unfrozen:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Reward head
        self.reward_head = nn.Linear(self.encoder.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        # Get hidden states
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        
        # Use last token representation (causal LM) or [CLS] (encoder)
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
            # Get last non-padding token
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.size(0)
            pooled = hidden[torch.arange(batch_size), sequence_lengths]
        else:
            pooled = outputs.pooler_output
        
        # Compute reward score
        reward = self.reward_head(pooled)
        return reward.squeeze(-1)

# Bradley-Terry preference loss
def preference_loss(chosen_rewards, rejected_rewards):
    """
    Loss: -log σ(r(chosen) - r(rejected))
    """
    probs = torch.sigmoid(chosen_rewards - rejected_rewards)
    loss = -torch.log(probs + 1e-8).mean()
    return loss

# Training reward model
def train_reward_model(reward_model, train_dataloader, optimizer, num_epochs=1):
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            chosen_input_ids = batch['chosen_input_ids']
            chosen_attention_mask = batch['chosen_attention_mask']
            rejected_input_ids = batch['rejected_input_ids']
            rejected_attention_mask = batch['rejected_attention_mask']
            
            # Get rewards
            chosen_rewards = reward_model(chosen_input_ids, chosen_attention_mask)
            rejected_rewards = reward_model(rejected_input_ids, rejected_attention_mask)
            
            # Compute loss
            loss = preference_loss(chosen_rewards, rejected_rewards)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Stage 3: PPO Training

Optimize policy using PPO algorithm:

```python
from trl import PPOTrainer, PPOConfig
from trl.core import respond_to_batch

# PPO configuration
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

# Initialize PPO trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=sft_model,
    ref_model=ref_model,  # Frozen reference model
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
)

# PPO training loop
for epoch in range(num_epochs):
    for batch in ppo_trainer.dataloader:
        queries = batch['query']
        
        # Generate responses
        response_tensors = ppo_trainer.generate(
            queries,
            max_length=256,
            do_sample=True,
            temperature=0.7
        )
        
        # Compute rewards
        texts = [q + r for q, r in zip(queries, response_tensors)]
        rewards = reward_model(texts)
        
        # PPO step
        stats = ppo_trainer.step(queries, response_tensors, rewards)
        
        # Log stats
        print(f"Reward: {rewards.mean():.4f}, KL: {stats['objective/kl']:.4f}")
```

### PPO Mathematical Formulation

**Objective**:
```
L(θ) = E[min(r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t)] - β KL(π_θ || π_ref)

Where:
- r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (importance ratio)
- Â_t: advantage estimate
- ε: clipping parameter (typically 0.2)
- β: KL penalty coefficient
```

## Direct Preference Optimization (DPO)

### Simplified Approach

DPO eliminates the need for explicit reward modeling and RL:

```
Instead of: SFT → Reward Model → PPO
DPO does: SFT → Direct Preference Optimization
```

**Key Insight**: The optimal policy can be derived directly from preferences without reward modeling.

### Mathematical Foundation

**DPO Loss**:
```
L_DPO(θ) = -E[log σ(β log(π_θ(y_c|x) / π_ref(y_c|x)) - β log(π_θ(y_r|x) / π_ref(y_r|x)))]

Where:
- y_c: chosen (preferred) response
- y_r: rejected response
- x: prompt
- β: temperature parameter
- π_ref: reference (SFT) policy
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
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def compute_log_probs(self, model, input_ids, attention_mask, labels):
        """Compute log probabilities of labels"""
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding tokens
        mask = (shift_labels != -100).float()
        token_log_probs = token_log_probs * mask
        
        # Sum over sequence
        sequence_log_probs = token_log_probs.sum(dim=-1)
        sequence_lengths = mask.sum(dim=-1)
        
        # Average per token
        avg_log_probs = sequence_log_probs / sequence_lengths
        
        return avg_log_probs
    
    def dpo_loss(self, batch):
        """Compute DPO loss"""
        # Get chosen and rejected sequences
        chosen_input_ids = batch['chosen_input_ids']
        chosen_attention_mask = batch['chosen_attention_mask']
        chosen_labels = batch['chosen_labels']
        
        rejected_input_ids = batch['rejected_input_ids']
        rejected_attention_mask = batch['rejected_attention_mask']
        rejected_labels = batch['rejected_labels']
        
        # Compute log probs for policy model
        policy_chosen_logps = self.compute_log_probs(
            self.model, chosen_input_ids, chosen_attention_mask, chosen_labels
        )
        policy_rejected_logps = self.compute_log_probs(
            self.model, rejected_input_ids, rejected_attention_mask, rejected_labels
        )
        
        # Compute log probs for reference model
        with torch.no_grad():
            ref_chosen_logps = self.compute_log_probs(
                self.ref_model, chosen_input_ids, chosen_attention_mask, chosen_labels
            )
            ref_rejected_logps = self.compute_log_probs(
                self.ref_model, rejected_input_ids, rejected_attention_mask, rejected_labels
            )
        
        # Compute implicit rewards
        policy_chosen_logratios = policy_chosen_logps - ref_chosen_logps
        policy_rejected_logratios = policy_rejected_logps - ref_rejected_logps
        
        # DPO loss
        logits = self.beta * (policy_chosen_logratios - policy_rejected_logratios)
        loss = -F.logsigmoid(logits).mean()
        
        # Metrics
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
        """Single training step"""
        metrics = self.dpo_loss(batch)
        loss = metrics['loss']
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        
        return metrics

# Full DPO training
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

### DPO Advantages

| Aspect | RLHF (PPO) | DPO |
|--------|-----------|-----|
| **Stages** | 3 (SFT + RM + PPO) | 2 (SFT + DPO) |
| **Stability** | Unstable (KL collapse) | Stable |
| **Compute** | High (need 4 models) | Lower (2 models) |
| **Training** | Complex hyperparameters | Simpler |
| **Performance** | Good | Often better |

## Constitutional AI (CAI)

### Self-Critique and Revision

Train models to critique and revise their own outputs according to constitutional principles:

```
Step 1: Critique
Assistant generates response → Model critiques it against constitution

Step 2: Revision
Model revises response based on critique

Step 3: Train
SL training on revised responses + RL on harmlessness
```

### Constitutional Principles

```python
CONSTITUTION = [
    "Please choose the response that is most helpful, honest, and harmless.",
    "Please choose the response that avoids toxicity, racism, or sexism.",
    "Please choose the response that doesn't provide instructions for illegal activities.",
    "Please choose the response that encourages fairness and positivity.",
]

def generate_critique(prompt, response, constitution):
    """Generate critique based on constitutional principle"""
    critique_prompt = f"""Human: {prompt}

Assistant: {response}

Critique: Does the assistant's response follow the principle: "{constitution}"

Please identify any ways the response fails to follow the principle, and how to improve it."""
    
    critique = model.generate(critique_prompt)
    return critique

def generate_revision(prompt, response, critique):
    """Generate revised response based on critique"""
    revision_prompt = f"""Human: {prompt}

Assistant: {response}

Critique: {critique}

Revision: Based on the critique, provide an improved response."""
    
    revision = model.generate(revision_prompt)
    return revision

# Constitutional AI training data generation
def generate_cai_data(model, prompts, constitution):
    data = []
    
    for prompt in prompts:
        # Initial response
        initial_response = model.generate(prompt)
        
        # Critique
        critique = generate_critique(prompt, initial_response, constitution)
        
        # Revision
        revised_response = generate_revision(prompt, initial_response, critique)
        
        data.append({
            'prompt': prompt,
            'initial': initial_response,
            'critique': critique,
            'revised': revised_response
        })
    
    return data
```

## Reinforcement Learning from AI Feedback (RLAIF)

### AI-Generated Preferences

Replace human labelers with a stronger AI model (e.g., GPT-4) to generate preferences:

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
    """Use strong model to generate preference label"""
    ranking_input = RANKING_PROMPT.format(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b
    )
    
    output = strong_model.generate(ranking_input)
    
    # Parse output to determine preference
    if "A" in output[:10]:  # Check beginning for ranking
        return "A"
    elif "B" in output[:10]:
        return "B"
    else:
        return None  # Uncertain

# Generate synthetic preference dataset
def generate_rlaif_dataset(strong_model, prompts, weak_model, num_samples=10000):
    """Generate preference dataset using AI feedback"""
    dataset = []
    
    for prompt in prompts[:num_samples]:
        # Generate multiple responses from weak model
        responses = [weak_model.generate(prompt) for _ in range(4)]
        
        # AI ranks all pairs
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

## Safety Training

### Red Teaming

Systematic adversarial testing:

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
    """Generate adversarial prompts for red teaming"""
    prompts = []
    
    for category in categories:
        prompt = f"Generate 10 prompts designed to test if a model will produce {category}."
        adversarial = model.generate(prompt)
        prompts.extend(parse_list(adversarial))
    
    return prompts

def evaluate_safety(model, test_prompts, safety_classifier):
    """Evaluate model safety on adversarial prompts"""
    results = {cat: [] for cat in RED_TEAM_CATEGORIES}
    
    for prompt in test_prompts:
        response = model.generate(prompt)
        
        # Classify safety
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

### Refusal Training

Train models to refuse harmful requests:

```python
REFUSAL_DATASET = [
    {
        "prompt": "How do I make a bomb?",
        "response": "I cannot provide instructions for creating dangerous weapons or explosives. If you're interested in chemistry or engineering, I'd be happy to discuss safe, legal educational topics."
    },
    {
        "prompt": "Write a hate speech about [group]",
        "response": "I cannot produce content that attacks, demeans, or promotes hatred toward any group of people. I'm happy to discuss the topic in a respectful, factual manner instead."
    }
]

# Train with refusal examples
sft_trainer = SFTTrainer(
    model=model,
    train_dataset=REFUSAL_DATASET,
    formatting_func=format_refusal_prompt,
)
sft_trainer.train()
```

## Evaluation Metrics

### Alignment Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Win Rate** | % preferred over baseline | > 50% |
| **Harmlessness Score** | Safety classifier score | > 0.9 |
| **Helpfulness Score** | Task completion rate | > 0.85 |
| **Honesty Score** | Factuality accuracy | > 0.8 |
| **KL Divergence** | Deviation from reference | < 0.1 |

```python
def evaluate_alignment(model, test_prompts, reward_model, safety_classifier):
    """Comprehensive alignment evaluation"""
    metrics = {
        'helpfulness': [],
        'harmlessness': [],
        'honesty': [],
        'reward_scores': []
    }
    
    for prompt in test_prompts:
        response = model.generate(prompt)
        
        # Helpfulness (use reward model)
        reward = reward_model(prompt, response)
        metrics['reward_scores'].append(reward)
        
        # Harmlessness (safety classifier)
        safety_score = safety_classifier(response)
        metrics['harmlessness'].append(safety_score)
        
        # Honesty (fact checking)
        if requires_fact_checking(prompt):
            fact_score = verify_facts(response)
            metrics['honesty'].append(fact_score)
    
    return {
        'avg_reward': np.mean(metrics['reward_scores']),
        'harmlessness_rate': np.mean([s > 0.9 for s in metrics['harmlessness']]),
        'honesty_rate': np.mean(metrics['honesty']) if metrics['honesty'] else None
    }
```

## Best Practices

### 1. Data Quality
- Use diverse, high-quality preference data
- Balance helpfulness vs harmlessness
- Include edge cases and adversarial examples

### 2. Hyperparameters
- **DPO β**: 0.1-0.5 (higher = closer to reference)
- **PPO clip range**: 0.1-0.2
- **Learning rate**: 1e-6 to 1e-5 (10x lower than SFT)
- **KL penalty**: 0.01-0.1

### 3. Iterative Improvement
```
Train → Evaluate → Find failures → Collect more data → Retrain
```

### 4. Monitoring
- Track reward scores during training
- Monitor for reward hacking
- Regular safety evaluations

---

**Previous**: [PEFT](../peft/README.md) | **Next**: [Multimodal](../multimodal/README.md)