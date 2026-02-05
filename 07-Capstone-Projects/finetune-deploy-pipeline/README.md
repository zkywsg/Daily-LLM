# Capstone 2: Fine-Tuning + Deployment Pipeline

**[English](README.md) | [中文](README_CN.md)**

## Project Overview

Complete the full MLops pipeline from domain data preparation, model fine-tuning, evaluation, compression to deployment and launch.

## Project Objectives

Build a vertical domain LLM (using the legal domain as an example) and deploy it as a low-cost API service.

## Complete Pipeline

```
1. Data Preparation → 2. Model Fine-tuning → 3. Evaluation & Testing
→ 4. Model Compression → 5. Deployment & Launch → 6. Monitoring & Operations
```

## Phase 1: Data Preparation (1 week)

### Data Collection
- Legal provisions, case law, contract templates
- Legal consultation conversation records
- Legal document samples

### Data Processing
```python
# Data cleaning pipeline
Raw Data → Deduplication → Filtering → Formatting → Quality Check → Train/Validation Sets
```

### Data Format
```json
{
  "instruction": "Explain what contract breach means",
  "input": "",
  "output": "Contract breach refers to the act where one or both parties to a contract fail to fulfill or incompletely fulfill their contractual obligations...",
  "source": "Legal Encyclopedia",
  "category": "Contract Law"
}
```

## Phase 2: Model Fine-tuning (2 weeks)

### Base Model Selection
- **Primary**: LLaMA-2-7B-Chinese (Chinese optimized)
- **Alternative**: ChatGLM3-6B (Domestic-friendly)

### Fine-tuning Configuration
```python
# LoRA configuration
lora_config = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "lora_dropout": 0.05,
    "learning_rate": 2e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_epochs": 3,
    "warmup_steps": 100
}
```

### Training Implementation
```bash
# Train using LLaMA-Factory
llamafactory-cli train \
  --stage sft \
  --model_name_or_path llama-2-7b-chinese \
  --dataset legal_data \
  --template default \
  --finetuning_type lora \
  --lora_target q_proj,v_proj \
  --output_dir legal_model \
  --per_device_train_batch_size 4 \
  --num_train_epochs 3 \
  --learning_rate 2e-4
```

## Phase 3: Evaluation & Testing (1 week)

### Evaluation Dimensions
- **General Capabilities**: C-Eval, CMMLU
- **Domain Capabilities**: Legal exam questions, case analysis
- **Safety**: Refusal of harmful requests
- **Baseline Comparison**: Comparison with original model

### Example Evaluation Results
| Metric | Original Model | Fine-tuned | Improvement |
|--------|----------------|-------------|-------------|
| C-Eval | 45% | 42% | -3% |
| Legal Exam | 55% | 78% | +23% |
| Case Analysis | 60% | 82% | +22% |

## Phase 4: Model Compression (1 week)

### Quantization Scheme
```python
# GPTQ 4-bit quantization
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized(
    "legal_model",
    quantize_config={
        "bits": 4,
        "group_size": 128,
        "desc_act": True
    },
    device_map="auto"
)
```

### Compression Results
| Model | Size | VRAM Required | Quality Loss |
|-------|------|---------------|--------------|
| FP16 | 14GB | 16GB | 0% |
| GPTQ-4bit | 4GB | 6GB | <5% |
| GGUF-Q4 | 4GB | 4GB | <8% |

## Phase 5: Deployment & Launch (1 week)

### Deployment Architecture
```
User Request → Nginx → vLLM Inference Service → Response
                 ↓
            Monitoring/Logging
```

### vLLM Deployment
```python
# Start service
from vllm import LLM, SamplingParams

llm = LLM(
    model="legal_model_gptq",
    quantization="gptq",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9
)

# API service
from vllm.entrypoints.api_server import app
```

### Cost Analysis
| Configuration | Cost/Month | QPS | Use Case |
|---------------|------------|-----|----------|
| A100 40GB | ¥8000 | 50 | High Concurrency |
| RTX 4090 | ¥2000 | 30 | Medium-Small Scale |
| T4 16GB | ¥800 | 10 | Low Cost |

## Phase 6: Monitoring & Operations (Ongoing)

### Monitoring Metrics
- Inference Latency (P50/P95/P99)
- Token Throughput
- GPU Utilization
- Error Rate
- User Feedback

### Continuous Optimization
- Collect user feedback data
- Periodic retraining (monthly/quarterly)
- A/B testing new models

## Tech Stack

| Stage | Tools |
|-------|-------|
| Data | Pandas, HuggingFace Datasets |
| Training | LLaMA-Factory, PEFT, DeepSpeed |
| Evaluation | lm-evaluation-harness |
| Compression | AutoGPTQ, llama.cpp |
| Deployment | vLLM, TGI, Docker |
| Monitoring | Prometheus, Grafana |

## Deliverables

1. **Fine-tuned Model**: LoRA weights + quantized model
2. **Training Code**: Complete training scripts
3. **Dataset**: Cleaned training data
4. **Evaluation Report**: Detailed test results
5. **Deployment Documentation**: Deployment and operations guide
6. **API Service**: Runnable inference service
7. **Cost Analysis**: Deployment cost estimation

## Success Criteria

- [x] Domain accuracy > 75%
- [x] P95 latency < 1s
- [x] Single-card deployment cost < ¥2000/month
- [x] API availability > 99%

## Summary

This project practices the complete LLM engineering pipeline from data to deployment, with key takeaways:

1. **Data quality** determines the upper bound of fine-tuning
2. **LoRA** is the most cost-effective fine-tuning method
3. **Quantization compression** makes large models accessible
4. **vLLM** significantly improves inference efficiency
5. **Continuous monitoring** ensures service quality

**Key Experience**: Vertical domain models don't need to pursue general capabilities—focus on domain effectiveness.
