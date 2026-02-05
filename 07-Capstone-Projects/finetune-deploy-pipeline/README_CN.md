# Capstone 2: 微调+部署一体化项目

**[English](README.md) | [中文](README_CN.md)**

## 项目概述

完成从领域数据准备、模型微调、评估、压缩到部署上线的完整MLops流程。

## 项目目标

构建一个垂直领域LLM (以法律领域为例)，并部署为低成本的API服务。

## 完整流程

```
1. 数据准备 → 2. 模型微调 → 3. 评估测试
→ 4. 模型压缩 → 5. 部署上线 → 6. 监控运营
```

## Phase 1: 数据准备 (1周)

### 数据收集
- 法律条文、案例、合同模板
- 法律咨询对话记录
- 法律文书样本

### 数据处理
```python
# 数据清洗流程
原始数据 → 去重 → 过滤 → 格式化 → 质量检查 → 训练集/验证集
```

### 数据格式
```json
{
  "instruction": "解释什么是合同违约",
  "input": "",
  "output": "合同违约是指合同当事人一方或双方不履行或不完全履行合同义务的行为...",
  "source": "法律百科",
  "category": "合同法"
}
```

## Phase 2: 模型微调 (2周)

### 基础模型选择
- **主选**: LLaMA-2-7B-Chinese (中文优化)
- **备选**: ChatGLM3-6B (国内友好)

### 微调配置
```python
# LoRA配置
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

### 训练实施
```bash
# 使用LLaMA-Factory训练
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

## Phase 3: 评估测试 (1周)

### 评估维度
- **通用能力**: C-Eval、CMMLU
- **领域能力**: 法律考试题、案例分析
- **安全性**: 拒绝有害请求
- **对比基线**: 与原始模型对比

### 评估结果示例
| 指标 | 原始模型 | 微调后 | 提升 |
|------|---------|--------|------|
| C-Eval | 45% | 42% | -3% |
| 法律考试 | 55% | 78% | +23% |
| 案例分析 | 60% | 82% | +22% |

## Phase 4: 模型压缩 (1周)

### 量化方案
```python
# GPTQ 4-bit量化
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

### 压缩效果
| 模型 | 大小 | 显存需求 | 效果损失 |
|------|------|---------|---------|
| FP16 | 14GB | 16GB | 0% |
| GPTQ-4bit | 4GB | 6GB | <5% |
| GGUF-Q4 | 4GB | 4GB | <8% |

## Phase 5: 部署上线 (1周)

### 部署架构
```
用户请求 → Nginx → vLLM推理服务 → 响应
                ↓
           监控/日志
```

### vLLM部署
```python
# 启动服务
from vllm import LLM, SamplingParams

llm = LLM(
    model="legal_model_gptq",
    quantization="gptq",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9
)

# API服务
from vllm.entrypoints.api_server import app
```

### 成本分析
| 配置 | 成本/月 | QPS | 适用场景 |
|------|---------|-----|---------|
| A100 40GB | ¥8000 | 50 | 高并发 |
| RTX 4090 | ¥2000 | 30 | 中小规模 |
| T4 16GB | ¥800 | 10 | 低成本 |

## Phase 6: 监控运营 (持续)

### 监控指标
- 推理延迟 (P50/P95/P99)
- Token吞吐量
- GPU利用率
- 错误率
- 用户反馈

### 持续优化
- 收集用户反馈数据
- 定期重训练 (每月/每季度)
- A/B测试新模型

## 技术栈

| 环节 | 工具 |
|------|------|
| 数据 | Pandas, HuggingFace Datasets |
| 训练 | LLaMA-Factory, PEFT, DeepSpeed |
| 评估 | lm-evaluation-harness |
| 压缩 | AutoGPTQ, llama.cpp |
| 部署 | vLLM, TGI, Docker |
| 监控 | Prometheus, Grafana |

## 交付物

1. **微调模型**: LoRA权重 + 量化模型
2. **训练代码**: 完整训练脚本
3. **数据集**: 清洗后的训练数据
4. **评估报告**: 详细测试结果
5. **部署文档**: 部署和运维指南
6. **API服务**: 可运行的推理服务
7. **成本分析**: 部署成本估算

## 成功标准

- [x] 领域准确率 > 75%
- [x] P95延迟 < 1s
- [x] 单卡部署成本 < ¥2000/月
- [x] API可用性 > 99%

## 总结

本项目完整实践了从数据到部署的LLM工程化流程，核心收获：

1. **数据质量**决定微调上限
2. **LoRA**是性价比最高的微调方式
3. **量化压缩**让大模型平民化
4. **vLLM**大幅提升推理效率
5. **持续监控**保障服务质量

**关键经验**: 垂直领域模型不需要追求通用能力，专注领域效果即可。
