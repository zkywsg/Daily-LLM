[English](README.md) | [中文](README_CN.md)
# 模型压缩策略 (Model Compression Strategies)

## 目录

1. [背景 (Why Model Compression?)](#1-背景-why-model-compression)
2. [核心概念 (Quantization, Pruning, Distillation)](#2-核心概念-quantization-pruning-distillation)
3. [数学原理 (Compression Ratios, Accuracy Trade-off)](#3-数学原理-compression-ratios-accuracy-trade-off)
4. [代码实现 (Compression Techniques)](#4-代码实现-compression-techniques)
5. [实验对比 (Compression Methods)](#5-实验对比-compression-methods)
6. [最佳实践与常见陷阱](#6-最佳实践与常见陷阱)
7. [总结](#7-总结)

---

## 1. 背景 (Why Model Compression?)

### 1.1 大模型的挑战

- **部署成本**: 大模型需要昂贵GPU
- **推理延迟**: 影响用户体验
- **功耗**: 边缘设备无法承受
- **存储**: 模型文件过大

### 1.2 压缩的价值

- **降低成本**: 节省70%+计算资源
- **提升速度**: 2-4x推理加速
- **边缘部署**: 手机/IoT设备运行
- **环保**: 减少碳排放

---

## 2. 核心概念 (Quantization, Pruning, Distillation)

### 2.1 量化 (Quantization)

降低参数精度：
- FP32 → FP16: 2x压缩
- FP32 → INT8: 4x压缩
- FP32 → INT4: 8x压缩

### 2.2 剪枝 (Pruning)

移除不重要参数：
- **非结构化**: 单个权重置零
- **结构化**: 移除整个通道/层

### 2.3 蒸馏 (Distillation)

小模型学习大模型知识。

---

## 3. 数学原理 (Compression Ratios, Accuracy Trade-off)

### 3.1 压缩比

$$
\text{Compression Ratio} = \frac{\text{Original Size}}{\text{Compressed Size}}
$$

### 3.2 精度-效率权衡

$$
\text{Efficiency} = \frac{\text{Accuracy}}{\text{Model Size}} \times \text{Speed}
$$

---

## 4. 代码实现 (Compression Techniques)

### 4.1 INT8动态量化

```python
import torch

# 原始模型
model = MyModel().eval()

# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.LSTM},
    dtype=torch.qint8
)

# 保存
torch.save(quantized_model.state_dict(), "quantized.pth")
```

### 4.2 GPTQ 4-bit量化

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 4-bit配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=bnb_config,
    device_map="auto"
)

# 直接使用
output = model.generate(input_ids, max_new_tokens=100)
```

### 4.3 知识蒸馏

```python
class DistillationLoss(nn.Module):
    """蒸馏损失"""
    
    def __init__(self, temperature=2.0):
        super().__init__()
        self.T = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        # 软目标损失
        soft_loss = self.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(teacher_logits / self.T, dim=1)
        ) * (self.T * self.T)
        
        # 硬目标损失
        hard_loss = F.cross_entropy(student_logits, labels)
        
        return 0.7 * soft_loss + 0.3 * hard_loss

# 训练循环
for batch in dataloader:
    inputs, labels = batch
    
    # 教师模型 (冻结)
    with torch.no_grad():
        teacher_logits = teacher_model(inputs)
    
    # 学生模型
    student_logits = student_model(inputs)
    
    # 蒸馏损失
    loss = distillation_loss(student_logits, teacher_logits, labels)
    
    loss.backward()
    optimizer.step()
```

---

## 5. 实验对比 (Compression Methods)

### 5.1 方法对比

| 方法 | 压缩比 | 精度损失 | 速度提升 | 适用 |
|------|--------|---------|---------|------|
| **FP16** | 2x | <1% | 2x | 通用 |
| **INT8** | 4x | 2-3% | 3x | 通用 |
| **INT4** | 8x | 4-6% | 4x | 边缘 |
| **剪枝** | 2-10x | 3-5% | 2x | 专用 |
| **蒸馏** | 10-100x | 5-10% | 10x | 特定 |

### 5.2 组合效果

| 组合 | 压缩比 | 精度 |
|------|--------|------|
| INT8 | 4x | 97% |
| INT4 | 8x | 94% |
| INT4 + 剪枝 | 16x | 92% |
| 蒸馏 + INT8 | 40x | 90% |

---

## 6. 最佳实践与常见陷阱

### 6.1 最佳实践

1. **先量化**: 先尝试INT8，效果不佳再试INT4
2. **校准数据**: 静态量化需要代表性数据
3. **层选择**: 某些层对量化敏感，保持FP32
4. **逐步压缩**: 不要一次压缩太多
5. **评估全面**: 关注准确率和延迟

### 6.2 方法选择

```
部署环境?
├── 服务器GPU → FP16/INT8
├── 服务器CPU → INT8
├── 边缘设备 → INT4/蒸馏
└── 手机/IoT → 蒸馏 + INT4
```

---

## 7. 总结

模型压缩是部署大模型的必备技术：

1. **量化**: INT8是通用选择，INT4用于极限压缩
2. **剪枝**: 结构化剪枝更易部署
3. **蒸馏**: 训练时压缩，效果最佳但成本高
4. **组合**: 多种方法组合使用

**推荐策略**:
- 服务器: INT8量化
- 边缘: INT4量化
- 手机: 蒸馏 + INT4
- 成本敏感: GPTQ/AWQ
