---
name: "Adapter Tuning"
year: 2019
family: "11-peft-lora"
order: 1
paper: "Parameter-Efficient Transfer Learning for NLP"
authors: ["Neil Houlsby", "Andrei Giurgiu", "Stanislaw Jastrzebski", "Bruna Morrone", "Quentin de Laroussilhe", "Andrea Gesmundo", "Mona Attariyan", "Sylvain Gelly"]
key_idea: "在每层 Transformer 插入 small bottleneck adapter 模块(down → ReLU → up + residual),base 模型完全冻结,只训 3% 参数达到全参微调 96% 性能;PEFT 起源,后续 LoRA / Prefix Tuning 都受其启发"
---

## 前作进展

2018 年 [BERT](../06-bert-family/01-bert.md) 发布,"预训练 + 微调"成为 NLP 范式。但全参微调(Full Fine-Tuning)很快暴露工程问题:

**1. 存储爆炸** —— GLUE 9 个任务每个微调一份 BERT-large(340M),总共 9 × 1.3GB ≈ 12GB。如果再扩到 36 个任务(SuperGLUE / SQuAD / NER 等),存储和分发成本巨大

**2. 训练成本高** —— 每个任务都要更新全部 340M 参数,需要大 batch + 多 epoch + 优化器状态

**3. 灾难性遗忘** —— 在新任务微调,模型可能丢失原 pretrain 学到的通用知识

**4. 没法 multi-task 高效** —— 想让一个模型同时擅长 GLUE 9 任务,要么 multi-task 训练(数据 mix 难调),要么 9 个独立 checkpoint(部署时切换慢)

Rebuffi 等人(2017)在 CV 上提出过"residual adapter"——给冻结的 ResNet 加 small adapter 适配新任务。但 NLP 领域 2018 年还没有等效工作。

Houlsby 等人(Google,2019 年 2 月)的论文把这一思路完整移植到 NLP,论文标题就是 **"Parameter-Efficient Transfer Learning for NLP"**——第一次正式提出"参数高效微调"这一术语。论文展示:

- **只训 3.6% 参数,达到全参微调 96.1% 性能**
- **9 个 GLUE 任务,共享同一 BERT base**,每任务只多 0.9M adapter 参数
- **不会灾难性遗忘**,base 完全冻结

这奠定了所有现代 PEFT 工作的基本理念——**冻结 base + 训小模块**。

## 核心思想:Bottleneck Adapter

在每层 Transformer block 里加入 adapter 模块。BERT 每层有两个位置:**attention 后** 和 **FFN 后**。

### Adapter 结构

每个 adapter 是一个 bottleneck:

```
input x (d=768)
   ↓
down: Linear(768 → 64)    # 压缩到 bottleneck 维度 r
   ↓
ReLU(或 GELU)
   ↓
up: Linear(64 → 768)      # 恢复
   ↓
+ x(residual,关键!)
   ↓
output
```

数学形式:

$$
h = x + W_{\text{up}}(\sigma(W_{\text{down}} \cdot x))
$$

**关键设计点 1:Residual 初始化为 0**

$W_{\text{up}}$ 初始化为接近 0,确保 adapter 在训练开始时几乎不影响 base 模型——`h ≈ x`。这样模型从"冻结 BERT"平滑过渡到"BERT + 小调整",不会一上来就 disrupt 预训练表示。

**关键设计点 2:Bottleneck 维度 r 小**

论文用 r=64(BERT-base d=768 → 64),adapter 参数 = 2 × 768 × 64 = 98K 每层。12 层 = 1.2M 参数 / 任务。BERT-large 总参 340M 的 0.36%。

### 插入位置

Adapter 在 BERT 每层插入两次:

```
Layer N:
  x
   ↓ Multi-Head Attention
   + skip
   ↓ LayerNorm
   ↓ Adapter₁ ← 新增
   + skip
   ↓ FFN
   + skip
   ↓ LayerNorm
   ↓ Adapter₂ ← 新增
   ↓ output to Layer N+1
```

训练时:**所有 BERT 原参数冻结,只有 Adapter₁、Adapter₂、LayerNorm 参数、最后的任务 head 参与更新**。

### 后续 Adapter 变体

Houlsby 2019 之后大量 adapter 变体:

- **Pfeiffer Adapter(2020)** —— 只在 FFN 后加一个 adapter(更省参数)
- **Parallel Adapter** —— adapter 与 attention/FFN 并行,而非串行
- **Compacter** —— 用低秩 + Kronecker 进一步压缩 adapter
- **AdapterFusion** —— 把多任务 adapter 通过 attention 融合,做 multi-task

## 关键代码

PyTorch 实现一个 Houlsby Adapter,插入到 HuggingFace BERT:

```python
import torch
import torch.nn as nn

class HoulsbyAdapter(nn.Module):
    def __init__(self, hidden_size=768, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_size)
        # 关键:up 初始化为 0,保证 residual 在 train 开始时 ≈ 0
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


# 把 adapter 注入到 BERT 的每层
from transformers import BertModel

def add_adapters_to_bert(model: BertModel, bottleneck=64):
    for layer in model.encoder.layer:
        # attention 后 adapter
        layer.attention.output.adapter = HoulsbyAdapter(768, bottleneck)
        original_attn_forward = layer.attention.output.forward
        def new_attn_fwd(hidden_states, input_tensor, _orig=original_attn_forward, _ad=layer.attention.output.adapter):
            h = _orig(hidden_states, input_tensor)
            return _ad(h)
        layer.attention.output.forward = new_attn_fwd
        # FFN 后 adapter
        layer.output.adapter = HoulsbyAdapter(768, bottleneck)
        original_ffn_forward = layer.output.forward
        def new_ffn_fwd(hidden_states, input_tensor, _orig=original_ffn_forward, _ad=layer.output.adapter):
            h = _orig(hidden_states, input_tensor)
            return _ad(h)
        layer.output.forward = new_ffn_fwd

# 冻结 base,只训 adapter + classifier head
model = BertModel.from_pretrained("bert-base-uncased")
add_adapters_to_bert(model)
for name, p in model.named_parameters():
    p.requires_grad = "adapter" in name or "LayerNorm" in name

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable / total:.2%}")
# → 约 0.9%(只 adapter + LayerNorm)
```

现代实现可以用 `adapter-transformers` / `peft` 库一行调用,但底层都是这套机制。

## 性能数据

Houlsby 2019 在 GLUE 9 任务上对比 Full FT vs Adapter Tuning(BERT-large):

| 任务 | Full FT(340M) | Adapter(0.9M) | 差距 |
|------|------|------|------|
| MNLI | 86.7 | 84.9 | -1.8 |
| QQP | 89.6 | 88.3 | -1.3 |
| QNLI | 92.7 | 91.4 | -1.3 |
| SST-2 | 94.9 | 93.5 | -1.4 |
| CoLA | 60.5 | 56.9 | -3.6 |
| STS-B | 86.5 | 84.7 | -1.8 |
| MRPC | 89.3 | 86.9 | -2.4 |
| RTE | 70.1 | 71.8 | **+1.7** |
| **GLUE Avg** | **83.7** | **82.5** | **-1.2(96.6% Full)** |

关键观察:

- **3% 参数 → 96.6% 性能** —— 大部分任务差距 < 2 点,部分任务(RTE)甚至 adapter 更好
- **小数据任务上 adapter 更强**(RTE, CoLA) —— 小数据时 Full FT 容易过拟合,adapter 的参数限制反而是正则
- **存储节省巨大** —— 9 任务总存储:Full FT 12GB → Adapter 1.3GB + 9 × 8MB = 1.4GB(9× 节省)

SQuAD QA 任务:

| Model | F1 | EM |
|------|------|------|
| BERT-large Full FT | 90.9 | 84.1 |
| Adapter(r=256) | 90.5 | 83.4 |

QA 这种长输出任务上 adapter 表现同样接近全参。

## 影响 / 后续

Houlsby 2019 在 LLM 历史的位置:**PEFT 起源,直接定义了"冻结 base + 训小模块"范式**。

**1. 大量 adapter 变体涌现** —— 2019-2021 两年里 Pfeiffer Adapter / Parallel Adapter / Compacter / MAD-X / AdapterFusion 等几十种变体。AdapterHub.ml 收集了 100+ 不同任务的 adapter 权重

**2. PEFT 概念被正式提出** —— "Parameter-Efficient Transfer Learning" / "Parameter-Efficient Fine-Tuning" 的术语和研究方向被建立。后来 HuggingFace 的 PEFT 库就以此命名

**3. 启发 LoRA 等后续工作** —— [LoRA](03-lora.md)(2021)的 BA 低秩分解可以看作 adapter 的"线性化"——adapter 的非线性(ReLU)被去掉,变成线性低秩,获得"推理零延迟"的关键优势

**4. Adapter 在多语言 / 多任务上找到主战场** —— MAD-X(2020)用 adapter 做 cross-lingual transfer,每语言一个 adapter;后续多语言模型(XLM-R adapter, mBART adapter)广泛使用 adapter

**5. 工业部署优势** —— 一个 base + N 个 adapter 的"插件式"部署被 Hugging Face / 各大云服务采用。模型路由 / multi-tenant serving 等场景里 adapter 比 LoRA 更早被工程化

**6. 不能合并的局限** —— Adapter 的非线性结构无法合并回 base 权重,推理时必须保留 adapter 层,引入 5-10% 延迟开销。这一短板直接催生 LoRA(可合并)

Houlsby 2019 留下的开放问题:

- **推理延迟** —— Adapter 层无法合并,每次 forward 多算一次 → LoRA 解决
- **参数效率有限** —— 3% 参数已经不少,能不能更少 → Prefix/Prompt Tuning(0.1%)
- **超参数敏感** —— bottleneck r 怎么选、放哪些层都需要调 → 后续 AutoML for PEFT
- **不能跨任务组合** —— 两个 adapter 怎么组合在一个 forward 里 → AdapterFusion

→ [02-prefix-tuning.md](02-prefix-tuning.md) · 更极致的参数效率,只训 soft prompt
→ [03-lora.md](03-lora.md) · 线性化 adapter,推理零延迟,工业标配
→ [04-qlora.md](04-qlora.md) · LoRA + 4-bit 量化,极致显存效率
→ [../06-bert-family/01-bert.md](../06-bert-family/01-bert.md) · Adapter 最早在 BERT 上验证
→ [../05-transformer/01-transformer.md](../05-transformer/01-transformer.md) · Adapter 改的是 Transformer block 结构
