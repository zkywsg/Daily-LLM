---
name: "Flamingo"
year: 2022
family: "09-multimodal-clip"
order: 3
paper: "Flamingo: a Visual Language Model for Few-Shot Learning"
authors: ["Jean-Baptiste Alayrac", "Jeff Donahue", "Pauline Luc", "Antoine Miech", "Iain Barr", "Yana Hasson", "et al."]
key_idea: "冻结大 LLM(Chinchilla 70B)+ Perceiver Resampler 视觉适配 + 间隔 cross-attention 注入,8 例 in-context 学新视觉任务的少样本 VLM 范式"
---

## 前作进展

到 2022 年中,VLM 训练有两条主流路线但都有明显缺陷:

**1. 从零联合训练**(ViLBERT / Unified-IO / Florence)—— 视觉和语言模块同时训练,计算成本极高;且 LLM 部分不能从顶级语言模型(GPT-3)起步(因为这些模型不开源/不能从零训)

**2. 双塔对比(CLIP)**—— 只能匹配不能生成 / 对话 / 推理

社区急需一条路线:**充分利用现有的顶级大 LLM(zero-shot 能力强、in-context learning 涌现)+ 加上视觉理解**。DeepMind 团队 2022 年 4 月发表 *Flamingo: a Visual Language Model for Few-Shot Learning* 给出答案——**冻结一个 70B Chinchilla LLM + 加上轻量视觉适配模块**,让模型同时拥有 LLM 的语言能力和视觉理解能力。

关键创新:**Flamingo 直接继承了 LLM 的 in-context learning 能力**。给 4-8 个 (图, 文本) few-shot 例子作为 prompt,Flamingo 就能 zero-shot 完成新视觉任务——这种 "show, don't tell" 的能力是之前所有 VLM 都不具备的。

Flamingo 在 16 个视觉理解 benchmark 上 4-shot in-context 达到 SOTA,在 6 个 benchmark 上甚至超过当时最强的 fine-tuned 模型。这是 VLM 第一次展现"通用智能"的雏形——一个模型不微调就能做任何视觉任务。

Flamingo 没开源,但它定义的 "冻结 LLM + 视觉适配 + cross-attention 注入" 架构直接影响了 IDEFICS(2023, HuggingFace 开源复现 Flamingo)、Otter、Qwen-VL 等后续模型。

## 核心思想

### 直觉:冻结大 LLM + 视觉接口 + 交错数据,继承 in-context learning 能力

理解 Flamingo 真正需要先抓一件事:**[GPT-3](../07-gpt-scaling/03-gpt3.md) 的 in-context learning 是 175B 大 LLM 涌现的能力,这能力本质上和"模态"无关 — 给几个例子,模型就能学新任务**。前作 ViLBERT / Florence 从零联合训练 VLM,**LLM 部分根本起不到这个规模**(都是 BERT-level)。CLIP 只能匹配不能生成。BLIP-2 用 11B LLM 但 in-context 弱。Alayrac 等人 2022 反问:**为什么不冻结一个 70B Chinchilla LLM、加一个轻量视觉接口、用交错图文序列训练,让 LLM 把单模态的 in-context learning 能力直接迁移到多模态?**

三件事必须同时成立才让 Flamingo 在 2022 年成立:

- **冻结 70B Chinchilla LLM,保留其 in-context learning 能力** — LLM 不动,只训视觉适配 + cross-attention 注入层;**70B 才是 in-context learning 涌现的关键阈值**,BLIP-2 的 11B Flan-T5 完全做不到
- **Perceiver Resampler + 间隔 gated cross-attention** — Perceiver 把可变大小视觉特征压成 64 个固定 token / gated cross-attn 在 LLM 每 7 层插一次,**tanh(0) 初始化保证训练初期完全等于原 LLM**
- **MultiModal MassiveWeb(M3W)交错图文序列** — 43M 网页,每个是 `"text image text image text..."` 自然交错;**这种数据天然教模型"看到图后预测对应文本",in-context learning 自然涌现到多模态**

三件事合起来:**Flamingo 4-shot in-context 在 16 个视觉 benchmark 上 SOTA,6 个 benchmark 上超过 fine-tuned 模型**(VQAv2 32-shot 60.0、OK-VQA 50.6)。这是 VLM 第一次展现**"通用智能"雏形** — 不微调就能做任何视觉任务。**核心方法论贡献**:展示了"大 LLM 是 in-context learning 的载体,视觉只是新增的输入模态" — 这一思想直接影响 GPT-4V / Claude 3 / Gemini 的多模态训练范式,**让所有现代 multimodal LLM 都基于"冻结大 LLM + 视觉接口"路线**。

![Flamingo vs BLIP-2 vs LLaVA — In-Context Learning 来源](assets/03-flamingo-icl-paradigm.svg)
*图 1:三个开源 VLM 对比 — **Flamingo** 用 Chinchilla 70B(冻结)+ Perceiver Resampler + 间隔 cross-attn + M3W 交错数据 → 4-shot ICL VQAv2 56.3;**BLIP-2** 用 Flan-T5 11B(冻结)+ Q-Former + 标准图文对 → 0-shot VQAv2 65.2 但无 ICL;**LLaVA** 用 LLaMA 7B(微调)+ Linear projection + instruction tuning → 强对话但 ICL 弱。底部 callout:**70B LLM + 交错数据是 ICL 涌现的核心两件**,Flamingo 第一次在 VLM 上验证。*

## 机制一:冻结 70B LLM + Vision Encoder — 保留 LLM 全部能力

Flamingo 的架构由三部分组成:

```mermaid
graph LR
    img["Image / Video frames"]:::input --> vit["NF-ResNet F6<br/>Vision Encoder<br/>(冻结)"]:::compute
    vit --> percv["Perceiver<br/>Resampler<br/>(可训练)"]:::compute
    percv --> visual_emb["64 visual tokens<br/>(固定数量)"]:::compute
    txt["Text tokens"]:::input --> llm["Chinchilla 70B<br/>(冻结)"]:::compute
    visual_emb -.-> xattn["Cross-Attention<br/>层(可训练)<br/>间隔插入 LLM"]:::compute
    llm --> xattn
    xattn --> out["Generated text"]:::output

    classDef input fill:#fef3c7,stroke:#d97706,color:#92400e;
    classDef compute fill:#fce7f3,stroke:#db2777,color:#9d174d;
    classDef output fill:#ecfdf5,stroke:#059669,color:#065f46;
```

*图 1:Flamingo 架构 — 冻结视觉编码器 + 冻结 70B LLM,中间加 Perceiver Resampler 把视觉特征统一成 64 个 tokens,通过间隔插入的 cross-attention 注入 LLM。可训练参数只占 10%。*

**Vision Encoder(冻结,435M)** —— NF-ResNet F6,DeepMind 自己的 ResNet 变体。从图像或视频帧中提取 patch features。任意分辨率任意视频长度都能处理。

**LLM(冻结,Chinchilla 70B)** —— Flamingo 的核心选择。**70B 是 in-context learning 涌现的阈值** — BLIP-2 用 Flan-T5 11B 完全做不到 ICL,Flamingo 用 70B 才能"看几个例子学新任务"。Chinchilla(DeepMind 2022)是当时 SOTA LLM,语言能力 + ICL 能力都顶级。

**为什么必须冻结 LLM?** 三个原因:

1. **保留 LLM 的全部语言能力** — 微调会破坏 LLM 学到的 zero-shot / ICL / 推理能力
2. **训练成本可控** — 70B 全微调需要几百万美元算力,冻结只训新加层(~10B 可训练),成本降到 ~$1M
3. **模块化复用** — 同一个 LLM 可以接不同视觉模块做不同模态扩展(图像 / 视频 / 音频)

冻结 LLM + 加视觉接口是 Flamingo 的根本架构选择,这一选择被后续 BLIP-2 / LLaVA / Qwen-VL 全部继承(只是接口形式不同)。

## 机制二:Perceiver Resampler + 间隔 Gated Cross-Attention

视觉特征怎么进入冻结的 LLM?Flamingo 设计了两层接口:

**Perceiver Resampler(可训练,200M)** —— **关键创新**。视觉编码器输出的 patch features 数量随分辨率变化(64×64 vs 16×16),视频还有时序维度。Perceiver Resampler 用一组**固定数量的可学习 query tokens(64 个)**来"采样"任意大小的视觉特征:

- Query 是 64 个可学习 token
- Key/Value 是视觉编码器的输出 patches(数量可变)
- 多层 cross-attention 让 64 个 queries 提炼出"图像/视频里关键的语言相关信息"
- 输出统一是 `[64, hidden_dim]` — 任意输入大小都变成 64 tokens

这一思路和 [BLIP-2 的 Q-Former](02-blip.md) 几乎一样,只是 **Flamingo 早 9 个月发布**。两者并行独立发现"用 learned queries 桥接视觉和语言"的范式。

**间隔 Gated Cross-Attention(可训练,200M)** —— Flamingo 不把视觉 tokens 直接拼到 LLM 输入,而是**在 LLM 内部每隔 7 层插入一个新的 cross-attention 层**,让 LLM 的 hidden state 可以 attend 到 64 个视觉 tokens:

```
LLM original block:
    self-attention → FFN              ← 冻结

Flamingo block(每 7 层插入一次):
    self-attention → FFN              ← 冻结
    gated cross-attention → FFN        ← 新加,只这层可训练
```

**Gated cross-attention 的"gated"是关键** — 在残差连接里加一个可学习的 `tanh(α)` 门控,初始化 α=0:

$$
x \leftarrow x + \tanh(\alpha_{\text{attn}}) \cdot \text{CrossAttn}(x, V) + \tanh(\alpha_{\text{ffn}}) \cdot \text{FFN}(x)
$$

`tanh(0) = 0` 让训练初期 cross-attention 完全是 identity,**Flamingo 一开始的 forward 严格等于原 LLM forward**;gate 随训练逐渐学到非零,视觉影响才开启。这一设计**避免了"加视觉模块导致 LLM 语言能力退化"的常见问题**,后被 LoRA / Adapter / Prompt Tuning 等 PEFT 工作沿用。

## 机制三:M3W 交错图文序列 — 让 ICL 涌现到多模态

Flamingo 训练数据的核心是 **MultiModal MassiveWeb(M3W)** — 43M 网页, 185M 图像,每个网页是 `"text image text image text..."` 的自然交错序列:

```
"今天去公园看到一只可爱的金毛 [金毛图片] 它正在追蝴蝶。
旁边还有一只小柯基 [柯基图片],两只狗一起玩得很开心。
最后看到一只猫 [猫图片] 在树荫下打盹。"
```

**这种数据形态是 ICL 涌现到多模态的物理基础** — LLM 在预训练时已学到"看到上下文模式,猜下一个 token"的元学习能力;M3W 提供"看到几个 (图, 对应文本) 模式,预测下一对"的多模态版本。**Flamingo 自然学到"few-shot 学新视觉任务"**。

四个数据集按比例混合训练:

| 数据集 | 描述 | 规模 |
|---|---|---|
| **M3W** | 交错图文网页(关键!) | 43M 网页, 185M 图像 |
| **ALIGN** | 短 caption 图文对 | 1.8B 对 |
| **LTIP** | 长 caption 图文对 | 312M 对 |
| **VTP** | 短视频 + caption | 27M 对 |

M3W 提供 ICL 模式,ALIGN/LTIP 提供短/长 caption 监督,VTP 提供视频时序能力。**没有 M3W,Flamingo 即使用 70B LLM 也学不到多模态 ICL** — 这是机制三与机制一/二同等重要的原因。

## 三件套协同:冻结 70B LLM + 视觉接口 + M3W 交错数据 缺一不可

Flamingo 在 2022 年能让 VLM 第一次展现"通用智能"雏形,**不是单一改进**,而是三件套同时调到协同点 —— 任何一个抽掉 Flamingo 都不成立,这一点和 [ResNet](../01-cnn/05-resnet.md) 的 `shortcut + BN + He 初始化` 协同关系一致:

- **只有 70B 冻结 LLM + 视觉接口,没有 M3W 交错数据** — 退化成普通 VLM(类似 BLIP-2),0-shot 能做 captioning / VQA,**但无法 few-shot 学新任务**;LLM 的 ICL 能力没有数据形态触发,无法迁移到多模态
- **只有 M3W + 视觉接口,没有 70B 大 LLM(用 11B Flan-T5)** — ICL 是 70B+ LLM 才涌现的能力,11B 模型即使在交错数据上训练,**4-shot 性能几乎不超过 0-shot**(BLIP-2 实测)
- **只有 70B LLM + M3W,没有 gated cross-attention 视觉接口** — 视觉特征无法注入 LLM,或者注入方式破坏 LLM 原始能力(普通 cross-attention 没有 gate 训练初期会扰动 LLM 输出),**Flamingo 的"保留全部 LLM 能力 + 加视觉理解"做不到**

三件套合起来才让 Flamingo 在 16 个视觉 benchmark 上 4-shot SOTA,**6 个 benchmark 上 4-shot 超过 fine-tuned 模型**。**核心方法论贡献**:LLM 是 in-context learning 的载体,只要视觉接口设计得当(不破坏 LLM)+ 训练数据有交错模式,**ICL 能力直接迁移到多模态**。这一发现直接影响 GPT-4V / Claude 3 / Gemini 等所有现代 multimodal LLM — 它们的多模态 ICL 能力本质上都来自这条路线。

![Flamingo 架构 + Gated Cross-Attention 训练动态](assets/03-flamingo-gated-attention.svg)
*图 2:**上半** Flamingo 完整架构 — 图像/视频 → 冻结 NF-ResNet F6 → Perceiver Resampler(64 query tokens cross-attn 可变视觉特征 → 固定 64 tokens)→ 冻结 Chinchilla 70B(每 7 层插入 gated cross-attn,只这些层可训练)→ 生成文本。**下半** Gated Cross-Attention 训练动态曲线 — α 初始化 0,前 1K steps 几乎为 0(Flamingo ≡ 原 LLM),1K-10K 逐步学到非零(视觉信号开始影响 LLM),10K+ 趋于稳定值;**初始 identity 保护 LLM 原始能力,逐步开启视觉**。底部 callout:tanh(0)=0 这一 trick 后被 LoRA / Adapter / Prompt Tuning 等 PEFT 工作沿用。*

## In-Context Learning 能力

Flamingo 真正的差异化能力是**多模态 in-context learning**:

```
Example 1: [image of dog] -> "A photo of a dog."
Example 2: [image of cat] -> "A photo of a cat."
Example 3: [image of bird] -> "A photo of a bird."
Query:     [image of fish] -> ?
```

Flamingo 给出 "A photo of a fish." —— **完全没在这种任务上训过,但学到了 prompt 里的"看图生成 caption"格式**。

为什么 Flamingo 能 in-context learning,BLIP-2 不能?核心区别:

- **Flamingo backbone 是 Chinchilla 70B** —— 这是当时 SOTA LLM,in-context learning 能力强
- **BLIP-2 backbone 是 Flan-T5 / OPT** —— 都不到 11B,in-context learning 弱
- **Flamingo 训练数据混合了交错 image-text 序列** —— 论文用 MultiModal MassiveWeb(M3W, 43M 网页),每个网页是 "text image text image text..." 的交错序列。LLM 在这种数据上自然学到"看到图后预测对应文本"的模式

具体 in-context 能力(论文 Table 1):

| 任务 | 0-shot | 4-shot | 32-shot |
|------|------|------|------|
| VQAv2 | 49.2 | 56.3 | **60.0** |
| OK-VQA | 41.2 | 47.4 | **50.6** |
| TextVQA | 30.1 | 32.7 | **36.0** |
| NoCaps CIDEr | 92.7 | 99.0 | — |

从 0-shot 到 32-shot 提升 5-11 分,**clear in-context learning 信号**。在 6 个 benchmark 上 4-shot 甚至超过 fine-tuned SOTA,证明 Flamingo 实现了 "通用视觉智能" 的雏形。

## 训练细节

| 维度 | Flamingo 80B |
|------|------|
| Vision encoder | NF-ResNet F6, 435M, **冻结** |
| Perceiver Resampler | 200M, **可训练** |
| LLM | Chinchilla 70B, **冻结** |
| Cross-attention 层 | 插入到 LLM 每 7 层一次, 总计 ~200M, **可训练** |
| 总参数 | 80B(其中 ~10B 可训练 ≈ 12.5%) |
| 训练数据 | 4 个数据集混合,~3000B token-equivalent |
| Batch size | 1024(主要算力 = LLM 的 forward) |
| 训练硬件 | ~1500 × TPU v4 |
| 训练时间 | ~15 天 |
| 训练成本 | 估计 $1M+ |

注意 Flamingo 训练成本仍然很高(虽然不微调 LLM,但 70B 的 forward 在 1500 TPU 上也很贵)。**BLIP-2 用 Q-Former 把训练成本降到 1/100**,但 Flamingo 因为 LLM 更大、保留了 in-context learning,质量上限更高。

## 关键代码

Flamingo 的核心是 **Perceiver Resampler + gated cross-attention** 两个新组件。这里展示 gated cross-attention 块:

```python
import torch
import torch.nn as nn

class GatedCrossAttention(nn.Module):
    """Flamingo 在 LLM 内部插入的 cross-attention 层
    - 让 LLM hidden state 可以 attend 到视觉 tokens
    - 用 gated 残差,初始 identity,逐步开启"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # 关键:gated 残差,初始 tanh(0) = 0,不影响原 LLM 行为
        self.attn_gate = nn.Parameter(torch.tensor(0.0))

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.ffn_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, visual_tokens):
        # x: [B, T_text, dim] LLM hidden state
        # visual_tokens: [B, T_vis, dim] Perceiver Resampler 输出
        attn_out, _ = self.cross_attn(
            query=self.norm1(x),
            key=visual_tokens,
            value=visual_tokens,
        )
        # 关键:tanh(gate) 让训练初期是 identity
        x = x + torch.tanh(self.attn_gate) * attn_out
        # FFN 也 gated
        ffn_out = self.ffn(self.norm2(x))
        x = x + torch.tanh(self.ffn_gate) * ffn_out
        return x

class PerceiverResampler(nn.Module):
    """把任意大小的视觉特征统一压成 64 个 tokens"""
    def __init__(self, dim, num_latents=64, num_layers=6, num_heads=8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))  # 可学的 query
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

    def forward(self, visual_features):
        # visual_features: [B, N_patches, dim] - 可变大小
        B = visual_features.size(0)
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B, 64, dim]
        for layer in self.layers:
            # cross-attention: latents 查询 visual_features
            attn_out, _ = layer(latents, visual_features, visual_features)
            latents = latents + attn_out
        return latents  # [B, 64, dim] 固定大小

# Flamingo 完整结构(简化)
class Flamingo(nn.Module):
    def __init__(self, frozen_vision, frozen_llm, perceiver, dim, num_heads,
                 cross_attn_every_n_layers=7):
        super().__init__()
        self.vision = frozen_vision
        self.llm = frozen_llm  # 包含 num_layers 个 self-attention block
        self.perceiver = perceiver
        # 在 LLM 每 7 层插入一个 cross-attention
        self.cross_attns = nn.ModuleList([
            GatedCrossAttention(dim, num_heads)
            for _ in range(self.llm.num_layers // cross_attn_every_n_layers)
        ])

    def forward(self, image, text_tokens):
        with torch.no_grad():
            vision_feat = self.vision(image)               # 冻结 forward
        vis_tokens = self.perceiver(vision_feat)            # [B, 64, dim] 可训练
        # 走 LLM,在指定层插入 cross-attention
        x = self.llm.embed(text_tokens)
        for i, block in enumerate(self.llm.blocks):
            x = block(x)  # 冻结 self-attention + FFN
            # 每 7 层插入一次视觉 cross-attention
            if (i + 1) % 7 == 0:
                cross_idx = (i + 1) // 7 - 1
                x = self.cross_attns[cross_idx](x, vis_tokens)
        return self.llm.head(x)  # 输出 logits
```

工程要点:

- **`torch.tanh(self.attn_gate)` 初始 0** —— 让 Flamingo 训练初期完全等于原 LLM forward;gate 学到非零后视觉影响才开启,避免训练初期破坏 LLM 能力
- **`cross_attn_every_n_layers=7`** —— Flamingo-80B 用 7,小模型可以用更小(每 4 层一次);太密则参数多训练慢,太疏则视觉影响弱
- **with torch.no_grad() 视觉部分** —— 节省显存和算力

## 影响 / 后续

Flamingo 在 VLM 历史的位置:**展示了"大 LLM + 视觉接口"的 in-context learning 潜力**。具体影响:

**1. 定义了 VLM 的开源参考架构** —— Flamingo 没开源,但论文的"冻结 LLM + Perceiver Resampler + 间隔 cross-attention"被 IDEFICS(HuggingFace, 2023)开源复现,后被 Otter / OpenFlamingo / IDEFICS-2 等迭代

**2. In-context learning 在多模态上的实证** —— 证明 LLM 的 in-context learning 能力可以扩展到多模态,只要训练数据有"交错图文序列"。这一观察直接影响 GPT-4V / Claude 3 / Gemini 的多模态训练范式

**3. Gated cross-attention 思想被广泛采用** —— "在冻结模型里插可训练 gated 层"被 LoRA、prompt tuning 等参数高效微调方法借鉴

**4. Perceiver Resampler 与 Q-Former 并列**——同样是"用 learned queries 桥接视觉和语言"的范式,Flamingo 和 BLIP-2 几乎同时独立发现。后续 LLaVA 选了更简单的 linear projection,但思想都来自这一脉

**5. 推动 VLM 评估的 in-context 维度** —— Flamingo 论文设计的 0/4/8/16/32-shot 评估方法后被 VLM 社区广泛采用

Flamingo 留下的问题:

- **训练成本仍然高** —— 1500 TPU × 15 天对开源社区不可行 → BLIP-2(16 A100 × 9 天)
- **闭源** —— 学界等了一年才有 IDEFICS 复现 → 开源 VLM 浪潮在 2023 年才真正爆发
- **缺少对话 / 指令跟随** —— Flamingo 是 base model,需要后续指令微调才能做对话 → [LLaVA](04-llava.md) 加 instruction tuning

→ [04-llava.md](04-llava.md) · 简化架构 + instruction tuning,开源 VLM 标准
→ [02-blip.md](02-blip.md) · 平行路线,Q-Former 替代 Perceiver Resampler
→ [01-clip.md](01-clip.md) · 视觉特征对齐的基础
→ [../07-gpt-scaling/03-gpt3.md](../07-gpt-scaling/03-gpt3.md) · in-context learning 思想源头,Flamingo 把它扩展到多模态
→ [../07-gpt-scaling/05-gpt4-llama.md](../07-gpt-scaling/05-gpt4-llama.md) · Chinchilla 是 Flamingo 的 LLM backbone
