---
name: "DiT"
year: 2022
family: "08-vit"
order: 4
paper: "Scalable Diffusion Models with Transformers"
authors: ["William Peebles", "Saining Xie"]
key_idea: "把 diffusion 模型的 U-Net backbone 替换成 ViT-style Transformer,展示更好的 scaling 性质,成为 Stable Diffusion 3 / Sora 的基座"
---

## 前作进展

到 2022 年中,生成式视觉模型的格局是:

- **GAN 系**(StyleGAN 等)—— 2014–2020 主流,但训练不稳、模式坍塌、难以条件生成
- **Autoregressive 系**(PixelRNN, ImageGPT)—— 顺序生成像素,慢且质量一般
- **Diffusion 系**(Stable Diffusion, Imagen, DALL-E 2)—— 2022 新王,质量高、稳定、可条件控制

但**所有 diffusion 模型的 backbone 都是 U-Net**——卷积下采样 + skip connection + 卷积上采样的经典视觉架构。U-Net 在 2015 年医学影像分割上提出,后来被 DDPM(2020 Ho)选作 diffusion 的去噪网络,从此成为 diffusion 的默认 backbone。

但 U-Net 有几个明显问题:

1. **不享受 Transformer 的 scaling law** —— 加参数 / 加数据,U-Net 的 loss 改善规律不像 Transformer 那么干净。在 [GPT/ViT scaling](../07-gpt-scaling/04-scaling-laws.md) 时代,这是结构性弱点
2. **架构选择多** —— U-Net 的下采样深度、skip 通道数、attention 插入位置都需要手调,没有统一原则
3. **跨任务难统一** —— 想从图像 diffusion 扩展到视频 / 3D / 多模态,U-Net 都要重新设计

2022 年 12 月 Peebles & Xie(UC Berkeley + NYU)发表 *Scalable Diffusion Models with Transformers*(DiT),做了一个简单又激进的实验:**把 diffusion 的 U-Net 完全替换成 ViT-style 的 Transformer,看看 scaling 性质如何**。结果:

- **DiT-XL/2(675M 参数)在 ImageNet 256² 类条件生成上拿到 FID 2.27**——SOTA,且训练稳定
- **scaling law 干净**——FID 随参数 / 算力按预测的幂律下降
- **架构选择极简**——一个超参 patch size,一个超参深度

这一结果在 2022 年并没引起太大反响——直到 2023 年的 **Stable Diffusion 3** 和 2024 年的 **OpenAI Sora** 公开声明用 DiT 作为基座,这才让 DiT 成为 2024 视觉生成的新基础架构。今天看 DiT 的影响力,这是把 Transformer 完成"视觉理解 → 视觉生成"全覆盖的关键一步。

## 核心思想:U-Net → ViT 替换

DiT 的架构本质就是 [ViT](01-vit.md) 加几个 diffusion 特定的输入条件:

```mermaid
graph LR
    img["Noisy image x_t<br/>(latent)"]:::input --> patch["Patchify"]:::compute
    patch --> seq["[patches] sequence"]:::compute
    seq --> add["+ timestep emb<br/>+ class emb"]:::compute
    add --> dit["DiT Block × N<br/>(Transformer + AdaLN)"]:::compute
    dit --> head["Linear unpatch<br/>→ noise prediction"]:::output

    classDef input fill:#fef3c7,stroke:#d97706,color:#92400e;
    classDef compute fill:#fce7f3,stroke:#db2777,color:#9d174d;
    classDef output fill:#ecfdf5,stroke:#059669,color:#065f46;
```

*图 1:DiT pipeline — 噪声 latent 切 patch → 加 timestep + class 条件 → Transformer encoder → 输出 patch-wise 噪声预测。整个流程没有 U-Net、没有卷积、没有 skip connection。*

**Step 1: Latent diffusion 输入**——DiT 用在 latent space(同 Stable Diffusion):图像先被 VAE 编码到 `32 × 32 × 4` 的 latent,在 latent 上做 diffusion。这比直接在 pixel space 做 diffusion 省 64× 显存

**Step 2: Patchify**——latent 切 `p × p` patch(典型 p=2),`32 × 32 / 4 = 256` 个 patch。每个 patch 拉平 + linear 投影到 `d_model` 维

**Step 3: 加条件信息**——diffusion 需要两个条件:**timestep `t`** 和 **class label `c`**(类条件生成时)。DiT 测试了 4 种条件注入方式:

- **In-context conditioning** —— `t` 和 `c` 作为额外 token 拼到序列前面
- **Cross-attention** —— 增加 cross-attention 层,query 是 patch,key/value 是 `(t, c)` embedding
- **Adaptive LayerNorm(AdaLN)** —— 让 LayerNorm 的 `γ, β` 参数由 `(t, c)` 通过 MLP 生成
- **AdaLN-Zero** —— AdaLN 加上"零初始化"，让 DiT block 初始时是 identity function

**AdaLN-Zero 效果最好**——FID 比 in-context 好 3+ 分。这是 DiT 的关键工程贡献,后来被 SD3 / Sora 全部沿用

**Step 4: Transformer encoder**——N 层标准 transformer block,但每个 block 的 LayerNorm 用 AdaLN(参数由条件生成)

**Step 5: Linear unpatch**——最后一层接 linear 输出 `p² × C` 维(预测每个 patch 的噪声),再 reshape 回 `H × W × C` 的噪声图

## AdaLN-Zero:DiT 的核心工程贡献

AdaLN(Adaptive LayerNorm)的想法不新——StyleGAN 早就用类似机制(AdaIN)做条件生成。但 DiT 把它应用到 diffusion + Transformer,且做了关键的"零初始化"改造。

**AdaLN 公式**:

$$
\text{AdaLN}(x, c) = \gamma(c) \cdot \frac{x - \mu}{\sigma} + \beta(c)
$$

其中 `γ(c), β(c)` 由一个 MLP 从条件 `c`(timestep + class embed)生成。这一设计的好处:

- **条件信号被注入到归一化层而不是加到 hidden state**——避免污染主信号流
- **MLP 让条件可以非线性变换**——`(t, c)` 通过 MLP 学到丰富的"条件表达"

**AdaLN-Zero 的零初始化**:DiT block 不仅用 AdaLN,还在每个 block 输出处加一个**额外的 scale `α(c)`**,且初始化为 0:

$$
\text{output}_\text{block} = x + \alpha(c) \cdot \text{Block}(x)
$$

`α(c) = 0` 在训练开始时意味着 **DiT block 是 identity function**——网络输出就是输入。这有什么好处?

- **训练初期非常稳定**——梯度直接流过 identity,不会被随机初始化的 block 干扰
- **逐步"开启"复杂性**——`α(c)` 随训练逐渐学到非零,block 逐渐参与计算

这一 trick 来自 ResNet-style residual + ConvNeXt 的"learnable scale" 思想。它让 DiT-XL 这种大模型训练稳定,从随机初始化到 SOTA 不需要 babysit lr 或加 warmup。

## Scaling 实验

DiT 论文最核心的部分是 **scaling 实验** —— 系统展示 DiT 的 scaling 性质比 U-Net 更"干净"。Peebles & Xie 训练了多组 DiT 模型,变化两个维度:

**模型大小**:S(33M)→ B(130M)→ L(458M)→ XL(675M)
**Patch size**:8(8×8 patches = 16 patches) → 4(64 patches)→ **2**(256 patches,更精细)

| 模型 | 参数 | Gflops/forward | FID(class-conditional, 400K steps) |
|------|------|------|------|
| DiT-S/8 | 33M | 1.4 | 68.4 |
| DiT-B/4 | 130M | 5.6 | 35.6 |
| DiT-L/2 | 458M | 23.0 | 9.62 |
| **DiT-XL/2** | **675M** | **29.1** | **6.40** |
| **DiT-XL/2(7M steps)** | 675M | 29.1 | **2.27**(SOTA) |

关键观察:

- **更小的 patch size + 更深的模型 = 更低 FID**——patch=2 比 patch=8 在同算力下显著好(64 个 patch vs 256 个 patch,后者细粒度优势明显)
- **FID 随 GFLOPs 按幂律下降**——和 LM scaling law 一样的形式
- **DiT-XL/2 训 7M steps 拿 2.27 FID**——超过 LDM(latent diffusion)的 3.6,直接 SOTA

这一 scaling 性质让 DiT 成为"可预测投资"的架构——加算力直接换 FID,不像 U-Net 那样需要架构调整。

## 训练细节

| 维度 | DiT-XL/2 |
|------|------|
| 架构 | 28 层 Transformer, d=1152, h=16, **675M 参数** |
| Patch size | 2(在 32×32 latent 上 = 256 patches) |
| 条件注入 | AdaLN-Zero |
| 输入 | VAE-encoded latent(32×32×4)+ timestep + class label |
| Diffusion schedule | DDPM,1000 steps |
| 训练数据 | ImageNet 256(1.3M 图像,1000 类条件) |
| 优化器 | AdamW,lr 1e-4,weight decay 0 |
| Batch | 256 |
| 训练 steps | 7M(对 ImageNet 1.3M 大约 1500 epoch) |
| EMA | decay 0.9999 |
| 训练硬件 | A100 集群,大约 200K GPU hours |

注意训练时间很长(7M steps,大约 200K GPU hours = 800 V100 天)——这是 DiT 在 ImageNet 上的"充分训练"。后来 Sora / SD3 在更大数据上训练时间更长。

## 关键代码

DiT block 的核心是 AdaLN-Zero:

```python
import torch
import torch.nn as nn

def modulate(x, shift, scale):
    """AdaLN 调制:γ * normed_x + β"""
    return x * (1 + scale) + shift

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        # AdaLN-Zero:从条件 c 生成 6 个调制系数(每个 sublayer 3 个:shift/scale/gate)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # c: 条件向量 [B, hidden_size](timestep + class embed 之和)
        # 把 6 个系数拆出来
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        # Self-attention with AdaLN modulation
        x_norm = modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out  # gate 控制 block 贡献
        # MLP with AdaLN modulation
        x_norm = modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x

class DiT(nn.Module):
    def __init__(self, input_size=32, patch_size=2, in_channels=4,
                 hidden_size=1152, depth=28, num_heads=16, num_classes=1000):
        super().__init__()
        self.patchify = nn.Conv2d(in_channels, hidden_size, patch_size, stride=patch_size)
        n_patches = (input_size // patch_size) ** 2
        self.pos_emb = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        self.t_embed = TimestepEmbedder(hidden_size)         # timestep → embed
        self.y_embed = LabelEmbedder(num_classes, hidden_size)  # class → embed
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        # Final layer:输出每个 patch 的 patch_size² * in_channels 维噪声预测
        self.final = nn.Linear(hidden_size, patch_size ** 2 * in_channels)
        # AdaLN-Zero 的关键:adaLN_modulation 最后一层 Linear 权重和 bias 初始化为 0
        self._zero_init()

    def _zero_init(self):
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

    def forward(self, x, t, y):
        # x: [B, 4, 32, 32] noisy latent
        x = self.patchify(x).flatten(2).transpose(1, 2) + self.pos_emb  # [B, 256, d]
        c = self.t_embed(t) + self.y_embed(y)  # [B, d]
        for block in self.blocks:
            x = block(x, c)
        x = self.final(x)  # [B, 256, p²*4]
        # Unpatch 回 [B, 4, 32, 32]
        return unpatchify(x, patch_size=2)
```

工程要点:

- **`elementwise_affine=False` 的 LayerNorm**——γ/β 由外部 AdaLN 提供,LayerNorm 本身不学
- **`_zero_init`**——AdaLN 调制系数的最后一层权重 + bias 全设为 0,这是 AdaLN-Zero 的精髓
- **`gate_msa`, `gate_mlp`** 是额外的 "gate" 调制——初始化为 0 后,整个 block 在训练初期是 identity

## 影响 / 后续

DiT 在视觉历史的位置:**让 Transformer 完成"理解 + 生成"全覆盖**。具体影响:

**1. Stable Diffusion 3(2024)采用 DiT 系架构**——SD3 用 MM-DiT(多模态 DiT,文本和图像在同序列里 attention),完全替代了 SD1/2 的 U-Net。SD3 在图像质量、文字渲染、复杂构图上比 SD1/2 显著提升,部分归因于 DiT scaling

**2. OpenAI Sora(2024 2 月)用 DiT**——Sora 是 OpenAI 第一个 SOTA 视频生成模型,公开声明 backbone 是 "Diffusion Transformer"。视频生成的 spacetime patches 是 DiT patchify 思想的 3D 扩展

**3. 视觉生成 backbone 全面 Transformer 化**——2023 后期 / 2024 几乎所有新出的视觉生成模型都用 DiT 或其变体:Pixart-α、Hunyuan-DiT、Flux 等

**4. AdaLN-Zero 成为条件生成的标配**——后续条件 diffusion / Flow Matching / Rectified Flow 模型几乎都用 AdaLN-Zero。这是 DiT 最具普适性的工程贡献

**5. 多模态生成的统一基础**——DiT 让"图像生成"和"视频生成"和"语音生成"用同一种架构。OpenAI 的 GPT-4o / Sora / Voice Engine 共享 Transformer backbone 是这条路线的延续

**6. ConvNet 在生成上的让位**——如果说 ViT 让 ConvNet 在视觉理解上让位,DiT 完成了同一件事在生成上。今天 ConvNet 在视觉生成的主流应用只剩 VAE 编码器(因为 latent 解码需要)

至此 08-vit 家族 4 节点完整:**[ViT](01-vit.md)(概念证明)→ [DeiT](02-deit.md)(数据效率)→ [Swin](03-swin.md)(层级窗口)→ [DiT](04-dit.md)(应用到生成)**——从 2020 概念到 2024 SOTA,完整覆盖视觉 Transformer 的演化主线。

→ [03-swin.md](03-swin.md) · 视觉 Transformer 在密集预测上的版本,与 DiT 在生成上的版本形成"理解 vs 生成"分工
→ [01-vit.md](01-vit.md) · 父结构,DiT 是 ViT 在生成任务上的应用
→ [../10-diffusion/](../10-diffusion/) · DiT 应用的 diffusion 家族,SD3 / Sora 的基座
→ [../09-multimodal-clip/](../09-multimodal-clip/) · CLIP 的图像编码器是 ViT,DiT 是它的生成版对应
→ [../07-gpt-scaling/04-scaling-laws.md](../07-gpt-scaling/04-scaling-laws.md) · DiT 展示了视觉生成的 scaling law,精神延续
