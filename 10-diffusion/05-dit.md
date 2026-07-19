---
name: "DiT"
year: 2022
family: "10-diffusion"
order: 5
paper: "Scalable Diffusion Models with Transformers"
authors: ["William Peebles", "Saining Xie"]
key_idea: "把 diffusion U-Net 整个换成 Transformer:latent → patchify → Transformer blocks + adaLN-Zero 条件 → 解 patchify。FLOPs 越大 FID 越低单调成立,DiT-XL/2 拿到 ImageNet 256 SOTA,为 Sora/SD3 提供骨架"
---

## 前作进展

[DDPM](01-ddpm.md) 2020 / [LDM](02-ldm.md) 2022 把 diffusion 推到了"个人 GPU 能跑"的形态,但**所有主流 diffusion 模型的骨架仍是 U-Net**——一个 2015 年为医学图像分割设计的 CNN encoder-decoder 结构。U-Net 在 diffusion 上 work 是因为 inductive bias 合适(局部卷积 + 多尺度跳连),但它有两个明显的天花板:

**1. 不容易 scale**——CNN 的 scaling laws 不如 Transformer 干净。把 U-Net 从 100M 推到 1B,FID 改善幅度难以预测,需要大量调架构(改通道数 / 改深度 / 改 attention 位置)
**2. 多模态难统一**——文本编码用 Transformer (CLIP / T5),视觉生成用 CNN,两者表示空间割裂,跨模态注入需要复杂的 cross-attention 设计

2017 年 Transformer 在 NLP 横扫,2020 年 [ViT](../08-vit/01-vit.md) 证明 Transformer 在视觉分类上也能超越 CNN——剩下的问题是:**生成式视觉(diffusion)能不能也彻底替换 CNN 用纯 Transformer?**

UC Berkeley 的 Peebles 和 Saining Xie 在 2022 年 12 月发表 *Scalable Diffusion Models with Transformers*(DiT),给出了完整的实证答案。他们做了一件看似简单但工程上需要很多细节的事:**把 LDM 里的 U-Net 整个换成 Transformer,在 ImageNet 256/512 class-conditional 生成上拿到了新 SOTA(FID 2.27)**。更关键的是,DiT 展示了 diffusion 模型第一次有了**干净的 scaling curve**——FLOPs 越大 FID 越低单调成立,可以像 LLM 那样按算力预算选模型大小。

DiT 不是单一性能突破,而是**架构层面的范式转移**:把视觉生成纳入 Transformer 统一框架,直接催生了 2024 年的 **Sora**(视频 diffusion + Transformer)、**SD3** / **PixArt-α** / **Hunyuan-DiT**(图像 diffusion + Transformer)、**OpenSora** / **Stable Video Diffusion**——今天所有前沿生成模型几乎都是 DiT 系骨架。

## 核心思想

### 直觉:diffusion 不绑定 U-Net,Transformer 一样能跑且更好 scale

理解 DiT 真正需要先抓一件事:**U-Net 之所以是 diffusion 默认骨架,只是历史路径依赖,不是数学必然**。

DDPM 论文用 U-Net 是因为 2020 年 image-to-image 任务的默认结构就是 U-Net,作者直接拿过来加 timestep embedding 用。后续工作 (ADM, GLIDE, LDM, Imagen) 都沿用这个选择——但**没有人系统验证过:diffusion 的 denoising 网络一定要有 U-Net 那种 encoder-decoder + skip connection 吗?**

Peebles & Xie 的反问把这个默认假设直接挑了:既然 [ViT](../08-vit/01-vit.md) 已经证明 Transformer 在图像分类上 work,那同样的"把图像切 patch → 当 token 序列 → 喂 Transformer"思路应该也能用在 denoising 上。**模型只需要在每个 timestep 上预测噪声 ε,这是个回归任务,Transformer 完全胜任**。

更深层的动机是 scaling:Transformer 在 LLM 上有 Kaplan / Chinchilla 这样的干净幂律 (loss vs N/D/C 都是 log-log 直线);CNN-based U-Net 在 diffusion 上没有类似的清晰规律。如果能把 diffusion 骨架换成 Transformer,**就可以套用 LLM 那套"按 FLOPs 预算选模型"的工程方法论**——这是 DiT 最有商业价值的发现。

实证结果非常干净:DiT-S / B / L / XL 四个 size 在 ImageNet 256 上 FID 单调下降,scaling 几乎完美线性。这是 diffusion 第一次有了**可外推的 scaling 实证**,直接铺平了 2024 年 Sora 那种"千亿参数视频 diffusion"的可行性。

### 机制一:Patchify Latents — 把 latent 当 token 序列

DiT 站在 [LDM](02-ldm.md) 的肩膀上,所有操作都在 VAE latent 空间(`32×32×4` 对应 256×256 输入图)而不是 pixel 空间——这点和 LDM 完全一致。差别是 LDM 在 latent 上用 U-Net,DiT 在 latent 上用 Transformer。

Transformer 要求输入是 token 序列,所以 DiT 第一步是 **patchify**——和 [ViT](../08-vit/01-vit.md) 完全同款思路:

- 取 latent `z ∈ R^{32×32×4}`
- 用 patch size `p` 切块,变成 `(32/p)² × (p²·4)` 的 patch 序列
- 每个 patch 过一个 linear projection 变成 `d` 维 token
- 加上 2D sin/cos 位置编码,得到 `T × d` 的 token 序列(`T = (32/p)²`)

`p` 是 DiT 的关键超参 —— `p=8` 给 4×4=16 个 token,`p=4` 给 8×8=64,`p=2` 给 16×16=256。**patch 越小 token 越多,Transformer 计算量随 token 数平方增长**,但生成质量也更好。DiT 论文最常用 `p=2`,因为 latent 已经被 VAE 压缩过 8× 了,再切大 patch 会丢细节。

输出端做反向操作:Transformer 输出 `T × d` 序列 → 投影回 `(32/p)² × (p²·4)` → reshape 回 `32×32×4`。**整个网络从输入到输出都是纯 Transformer + 两个 linear,没有任何卷积**。

### 机制二:adaLN-Zero — 把 timestep / class 条件注入到每个 block

Diffusion denoising 网络需要知道两件事:**当前 timestep `t`(决定噪声强度)和 class label `c`(条件生成的目标类)**。U-Net 里这两个信号通常加到通道维或通过 cross-attention 注入,DiT 用的是更优雅的方式——**adaLN-Zero(adaptive LayerNorm with zero init)**。

具体做法:把 `t` 和 `c` 编码成两个 embedding 向量,相加得到 `c_total`。然后在每个 Transformer block 里:

1. 用一个小 MLP 把 `c_total` 映射成 6 个参数 `(γ_1, β_1, α_1, γ_2, β_2, α_2)`
2. 把标准 LayerNorm `LN(x) = (x - μ) / σ` 替换成 `adaLN(x) = γ · LN(x) + β`
3. 在 attention 输出和 FFN 输出处分别乘 `α`(scale residual)

也就是说 block 变成:
```
x = x + α_1 · attn(adaLN(x, γ_1, β_1))
x = x + α_2 · ffn(adaLN(x, γ_2, β_2))
```

**"Zero"指的是 MLP 最后一层初始化为 0**,这样 `γ ≈ 1, β ≈ 0, α ≈ 0` ——初始 forward 上 adaLN 退化成恒等映射,残差路径不被破坏。训练时 MLP 逐步学到合适的调节量。这一 trick 让 DiT 训练**从一开始就稳定**,不会因为条件注入扰动初始的 Transformer 流。

adaLN-Zero 这一思想后来被广泛复用:**SD3 的 MM-DiT**、**Sora 的视频 DiT**、**PixArt-α**、**FLUX** 全部用 adaLN-Zero(或其变体)注入条件。它的优势是参数高效(每 block 多 ~6 个标量 × MLP 一层)+ 表达力强(LN 缩放 + 残差缩放双重控制)+ 训练稳定(zero init)。

### 机制三:Scaling — DiT-S / B / L / XL 干净的幂律

DiT 论文最有价值的实证不是 SOTA 数字,而是**第一次给 diffusion 提供了像 LLM 那样干净的 scaling curve**。Peebles 训了四个 size:

| 模型 | 层数 | d_model | num_heads | 参数量 | Gflops (p=2, 256²) |
|------|------|------|------|------|------|
| DiT-S | 12 | 384 | 6 | 33M | 6 |
| DiT-B | 12 | 768 | 12 | 130M | 23 |
| DiT-L | 24 | 1024 | 16 | 458M | 80 |
| DiT-XL | 28 | 1152 | 16 | 675M | 119 |

在 ImageNet 256×256 class-conditional 上,四个 size 在同样 400K iter 训练后的 FID 单调下降——而且**画在 log(Gflops) - FID 平面上几乎是一条直线**。这是 diffusion 模型第一次显示这种"按算力预算可外推"的性质。

更进一步,DiT-XL/2 训练到 7M iter 拿到 **FID 2.27**(ImageNet 256 class-conditional),打破了 ADM 之前的 SOTA 3.94。但论文真正强调的不是这个数,而是 **"如果继续 scale up,FID 还会继续降"的清晰外推性**——这一信号直接告诉行业:**diffusion 不会卡在 1B,可以推到 10B / 100B,只要算力跟上**。

后续 Sora(估计参数量 3-10B)、SD3-Large(8B)、FLUX-12B 都是沿着这条 scaling 曲线往上爬的产物。如果没有 DiT 这条干净幂律,各家公司无法说服资本投入百卡万卡训百亿级 diffusion 模型。

### 三件套协同:patchify + adaLN-Zero + scaling 缺一不可

DiT 真正改变行业,**不是任意一件单独成立**,而是这三件事同时被装进一个系统——任何一个抽掉整套范式都不成立,这一点和 [ResNet](../01-cnn/05-resnet.md) `shortcut + BN + He 初始化` 协同关系一致:

- **只有 patchify,没有 adaLN-Zero** —— 条件注入要么用 cross-attention(参数翻倍且训练不稳),要么用通道相加(表达力弱)。无法在每个 block 上灵活调节 t 和 c,生成质量差 [ADM](../10-diffusion/01-ddpm.md) 一截
- **只有 adaLN-Zero,没有 patchify** —— 输入仍是 latent 空间的 H×W×C 张量,Transformer 处理 image-as-spatial 需要展平 → 等价于 patch_size=1,token 数 1024,计算量爆炸,完全无法 scale 到 XL
- **只有 patchify + adaLN-Zero,没有 scaling 验证** —— Peebles 早期实验只有 DiT-S/B 时 FID 还不如 ADM。**必须做到 DiT-XL 才能展示纯 Transformer 路径的潜力**,且给出 log-log scaling curve 才能说服行业"这不是一次性 hack 而是可扩展骨架"

更深层的协同:**Transformer 把视觉生成纳入了"和 LLM 同一个工程栈"**。Patchify 是 ViT 的接口,adaLN-Zero 是 GANs/Style 注入的延续,scaling 是 LLM 时代的方法论——三者拼起来,diffusion 第一次拥有了和 LLM 平行的"按 FLOPs 选模型 / 多模态共享骨架"的能力。这是为什么 2024 年所有前沿生成模型(Sora / SD3 / FLUX / PixArt-α / Hunyuan-DiT)几乎全是 DiT 系列。

## DiT vs U-Net 对比

DiT 论文的关键对照实验:在同样 FLOPs / 训练步数下,DiT vs U-Net 谁好?

| 模型 | 参数量 | Gflops | FID-50K(ImageNet 256, cfg) |
|------|------|------|------|
| ADM(U-Net,SOTA) | 554M | 119 | 3.94 |
| LDM(U-Net + latent) | 400M | 104 | 3.60 |
| **DiT-XL/2** | **675M** | **119** | **2.27** |

观察:**DiT 在大致同等 FLOPs 下,FID 比 U-Net 路线低 30-40%**。更重要的是,DiT-XL/2 是论文中最大的模型——继续 scale 还有改善空间,而 U-Net 路线在 ADM 之后基本卡住。

为什么 Transformer 更适合 diffusion?几个推测:

1. **没有归纳偏置就是优势**——CNN 的 local + multi-scale 偏置在数据量小时是优势,但在 ImageNet 这种大数据集上反而限制表达力。Transformer 让网络从数据里自己学习哪些 patch 该互相 attend
2. **adaLN-Zero 比 group normalization + skip connection 灵活**——U-Net 的 timestep embedding 通常加到通道上或在 skip 处通过 FiLM 调节,adaLN-Zero 提供更细粒度的逐层 γ / β / α 控制
3. **scaling 更可预测**——LLM 工程团队已经掌握了 "Transformer + 数据 + 算力 → loss 单调下降" 的范式,可以直接套用到 DiT 上,不需要重新摸索

DiT 也有几个局限被后续工作攻克:

- **训练慢**——DiT-XL 要 7M iter 才完全收敛,比 U-Net SOTA 慢 3-5×。后续 SD3 / FLUX 用 Flow Matching + 改进的 schedule 加速
- **推理 token 数固定**——`p=2` 给 256 token,生成更高分辨率(512 / 1024)时 token 数平方增长,推理时延增大。SD3 用 MM-DiT 双流减轻,Sora 用稀疏注意力压成本
- **不直接支持文本条件**——DiT 原文只做 class-conditional,文生图要再加 cross-attention 把 CLIP/T5 文本 embedding 注入(SD3 / PixArt-α 都这么做)

## DiT 的衍生

DiT 在 2023-2024 年衍生出一系列重要工作,改写了整个生成式 AI 领域:

**PixArt-α**(Huawei, 2023 09)——把 DiT 推到文生图,用 T5 文本 embedding 通过 cross-attention 注入,FID 接近 SDXL 但训练成本仅 10%

**Stable Diffusion 3 (MM-DiT)**(Stability, 2024 03)——MultiModal DiT,文本和图像 token 在同一个 Transformer 里互相 attend(不是 cross-attention),双流共同更新

**FLUX**(Black Forest Labs, 2024 08)——SD3 团队出走后做的开源 DiT,12B 参数,在文生图质量上压过 Midjourney v6,标志开源 diffusion 进入 10B+ 时代

**Sora**(OpenAI, 2024 02)——把 DiT 推到视频生成,时空 patchify 把视频切成 4D patches,展示了 "scalable diffusion + 大量视频数据 = 涌现物理理解" 的可能性

**Hunyuan-DiT / OpenSora / CogVideoX** —— 各家中国公司的 DiT 衍生,多语言文生图 / 长视频生成

vanilla DiT 仍然是研究 baseline,但生产部署几乎全用 SD3-style MM-DiT 或 FLUX-style 改良版。

## 训练细节

| 维度 | DiT-XL/2 配置 |
|------|------|
| Backbone | 28 层 Transformer, d_model=1152, h=16, d_ff=4608 |
| Patch size | 2 (latent 32×32 → 16×16 = 256 token) |
| Latent | VAE encoder (8× downsample), `32 × 32 × 4` for 256² input |
| 条件注入 | adaLN-Zero(timestep + class label embedding 共 1152 维) |
| 优化器 | AdamW, lr=1e-4(不需要 warmup,因为 adaLN-Zero 自动稳定) |
| Batch size | 256(全局) |
| Diffusion schedule | 1000 steps, linear β schedule(沿用 DDPM) |
| EMA | 0.9999 |
| 训练步数 | 7M iter for SOTA,400K iter for scaling 实验 |
| 训练硬件 | 8 × A100, 几周 |
| ImageNet 256 FID | **2.27**(cfg = 1.5) |

注意 DiT 训练**不需要 learning rate warmup**——这是 adaLN-Zero 的直接收益。U-Net 时代的 ADM / LDM 都要精细 warmup 防止初期梯度爆炸,DiT 因为初始时 adaLN 是恒等映射,从第一步就稳定。

## 关键代码

```python
import torch
import torch.nn as nn

class DiTBlock(nn.Module):
    """adaLN-Zero conditioned Transformer block"""
    def __init__(self, d_model, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )
        # adaLN-Zero: 把 condition 映射成 6 个调节参数
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )
        # Zero init 最后一层,保证初始 forward 是恒等
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        # c: condition embedding (timestep + class), [B, d_model]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        # attention with adaLN
        h = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h, _ = self.attn(h, h, h)
        x = x + gate_msa.unsqueeze(1) * h
        # FFN with adaLN
        h = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h
        return x


class DiT(nn.Module):
    def __init__(self, input_size=32, patch_size=2, in_channels=4,
                 d_model=1152, depth=28, num_heads=16, num_classes=1000):
        super().__init__()
        num_patches = (input_size // patch_size) ** 2
        # patchify: latent → token sequence
        self.patch_embed = nn.Conv2d(in_channels, d_model,
                                     kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        # timestep + class embedders
        self.t_embed = TimestepEmbedder(d_model)       # sinusoidal + MLP
        self.y_embed = nn.Embedding(num_classes, d_model)
        # transformer
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, num_heads) for _ in range(depth)
        ])
        # final norm + unpatchify
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_linear = nn.Linear(d_model, patch_size * patch_size * in_channels)
        self.patch_size = patch_size

    def forward(self, x, t, y):
        # x: [B, 4, 32, 32] latent
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, T, d]
        x = x + self.pos_embed
        c = self.t_embed(t) + self.y_embed(y)               # [B, d]
        for block in self.blocks:
            x = block(x, c)
        x = self.final_norm(x)
        x = self.final_linear(x)
        # unpatchify back to latent shape [B, 4, 32, 32]
        return unpatchify(x, self.patch_size, in_channels=4)
```

整套实现 ~250 行,比 U-Net 那种 encoder + skip + decoder 的代码简单很多。这种工程简洁性也是 DiT 被广泛采用的原因。

## 影响 / 后续

DiT 在 diffusion 历史上的位置:**让生成式视觉进入 Transformer 时代**。具体几条:

**1. Sora 直接选 DiT 做骨架**(2024 02)——OpenAI 视频生成模型公开技术报告里明确写"Diffusion Transformer (DiT)"是骨干,把图像 DiT 扩展到视频时空 patches。**没有 DiT,Sora 这种规模的视频 diffusion 找不到可扩展骨架**

**2. SD3 / FLUX 标志开源文生图进入 DiT 时代**(2024)——Stability AI 弃用 SDXL 的 U-Net,SD3 用 MM-DiT;前 SD3 团队的 Black Forest 出 FLUX 12B 也是 DiT 系。**2024 年下半年所有开源前沿文生图模型几乎都是 DiT**

**3. 多模态生成统一骨架**——文本 (LLM) / 图像 (DiT) / 视频 (Sora) / 3D / 音频 都可以用 Transformer + diffusion 组合,工程栈共享,优化经验跨模态迁移

**4. Scaling 方法论落地**——LLM 时代的 "按 FLOPs 选模型"工程范式被搬到 diffusion,各家公司可以按算力预算定 DiT-S/B/L/XL 选型,不用再手工调 U-Net 通道

**5. 视频 diffusion 起飞**——CogVideoX / Stable Video Diffusion / Mochi / Hunyuan-Video 全部 DiT 系骨架,2024 视频生成质量飞跃直接来自 DiT 的可扩展性

DiT 留下的几个推动后续工作的尾巴:

- **Token 数 vs 分辨率的矛盾**——512 / 1024 / 2048 分辨率下 token 数平方爆炸 → Sparse Attention / Window Attention / DiT-MoE 等方向
- **文本条件还要 cross-attention**——MM-DiT 把 cross-attention 改成 joint self-attention,后续会不会进一步统一?
- **训练效率**——DiT 比 U-Net 慢,Flow Matching / Rectified Flow 等是加速方向
- **采样步数**——sampling 仍要几十到几百步,蒸馏 (LCM / TCD / SDXL-Turbo) 是 1-step 方向

→ [02-ldm.md](02-ldm.md) · 父结构,DiT 是 LDM 把 U-Net 换成 Transformer 的版本
→ [01-ddpm.md](01-ddpm.md) · 祖父结构,diffusion 起点
→ [04-flow-matching.md](04-flow-matching.md) · 训练 schedule 改进,与 DiT 正交
→ [../08-vit/01-vit.md](../08-vit/01-vit.md) · patchify 思路的直接来源
→ [../05-transformer/01-transformer.md](../05-transformer/01-transformer.md) · backbone 来源
