---
name: "CycleGAN"
year: 2017
family: "04-gan"
order: 3
paper: "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"
authors: ["Jun-Yan Zhu", "Taesung Park", "Phillip Isola", "Alexei A. Efros"]
key_idea: "用 cycle consistency loss 实现无配对图像翻译——两个 G 互相 mapping(X→Y 和 Y→X),要求 F(G(x)) ≈ x;不需要成对训练数据就能做马↔斑马、夏↔冬、照片↔画风的转换"
---

## 前作进展

2016-2017 年图像翻译(image-to-image translation)是 CV 热门方向。代表工作 **pix2pix**(Isola 2017,与 CycleGAN 同组)用 conditional GAN 学 X → Y 的映射:

```
输入:配对数据 {(x_i, y_i)},如(边缘图, 真实照片)
训练:G(x_i) ≈ y_i,D 判别 (x_i, G(x_i)) 是否真实
```

pix2pix 在配对数据上效果惊艳——边缘图变照片、夜景变白天、街景变标注图。但**真实世界很多翻译任务没有配对数据**:

- 马 → 斑马:不存在"同一动物同一姿态"的马和斑马照片
- 莫奈画风 → 真实照片:莫奈画的是想象场景,没有对应真实照片
- 夏季 → 冬季:同一地点同一角度的夏冬照片极难收集

如果只用单边 GAN(G: X → Y),会出现**模式塌缩**:G 学到任何 X 都映射到一个"最容易骗 D"的 Y,失去 X 的内容信息。

Zhu 等人(Berkeley + UCB + UC Davis,2017 年 3 月)的 CycleGAN 给出关键洞察:**用"循环一致性"约束 G**。如果 X → Y → X 后能恢复原 X,说明 G 保留了 X 的内容。

CycleGAN 发布后立即引爆社区:

- 马↔斑马、夏↔冬、橙↔苹果、莫奈↔照片等 demo 成 AI 圈热门 GIF
- "把我的自拍变成动漫"成大众想象
- 后续 StarGAN(多域)、MUNIT、UNIT 等延续 cycle consistency 路线

CycleGAN 不是 GAN 训练技术上的突破,而是**应用层的爆发**——它把 GAN 从"实验室生成 MNIST"推到"普通人能玩的艺术工具"。

## 核心思想:Cycle Consistency

### 双向 mapping + 一致性约束

CycleGAN 有 **两个 generator + 两个 discriminator**:

- $G: X \to Y$(把马变斑马)
- $F: Y \to X$(把斑马变回马)
- $D_Y$:判别 Y 域真假
- $D_X$:判别 X 域真假

### Cycle Consistency Loss

核心约束:**X → Y → X 应该恢复原 X**:

$$
\mathcal{L}_{\text{cyc}}(G, F) = \mathbb{E}_{x \sim X}[\|F(G(x)) - x\|_1] + \mathbb{E}_{y \sim Y}[\|G(F(y)) - y\|_1]
$$

直觉:如果 G 把马 x 变成斑马 G(x),F 把这个斑马变回去,应该还是原来的马 x。这一约束**强迫 G 保留 X 的内容信息**,而不只是生成任意一个合理的 Y。

### 完整 loss

$$
\mathcal{L}(G, F, D_X, D_Y) = \mathcal{L}_{\text{GAN}}(G, D_Y, X, Y) + \mathcal{L}_{\text{GAN}}(F, D_X, Y, X) + \lambda \mathcal{L}_{\text{cyc}}(G, F)
$$

- $\mathcal{L}_{\text{GAN}}$ —— 标准 GAN 对抗 loss(让 G/F 生成的样本被 D 判为真)
- $\mathcal{L}_{\text{cyc}}$ —— cycle consistency loss
- $\lambda$ —— 平衡系数,通常 10

### 训练

```
for each batch (x, y) (x 来自 X 域,y 来自 Y 域,无配对):
    # 1. 训练 G (X→Y) 和 F (Y→X)
    fake_y = G(x)
    cycle_x = F(fake_y)
    fake_x = F(y)
    cycle_y = G(fake_x)

    # cycle loss
    cyc_loss = ||cycle_x - x||₁ + ||cycle_y - y||₁
    # adversarial loss
    g_loss = BCE(D_Y(fake_y), 1) + BCE(D_X(fake_x), 1) + λ * cyc_loss
    更新 G, F

    # 2. 训练 D_X, D_Y
    更新 D_Y 区分 y 和 fake_y
    更新 D_X 区分 x 和 fake_x
```

### Identity Loss(可选,提升效果)

论文还加了一个 identity loss:

$$
\mathcal{L}_{\text{idt}}(G, F) = \mathbb{E}_{y \sim Y}[\|G(y) - y\|_1] + \mathbb{E}_{x \sim X}[\|F(x) - x\|_1]
$$

直觉:如果输入已经是目标域(把"斑马"输入到 G: 马→斑马),应该输出原样。这一约束保留颜色 / 纹理信息,防止 G 给所有马都加上斑马纹但同时改变背景色等无关属性。

### 架构

- **Generator**:U-Net-like(早期版本)或 ResNet-based(论文最终版,9 个 residual blocks)
- **Discriminator**:PatchGAN(70×70 局部判别,而不是整图判别),减少参数 + 加快训练

## 关键代码

简化版 CycleGAN 训练循环:

```python
import torch
import torch.nn as nn
import itertools

# G_xy: X → Y,F_yx: Y → X
G_xy = ResNetGenerator().cuda()
F_yx = ResNetGenerator().cuda()
D_x = PatchGAN().cuda()
D_y = PatchGAN().cuda()

opt_G = torch.optim.Adam(
    itertools.chain(G_xy.parameters(), F_yx.parameters()),
    lr=2e-4, betas=(0.5, 0.999)
)
opt_Dx = torch.optim.Adam(D_x.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_Dy = torch.optim.Adam(D_y.parameters(), lr=2e-4, betas=(0.5, 0.999))

mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()
lambda_cyc = 10.0
lambda_idt = 5.0

for epoch in range(200):
    for x, y in zip(loader_X, loader_Y):  # 无配对
        x, y = x.cuda(), y.cuda()

        # ===== 训 G =====
        opt_G.zero_grad()
        # adversarial
        fake_y = G_xy(x)
        fake_x = F_yx(y)
        adv_loss = mse_loss(D_y(fake_y), torch.ones_like(D_y(fake_y))) \
                 + mse_loss(D_x(fake_x), torch.ones_like(D_x(fake_x)))
        # cycle
        cyc_x = F_yx(fake_y)
        cyc_y = G_xy(fake_x)
        cyc_loss = l1_loss(cyc_x, x) + l1_loss(cyc_y, y)
        # identity
        idt_y = G_xy(y)  # y 已经是 Y 域,G_xy 应原样输出
        idt_x = F_yx(x)
        idt_loss = l1_loss(idt_y, y) + l1_loss(idt_x, x)

        g_loss = adv_loss + lambda_cyc * cyc_loss + lambda_idt * idt_loss
        g_loss.backward()
        opt_G.step()

        # ===== 训 D_Y =====
        opt_Dy.zero_grad()
        d_loss_y = mse_loss(D_y(y), torch.ones_like(D_y(y))) \
                 + mse_loss(D_y(fake_y.detach()), torch.zeros_like(D_y(fake_y)))
        d_loss_y *= 0.5
        d_loss_y.backward()
        opt_Dy.step()

        # ===== 训 D_X =====
        opt_Dx.zero_grad()
        d_loss_x = mse_loss(D_x(x), torch.ones_like(D_x(x))) \
                 + mse_loss(D_x(fake_x.detach()), torch.zeros_like(D_x(fake_x)))
        d_loss_x *= 0.5
        d_loss_x.backward()
        opt_Dx.step()
```

注意 CycleGAN 用 **LSGAN loss**(MSE 替代 BCE)而非原 GAN 的 BCE。这是 2017 年的 trick,让 GAN 训练更稳定。

## 性能数据

CycleGAN 主要是定性结果突出。论文里几个标志性任务:

### 1. 马 ↔ 斑马

ImageNet 上马和斑马各 ~1000 张图,无配对。CycleGAN 训完后:

- 给一张马的照片,输出几乎完美的斑马(保留姿态、背景、光照)
- 反向也 work

### 2. 风景照 ↔ 莫奈 / 梵高 / 塞尚画风

- 莫奈画(1074 张)+ Flickr 风景照(6287 张),无配对
- 训完后照片能转成各画家风格,保留构图但改变笔触 / 色调

### 3. 夏 ↔ 冬(优胜美地照片)

- 同一国家公园不同季节照片,无配对
- 把绿草夏景变成雪覆冬景,保留地形和构图

### 4. 苹果 ↔ 橙子、橘子 ↔ 桃子等

- 简单形状的水果在外观上互转
- 但**几何形状变化时失败** ——CycleGAN 改不了 shape,只能改 texture / color

### 定量评估(Cityscapes labels↔photos)

| Method | per-pixel acc | per-class acc | class IoU |
|------|------|------|------|
| pix2pix(配对监督) | 0.71 | 0.25 | 0.18 |
| CoGAN(无配对) | 0.40 | 0.10 | 0.06 |
| **CycleGAN**(无配对) | **0.52** | **0.17** | **0.11** |

CycleGAN 在无配对数据上接近 pix2pix(配对监督)的一半,远超之前的无配对方法。

### Human Perceptual Study(AMT 真假判别)

让 25 个 AMT 用户判断"哪个是真照片,哪个是 CycleGAN 生成":

- 在 maps→aerial 任务上,**26% 用户被骗**(随机猜 50%)
- 在街景→labels 任务上,**23% 被骗**

数字看起来不高,但考虑 2017 年的技术水平,这是震撼的——人类无法可靠分辨 CycleGAN 输出与真实图像。

## 影响 / 后续

CycleGAN 在 GAN 历史的位置:**GAN 应用层的爆发,把 GAN 从实验室带到大众视野**。

**1. 无配对学习成新范式** —— CycleGAN 之前 image translation 必须要配对数据,之后 unpaired translation 成主流。后续 StarGAN(2018,多域翻译)、UNIT、MUNIT(disentangled)、DRIT 等都基于 cycle consistency 思想

**2. 文化影响巨大** —— CycleGAN 的"马变斑马"是 AI 圈 2017-2018 年最病毒式 demo。被引用、被模仿、被改编无数次。CycleGAN 是少数让"普通人也理解 GAN 在做什么"的工作

**3. 启发后续艺术 GAN 工具** —— PaintsChainer、StyleTransfer App、Prisma 等大众艺术 AI 工具都受 CycleGAN 启发

**4. Cycle consistency 思想扩散到其他领域** —— 机器翻译里的 back-translation、语音转换、视频生成等都用 cycle consistency 增强训练

**5. PatchGAN 成 GAN 标配** —— CycleGAN 的 PatchGAN discriminator(局部判别)成为后续 GAN 默认设计(StyleGAN、Pix2PixHD 都用)

**6. Phillip Isola / Alexei Efros 等成视觉 GAN 代表人物** —— Isola 现任 MIT 教授,Efros 是 Berkeley 教授,两人在 CycleGAN 之后继续做 GAN / image synthesis,影响整个 CV 生成方向

CycleGAN 留下的开放问题:

- **几何形状变化失败** —— CycleGAN 只能改 texture,改不了形状(猫↔狗失败)→ U-GAT-IT、Multimodal CycleGAN 部分解决
- **训练慢** —— 4 个网络同时训,内存和算力消耗大 → Lite 版本如 CycleGAN-Turbo
- **质量上限** —— 256×256 以上质量下降明显 → Pix2PixHD、SPADE
- **同语义内容失真** —— 偶尔出现"马变斑马时丢掉腿"等内容失真 → Contrastive Unpaired Translation (CUT)
- **Diffusion 时代部分被取代** —— [Stable Diffusion + ControlNet](../10-diffusion/02-ldm.md) 后能做更精细的图像翻译,但 CycleGAN 在计算成本上仍有优势

→ [04-stylegan.md](04-stylegan.md) · GAN 质量巅峰,与 CycleGAN 并列 GAN 后期代表
→ [02-dcgan.md](02-dcgan.md) · 父技术,CycleGAN 的 G/D 基于 DCGAN-style CNN
→ [01-gan.md](01-gan.md) · 数学起源,CycleGAN 加 cycle 约束扩展 GAN 应用
→ [../10-diffusion/](../10-diffusion/) · Diffusion 时代部分取代 CycleGAN
→ [../09-multimodal-clip/](../09-multimodal-clip/) · CLIP + Diffusion 让 text-guided translation 成为现代主流
