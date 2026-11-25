# 扩散模型

> **把生成问题转化为"从噪声逐步去噪"的反向过程,稳定训练、高质量、可控条件,2022 之后视觉生成的事实统治者。**

## 一句话定位

这家族解决的是 2014–2020 视觉生成里一个反复没解决好的问题——**怎么稳定地训练一个能从噪声生成高质量图像的模型**。这之前主流是 [GAN](../04-gan/) 系(生成器和判别器博弈),效果惊人但训练极其不稳:模式崩溃、判别器突然赢、超参一动就崩。2020 年 6 月 Ho 等人的 DDPM 给出了完全不同的方案——**把生成问题反转为"从纯噪声开始,逐步去噪"的反向扩散过程**,训练目标变成"预测每一步加的噪声",优化是标准 MSE 回归,稳定得像监督学习。这条路在 2021–2022 年迅速发展:**LDM(Latent Diffusion)/ Stable Diffusion** 让 diffusion 在 latent 空间运行,推理成本降 64×,把文生图带到消费硬件;**Imagen / DALL-E 2** 加入大语言模型作文本编码器 + classifier-free guidance,把质量推到 SOTA;**Flow Matching / Rectified Flow** 简化训练目标,SD3 / Flux 用。2024 年 OpenAI 的 Sora 把 diffusion 推到视频生成,Transformer backbone([DiT](../08-vit/04-dit.md))彻底替代 U-Net。这家族要回答的问题是:**从 DDPM 概念证明到 SD3 / Sora 时代,扩散模型这条路是怎么走出来的**。

## 概念本身

扩散模型的核心是**两个反向过程**:

**正向过程(forward diffusion)** —— 给一张图 `x_0` 逐步加噪声,经过 T 步(典型 1000)变成纯高斯噪声 `x_T ~ N(0, I)`。每一步加的噪声满足:

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \beta_t I)
$$

`β_t` 是预设的"加噪强度"(noise schedule),逐步增加。整个正向过程是固定的、无参数的,**不需要训练**。

**反向过程(reverse diffusion)** —— 学一个网络 `ε_θ(x_t, t)` **预测在 `x_t` 上加的噪声**。训练目标极简:

$$
\mathcal{L}_\text{simple} = \mathbb{E}_{t, x_0, \epsilon}\big[\| \epsilon - \epsilon_\theta(x_t, t)\|^2\big]
$$

也就是一个标准 MSE 回归——给定 noisy 图和 timestep,预测原始噪声。这一目标的稳定性是 GAN 系完全做不到的。

**采样(sampling)** —— 从纯噪声 `x_T` 出发,用 `ε_θ` 逐步去噪到 `x_0`。原版 DDPM 要 1000 步,DDIM(2020 Song)简化到 50 步,Consistency Model(2023 Song)进一步到 1-4 步。

这家族围绕几条主线演化:

- **效率**:把 1000 步推理压成 50/20/4/1 步;在 latent 而非 pixel space 工作
- **条件控制**:从无条件 → 类条件 → 文本条件 → 多模态条件
- **训练目标**:从 ε-prediction → v-prediction → score matching → flow matching
- **骨干网络**:U-Net(2020) → ViT-style DiT(2022) → MM-DiT(SD3)
- **应用模态**:从 2D 图像扩展到视频、3D、音频、蛋白质结构

理解 diffusion 不只是为了用 Stable Diffusion 画图——**它是从概率角度看生成问题的根本框架**,和 GAN(博弈)、VAE(变分推断)、autoregressive(逐元素)并列的第四条生成范式,且在 2022 年之后基本一统视觉生成。

## 子时间线

| 年份 | 名字 | 关键贡献 | 之前卡在哪 |
|------|------|---------|-----------|
| 2020 | **DDPM** | 把 2015 年 Sohl-Dickstein 提出的 diffusion 思想工程化:U-Net 预测噪声 + 简单 MSE 损失 + 1000 步采样,在 CIFAR-10 / LSUN 上质量超过 GAN | GAN 训练不稳模式崩溃;VAE 生成模糊;autoregressive 慢 |
| 2022 | **LDM / Stable Diffusion** | 把 diffusion 从 pixel 移到 latent space(VAE 编码),推理显存 64× 降低,把文生图带到消费 GPU | 像素级 diffusion 在 512² 分辨率上算力爆炸;只有大公司能训能跑 |
| 2022 | **Imagen / Classifier-Free Guidance** | 大文本编码器(T5-XXL) + CFG 让模型可在"忠实 vs 创造性"间精确调节;FID 2022 SOTA | DALL-E 系列文本理解弱;无 guidance 时模型不严格按 prompt 生成 |
| 2023 | **Flow Matching / Rectified Flow** | 把 diffusion 的 ε-prediction 推广到任意流形的"速度场学习",训练稳定 + 采样路径更直,SD3 / Flux 默认 | DDPM/score matching 的随机微分方程数学复杂,采样路径弯曲 |

## 依赖与延伸

**前置(foundations):**
- [../08-vit/04-dit.md](../08-vit/04-dit.md) —— Transformer backbone 替代 U-Net,SD3/Sora 的基座
- [../05-transformer/01-transformer.md](../05-transformer/01-transformer.md) —— DiT 和 text encoder 都基于 Transformer
- [../09-multimodal-clip/](../09-multimodal-clip/) —— 文本-图像对齐的基础,Stable Diffusion 用 CLIP text encoder
- `../foundations/03-optimizers/` —— Adam / EMA 等优化器细节

**通向哪些家族:**
- [../14-rag-agent/](../14-rag-agent/) —— 视觉 agent 用 diffusion 做图像生成 / 编辑
- [../04-gan/](../04-gan/) —— diffusion 取代 GAN 成生成主流;但 GAN 在某些场景(超分、风格迁移)仍有应用
- [../09-multimodal-clip/](../09-multimodal-clip/) —— 跨模态条件(图像 + 文本 → 图像生成)
