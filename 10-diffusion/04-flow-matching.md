---
name: "Flow Matching / Rectified Flow"
year: 2023
family: "10-diffusion"
order: 4
paper: "Flow Matching for Generative Modeling / Rectified Flow"
authors: ["Yaron Lipman", "Ricky T. Q. Chen", "Heli Ben-Hamu", "Maximilian Nickel", "Xingchao Liu", "Chengyue Gong", "Qiang Liu"]
key_idea: "把 diffusion 的 ε-prediction 推广到任意流形的'速度场学习',训练更稳 + 采样路径更直 + 数学更简洁,SD3 / Flux 默认"
---

## 前作进展

到 2022 年底,[DDPM](01-ddpm.md) 系 diffusion 模型已经主导视觉生成,但数学上仍有几个让人不满的细节:

**1. 训练目标和采样过程脱节** —— DDPM 训练目标是"预测噪声 ε",采样是按随机微分方程(SDE)反向积分。两者数学连接复杂,需要从变分下界(VLB)绕半天才能推到 ε-prediction loss

**2. 采样路径弯曲** —— SDE 反向过程的路径在 noise → data 之间不是直线,需要多步(50-100)才能采样高质量结果。Consistency Model(2023)和 Progressive Distillation 都试图缩短,但都是"事后修补"

**3. Noise schedule 是超参** —— β_t 的选择(linear / cosine / sigmoid)很影响最终质量,但没有原则性的选择标准,需要经验调

社区 2022-2023 年开始追问:**有没有更干净的数学框架,让训练和采样自然统一,且不依赖具体的 noise schedule?**

两条独立但本质相通的工作给出了答案:

**Flow Matching**(Lipman et al., Meta AI, 2023 年 2 月)—— 从 continuous normalizing flow(CNF)出发,提出"直接学速度场 `v(x, t)`",训练目标是 `v_pred ≈ (x_1 - x_0)`(在 noise 和 data 之间的直线方向)。完全跳过 noise schedule,数学最干净

**Rectified Flow**(Liu et al., UT Austin, 2022 年 9 月发表)—— 几乎同时提出"用直线连接 noise 和 data,学预测速度场"。和 Flow Matching 数学上是同一回事,但术语和推导路径不同

两条工作合并后被统称为 **Flow Matching**,在 2024 年成为主流文生图模型的训练目标:

- **Stable Diffusion 3**(2024 2 月)—— Flow Matching + MM-DiT
- **Flux**(Black Forest Labs, 2024 8 月)—— Rectified Flow + MM-DiT
- **AuraFlow**(2024)—— 开源 Flow Matching 实现

这一节聚焦 Flow Matching 的核心数学思想 + 工程意义,因为它代表了 diffusion 的下一代训练目标——更简洁、更稳定、未来主流。

## 核心思想:学速度场,不学噪声

Flow Matching 的核心想法:**给定数据点 `x_0`(真实图像)和噪声 `x_1`(纯高斯)之间的一条路径,学一个网络预测路径上每一点的速度方向**。

具体:定义 `x_t = (1 - t) · x_0 + t · x_1`,`t ∈ [0, 1]`——这是从 `x_0` 到 `x_1` 的**直线插值**。在这条直线上,任意点的**速度**(对 t 求导)就是:

$$
v(x_t, t) = \frac{dx_t}{dt} = x_1 - x_0
$$

Flow Matching 训练目标:

$$
\boxed{\mathcal{L}_\text{FM} = \mathbb{E}_{t \sim U[0,1], x_0, x_1}\big[\| v_\theta(x_t, t) - (x_1 - x_0) \|^2\big]}
$$

也就是说,**给定一个插值点 `x_t` 和时间 `t`,让网络预测"从 `x_0` 到 `x_1` 的方向"**。

对比 [DDPM](01-ddpm.md):

| | DDPM | Flow Matching |
|------|------|------|
| 网络预测 | 噪声 `ε` | 速度 `v = x_1 - x_0` |
| 训练目标 | `‖ε - ε_θ(x_t, t)‖²` | `‖v - v_θ(x_t, t)‖²` |
| `x_t` 构造 | `√ā_t · x_0 + √(1-ā_t) · ε`(曲线) | `(1-t) · x_0 + t · x_1`(直线) |
| Noise schedule | 需要(β_t / cosine 等) | **不需要**(t 均匀采) |
| 采样路径 | SDE 反向(随机) | ODE 反向(确定) |

```mermaid
graph LR
    x0["x_0<br/>(真实图)"]:::input --> path["直线插值<br/>x_t = (1-t)·x_0 + t·x_1"]:::compute
    x1["x_1<br/>(高斯噪声)"]:::input --> path
    path --> v["速度 v = x_1 - x_0<br/>(整条直线方向)"]:::compute
    v --> net["v_θ(x_t, t)<br/>预测速度"]:::compute
    net --> loss["MSE Loss"]:::output

    classDef input fill:#fef3c7,stroke:#d97706,color:#92400e;
    classDef compute fill:#fce7f3,stroke:#db2777,color:#9d174d;
    classDef output fill:#ecfdf5,stroke:#059669,color:#065f46;
```

*图 1:Flow Matching 训练 — 在 noise x_1 和 data x_0 之间直线插值,任意点 x_t 的真实速度是 (x_1 - x_0),让网络学预测它。*

## 采样:ODE 反向积分

采样阶段更优雅——把"预测速度"翻成"沿速度方向走"。从 `t=1`(纯噪声 `x_1`)出发,反向积分 ODE:

$$
\frac{dx}{dt} = v_\theta(x, t), \quad x(1) = x_1
$$

具体步数 N(典型 50,但 Flow Matching 在 10-20 步效果就很好):

```python
def sample(v_net, shape, N=50):
    x = torch.randn(shape)  # x at t=1
    dt = 1.0 / N
    for i in range(N):
        t = 1.0 - i * dt
        v_pred = v_net(x, t)
        x = x - dt * v_pred  # ODE Euler step: x_{t-dt} = x_t - dt · v
    return x  # 这就是生成的 x_0
```

**没有随机性**——SDE 反向有 noise injection,ODE 反向是确定的。这让 Flow Matching 采样比 DDPM 更可重复,也让"少步采样"质量更稳定。

## 为什么"直线"比"曲线"好

Flow Matching 在 noise 和 data 之间用**直线插值**,而 DDPM 的 noise schedule 决定 `x_t` 的路径是**曲线**(`x_t` 的方向不是直接指向 `x_0`,而是随 schedule 变化)。

**直线的工程好处**:

**1. 速度场恒定方向** —— 整条直线上速度都是 `x_1 - x_0`,网络只要学"这条直线的方向"——任务比"在曲线上每点的瞬时切线方向"简单得多

**2. 少步采样质量更好** —— 直线允许大步长 ODE 积分,Flow Matching 在 10-20 步采样质量已经接近 50 步;DDPM 在 10 步质量明显下降

**3. 多步可以 "rectify"** —— Rectified Flow 的核心 trick:用训好的 Flow 模型生成 (x_0, x_1) 对,然后再训一次,路径会更"直"——多次 rectify 可以推到 1 步生成

**4. 数学更简洁** —— 没有 β_t 调参,没有 VLB 推导,损失就是简单 MSE。整个理论可以在几页纸内说清楚,比 DDPM 的论文短

直观对比(在 2D toy 数据上):

```
DDPM noise schedule(曲线路径):
    x_T  →  弯弯曲曲  →  x_0
    (路径长且 wiggly,需要小步走)

Flow Matching(直线路径):
    x_1  →  直线  →  x_0
    (大步走就够)
```

## Stable Diffusion 3 的采用

SD3(2024 年 2 月)是 Flow Matching 在工业上首个 SOTA 应用。SD3 的几个关键设计:

**1. Rectified Flow 训练** —— 完全替代 DDPM 的 ε-prediction

**2. MM-DiT(Multi-Modal DiT)backbone** —— 把 [DiT](../08-vit/04-dit.md) 扩展到多模态:图像 patch 和文本 token 共享同一序列做 self-attention,而不是 cross-attention。Text 和 image 在同一 attention 内完全融合,文本理解和图像生成耦合更紧

**3. T5-XXL + CLIP 双文本编码器** —— 沿用 [Imagen](03-imagen.md) 的发现,大文本编码器是关键

**4. Logit-normal noise schedule** —— 不是均匀采 `t ∈ [0, 1]`,而是用 logit-normal 分布偏好中间 t(更难的时间点采样更多次,弥补它们的难度)

SD3 用 Flow Matching 在 800M-8B 参数规模下都展现出**比 DDPM 训练更稳、收敛更快、采样步数更少**的实证。这一结果让 2024 年的几乎所有新文生图模型(Flux, AuraFlow, HunyuanDiT v1.2, Pixart-Σ)都跟进 Flow Matching。

## 训练细节

| 维度 | Stable Diffusion 3 Medium(参考实现) |
|------|------|
| Backbone | MM-DiT, 2B 参数 |
| 文本编码器 | T5-XXL + CLIP-L + CLIP-G(三个并用) |
| 训练目标 | Rectified Flow(等价 Flow Matching) |
| Noise schedule | Logit-normal(t 偏好中间值) |
| 采样器 | Euler ODE(50 steps default,20 steps 也 work)|
| CFG | 用,scale 4-5(比 SD 1.5 的 7-10 低) |
| 训练数据 | LAION-5B + 内部高质量子集 |
| 训练硬件 | H100 集群 |
| 训练时间 | 数月 |

注意 **CFG scale 比 SD 1.5 低**——Flow Matching 训练的模型对条件控制更敏感,不需要高 cfg_scale 也能生成忠实于 prompt 的图。这是 Flow Matching 的实用优势之一。

## 关键代码

Flow Matching 的训练循环极其简短:

```python
import torch
import torch.nn.functional as F

class FlowMatching:
    def __init__(self, velocity_net):
        self.v_net = velocity_net  # 同 DDPM 的 U-Net 或 DiT,只是输出语义不同

    def training_step(self, x_0, condition=None):
        """Flow Matching 训练"""
        # 1. 采高斯噪声(对应 t=1 的点)
        x_1 = torch.randn_like(x_0)
        # 2. 均匀采 t(可以替换成 logit-normal 分布)
        t = torch.rand(x_0.size(0), device=x_0.device)
        # 3. 直线插值得到 x_t
        t_expand = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        # 4. 真实速度就是 x_1 - x_0
        v_target = x_1 - x_0
        # 5. 网络预测速度,MSE loss
        v_pred = self.v_net(x_t, t, condition=condition)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(self, shape, num_steps=50, condition=None, cfg_scale=4.0):
        """ODE Euler 反向采样"""
        x = torch.randn(shape)  # x at t=1
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = 1.0 - i * dt
            t_tensor = torch.full((shape[0],), t, device=x.device)
            # CFG: 同时算 cond 和 uncond 速度
            v_cond = self.v_net(x, t_tensor, condition=condition)
            v_uncond = self.v_net(x, t_tensor, condition=None)
            v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            # ODE Euler step
            x = x - dt * v_pred
        return x  # x at t=0,生成的图
```

对比 [DDPM](01-ddpm.md) 的训练循环,Flow Matching 少了:

- ❌ `alpha_bar` 系列预计算
- ❌ `add_noise` 的复杂公式(`√ā_t · x_0 + √(1-ā_t) · ε`)
- ❌ DDPM scheduler 的 step 函数

整个训练 10 行,采样 15 行——是历代最简洁的 diffusion 训练框架。

## 影响 / 后续

Flow Matching 在 diffusion 历史的位置:**diffusion 训练目标的现代化定型**。具体影响:

**1. SD3 / Flux / AuraFlow 全部采用** —— 2024 年开源 SOTA 文生图全转向 Flow Matching。DDPM 系训练目标在 2024-2025 逐步退出主流

**2. Consistency / few-step 模型的理论基础** —— Flow Matching 的"直线路径"思想直接催生了 Consistency Models / Consistency Distillation / Phased Consistency 等"少步采样"方法。LCM-LoRA(2023)让 SD 模型 1-4 步生成,核心是 Flow Matching 思想

**3. 跨模态生成的统一框架** —— Flow Matching 不只限于 2D 图像,扩展到了:
- **视频生成**(Sora 2024,推测用 Flow Matching 变体)
- **音频生成**(Stable Audio 2.0)
- **3D 生成**(InstantMesh)
- **蛋白质生成**(AlphaFlow / FoldFlow)

**4. 训练稳定性的进一步提升** —— Flow Matching 的简单 MSE 目标 + 无 noise schedule 让 hyperparameter 选择简化。社区报告训练 Flow Matching 模型比 DDPM 容易得多,新人入门门槛降低

**5. 数学社区的回应** —— Flow Matching 的简洁数学让 generative modeling 重新成为应用数学的研究热点。Optimal Transport / Schrödinger Bridge / Stochastic Interpolant 等数学工具开始被引入 diffusion 研究

**至此 10-diffusion 家族 4 节点完整**:[DDPM](01-ddpm.md)(概念证明)→ [LDM](02-ldm.md)(工程化)→ [Imagen + CFG](03-imagen.md)(文本控制)→ [Flow Matching](04-flow-matching.md)(训练目标现代化),覆盖 2020-2024 视觉生成完整演化主线。

→ [03-imagen.md](03-imagen.md) · 父方法,CFG 仍在 Flow Matching 里沿用
→ [02-ldm.md](02-ldm.md) · Flow Matching + latent space = SD3 / Flux
→ [01-ddpm.md](01-ddpm.md) · diffusion 范式起源,Flow Matching 是它的数学简化
→ [../08-vit/04-dit.md](../08-vit/04-dit.md) · SD3 用 MM-DiT + Flow Matching,两者合体定型现代文生图
→ [../09-multimodal-clip/](../09-multimodal-clip/) · SD3 的 CLIP-L/G 编码器贡献
