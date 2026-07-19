# 世界模型 / 视频生成

> **让模型学会预测"接下来会发生什么"——从在自己想象的世界里训练强化学习智能体,到生成分钟级连贯视频,再到把 diffusion 模型直接变成可交互的实时游戏引擎。**

## 一句话定位

这家族解决的是一个和"理解世界"直接相关的问题——**能不能让模型学出一个关于世界如何运作的内部模型,然后用这个模型做预测、做决策、甚至生成可交互的环境**? 2018 年 Ha & Schmidhuber 的 **World Models** 首次证明:一个强化学习智能体可以完全在自己学到的"梦境"(RNN 生成的想象轨迹)里训练策略,再迁移回真实环境。这条"学一个世界模型来做决策"的思路在 2023 年被 **DreamerV3** 规模化到 150+ 个跨领域任务、固定超参数不调参就能打平各领域的 model-free SOTA。与此同时,另一条独立发展的脉络是**用生成模型直接产出视频画面本身**——2022 年 **Video Diffusion Models** 把 DDPM 从图像推广到视频,2024 年 **Sora** 把 DiT 规模化到分钟级连贯视频,论文明确提出"video generation models as world simulators"的定位。这两条脉络在 2024 年汇合:**Genie** 证明可以从无标注的互联网视频里无监督学出逐帧可控制的生成式环境,**GameNGen** 则证明 diffusion 模型可以完全替代传统游戏引擎的渲染循环,实时生成可玩的 DOOM。这家族要回答的问题是:**从"在想象里训练策略"到"生成可交互的世界本身",世界模型这一支是怎么和视频生成技术合流的**。

## 概念本身

"世界模型"(world model)这个术语在这个家族里有两层含义,分别对应两条历史脉络:

### 脉络一:World Model 作为 RL 的决策工具

Ha & Schmidhuber 2018 的原始定义——世界模型是一个**学出来的、关于环境动态的内部模拟器**,通常拆成三部分:
- **V(Vision)**:把高维观测(像素)压缩成低维表征
- **M(Memory)**:在压缩表征空间里预测下一步会发生什么(建模时序动态)
- **C(Controller)**:基于 V/M 给出的表征做决策,通常刻意做得很小

这条脉络的核心价值是**样本效率**——真实环境交互往往昂贵(机器人、游戏引擎渲染耗时),如果智能体可以在学到的世界模型内部"想象"出大量虚拟经验来训练,就能大幅减少真实环境交互次数。DreamerV3(2023)是这条脉络目前的巅峰:用同一套固定超参数的 latent imagination 方法,横跨 Atari / DeepMind Control / Minecraft 等 150+ 任务。

### 脉络二:World Model 作为视频生成的目标

Sora 技术报告重新定义了"世界模型"——不是给 RL 智能体用的内部表征,而是**直接能生成逼真、时空一致的视频画面本身**的生成模型。这条脉络认为:如果一个模型能生成足够逼真且物理一致的视频(比如物体不会突然消失、光影变化符合物理规律),就说明它隐式学到了关于世界如何运作的知识。这条脉络的技术演化路径是:图像 diffusion(DDPM)→ 视频 diffusion(Video Diffusion Models)→ 规模化到分钟级(Sora)→ 加上可控制的交互性(Genie、GameNGen)。

### 两条脉络的合流

2024 年的 Genie 和 GameNGen 是两条脉络汇合的产物:它们既是"生成视频"的模型(继承脉络二的技术,diffusion / transformer 架构),又是"可交互的环境模拟器"(继承脉络一的目标,能响应动作输入、支持类似 RL 的交互循环)。

## 子时间线

| 年份 | 名字 | 关键贡献 | 之前卡在哪 |
|------|------|---------|-----------|
| 2018 | **World Models** | Ha & Schmidhuber——VAE(V)学视觉压缩 + MDN-RNN(M)学时序动态 + 极小线性 Controller(C)在 M 生成的"梦境"里用 CMA-ES 训练,首次证明智能体可以完全脱离真实环境、在自己学到的世界模型内部完成策略训练 | RL 智能体要么直接在原始像素上做 model-free RL(样本效率低),要么世界模型和策略网络耦合训练、难以独立评估 |
| 2022 | **Video Diffusion Models** | Ho et al.——把 DDPM 的去噪框架从图像推广到视频:时空分解卷积(2D 空间卷积 + 1D 时间卷积)替代昂贵的全 3D 卷积,图像/视频联合训练复用大规模图像数据 | 直接把图像 DDPM 的 2D U-Net 扩展成 3D 会让计算量爆炸式增长,视频数据集规模又远小于图像数据集 |
| 2023 | **DreamerV3** | Hafner et al.——把 latent imagination 式 model-based RL 规模化到跨领域通吃(Atari/DMC/Minecraft 等 150+ 任务),固定同一套超参数不调参就匹配甚至超过各领域 model-free SOTA | 此前 Dreamer 系列虽证明"想象里训练"可行,但每个领域往往需要针对性调超参数,跨领域泛化性差 |
| 2024 | **Sora** | OpenAI——把 DiT 规模化到分钟级、多分辨率、多时长连贯视频,用 spacetime patches 统一表示不同长宽比/时长的时空数据,技术报告提出"video generation models as world simulators" | Video Diffusion Models 证明了视频 diffusion 可行,但受限于固定分辨率/时长训练,U-Net 架构 scaling curve 不如 Transformer 干净 |
| 2024 | **Genie** | DeepMind——无监督地从海量无标注互联网视频里学出逐帧可控制的生成式环境,用 Latent Action Model 隐式推断出离散动作空间,不需要任何人工动作标注 | Sora 能生成逼真视频但只能被动播放,不能实时响应用户操作;DreamerV3 依赖已有明确动作空间的 RL 环境,不能直接用在无标注视频上 |
| 2024 | **GameNGen** | Google——用条件 diffusion 模型完全替代传统游戏引擎的渲染循环,实时交互式生成可玩的 DOOM 画面,证明神经网络可以端到端承担游戏引擎的职责 | Genie 证明了无监督可学出可控制环境,但画面质量/实时帧率不是其重点;传统游戏引擎则完全是手工编写的确定性程序 |

## 依赖与延伸

**前置(foundations):**
- [../05-transformer/01-transformer.md](../05-transformer/01-transformer.md) —— Sora 的 DiT 主干
- [../10-diffusion/01-ddpm.md](../10-diffusion/01-ddpm.md) —— Video Diffusion Models 直接扩展自 DDPM 的去噪框架
- [../10-diffusion/05-dit.md](../10-diffusion/05-dit.md) —— Sora 复用的 diffusion transformer 架构
- [../02-rnn-lstm/02-lstm.md](../02-rnn-lstm/02-lstm.md) —— World Models 的 MDN-RNN 前身

**延伸方向:**
- [../07-gpt-scaling/04-scaling-laws.md](../07-gpt-scaling/04-scaling-laws.md) —— scaling 叙事在 Sora 身上再次印证:视频生成质量随算力 / 数据规模提升的规律
