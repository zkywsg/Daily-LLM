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

（由 scripts/generate_timeline.py 自动生成节点卡片，此处无需手写）

## 依赖与延伸

**前置(foundations):**
- [../05-transformer/01-transformer.md](../05-transformer/01-transformer.md) —— Sora 的 DiT 主干
- [../10-diffusion/01-ddpm.md](../10-diffusion/01-ddpm.md) —— Video Diffusion Models 直接扩展自 DDPM 的去噪框架
- [../10-diffusion/05-dit.md](../10-diffusion/05-dit.md) —— Sora 复用的 diffusion transformer 架构
- [../02-rnn-lstm/02-lstm.md](../02-rnn-lstm/02-lstm.md) —— World Models 的 MDN-RNN 前身

**延伸方向:**
- [../07-gpt-scaling/04-scaling-laws.md](../07-gpt-scaling/04-scaling-laws.md) —— scaling 叙事在 Sora 身上再次印证:视频生成质量随算力 / 数据规模提升的规律
