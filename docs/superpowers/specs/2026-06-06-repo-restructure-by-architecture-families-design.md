# 仓库重构：以"架构/范式家族"为主轴

**日期**：2026-06-06
**作者**：通过 brainstorming 共同确定
**状态**：Design Approved，等待写实施计划

---

## 1. 背景与动机

当前仓库存在两条并行的"主轴"在抢正本：

- `timeline/` — 编年体（2012–2025）
- `tracks/{vision,language,scale-multimodal,alignment,systems}` — 主题线

这导致四个具体问题：

1. **重复感**：同一工作（如 ResNet）在 `timeline/2015` 和 `tracks/vision` 都出现，写新内容时不知道哪边是正本
2. **导航乱**：缺一个清晰的"主入口"——是按年还是按主题进
3. **概念乱**：2020 之后视觉/语言已经汇流，再分独立 track 别扭
4. **Web 对不上**：Web 端是纯时间线，但 GitHub 有 track，两边心智模型不一致

经过讨论确认：根因是**时间线和 track 两条轴在抢主位**。

## 2. 核心决策

**采用"架构/范式家族"作为唯一主轴**，按家族登场时间排序。每个家族内部挂一条子时间线，列出代表作。横切的基础知识（激活函数、反向传播、优化器等）抽到 `foundations/`。

这样：

- 大家耳熟能详的"概念本身"（CNN、LSTM、Transformer、GAN、扩散…）有一级席位
- 具体代表作（LeNet、AlexNet、ResNet…）作为子时间线节点，归属唯一
- 原来的 `tracks/` 自然消解为"主轴上若干家族连起来的路径"，不再需要独立目录
- 原来的 `timeline/` 由仓库根 README 顶部的"总时间线表" + 自动生成的 `TIMELINE.md` 替代

## 3. 主轴：15 个家族清单

按家族登场时间排序，编号即顺序。

| # | 家族 | 关键年份 | 内部子时间线（代表作） |
|---|------|---------|--------------------|
| 01 | **CNN** 卷积神经网络 | 2012– | LeNet → AlexNet → VGG → Inception → ResNet → DenseNet → EfficientNet |
| 02 | **RNN / LSTM / GRU** 循环网络 | 1997, 2014– | RNN → LSTM → GRU → Seq2Seq → Attention (Bahdanau/Luong) |
| 03 | **Word Embedding** 词嵌入 | 2013– | NNLM → Word2Vec → GloVe → FastText → ELMo |
| 04 | **GAN** 生成对抗 | 2014– | original → DCGAN → WGAN → CycleGAN → StyleGAN → BigGAN |
| 05 | **Transformer** 架构本身 | 2017 | 原论文 → Attention 变体 → 位置编码演化 |
| 06 | **预训练语言模型（BERT 系）** | 2018– | BERT → RoBERTa → ALBERT → DeBERTa |
| 07 | **大语言模型（GPT 系 + Scaling）** | 2018–2020 | GPT-1/2/3 → Scaling Laws → Chinchilla |
| 08 | **视觉 Transformer** | 2020– | ViT → DeiT → Swin → MAE |
| 09 | **多模态对齐** | 2021– | CLIP → ALIGN → BLIP → Flamingo → LLaVA |
| 10 | **扩散模型** | 2020– | DDPM → DDIM → LDM/Stable Diffusion → DiT |
| 11 | **参数高效微调（PEFT）** | 2021– | Adapter → Prefix → LoRA → QLoRA |
| 12 | **对齐与 RLHF** | 2022– | InstructGPT → ChatGPT → DPO → Constitutional AI |
| 13 | **MoE 与高效推理** | 2023– | Switch → Mixtral → DeepSeek-MoE → Flash/Ring Attention |
| 14 | **RAG 与 Agent** | 2023– | RAG → ReAct → Tool use → MCP |
| 15 | **推理模型（Test-time compute）** | 2024– | CoT → o1 → R1 → 过程奖励 |

## 4. 目录结构

### 4.1 仓库根

15 个家族文件夹**直接挂在仓库根**（不再有 `families/` 这一层），保证 `ls` 出来就是一条时间线。

```
Daily-LLM/
├── README.md              # 主入口 + 总时间线表
├── TIMELINE.md            # 自动生成的按年索引（脚本扫家族节点 frontmatter）
├── foundations/           # 横切基础
├── 01-cnn/
├── 02-rnn-lstm/
├── 03-word-embedding/
├── 04-gan/
├── 05-transformer/
├── 06-bert-family/
├── 07-gpt-scaling/
├── 08-vit/
├── 09-multimodal-clip/
├── 10-diffusion/
├── 11-peft-lora/
├── 12-rlhf-alignment/
├── 13-moe-efficient/
├── 14-rag-agent/
├── 15-reasoning-o1-r1/
├── projects/              # 跨家族实战项目（保留）
└── web/                   # 可视化网页（保留）
```

迁移完成后 `timeline/` 和 `tracks/` 删除。

### 4.2 家族目录模板

以 `01-cnn/` 为例。**每个家族用统一结构**，否则会乱：

```
01-cnn/
├── README.md              # 家族脸面（核心）
├── 01-lenet.md
├── 02-alexnet.md
├── 03-vgg.md
├── 04-inception.md
├── 05-resnet.md           # 简单节点：单文件
├── 06-densenet.md
├── 07-efficientnet/       # 复杂节点：升级成文件夹
│   ├── README.md
│   ├── code.py
│   └── notes.md
└── assets/                # 该家族的图、示意（可选）
```

**节点形态规则（混合 C）**：

- **默认**：单 .md 文件（一两千字 + 公式 + 极简代码片段就够）
- **升级条件**：当节点需要带完整可跑代码、多张图、长 notes 时，升级为子文件夹 `NN-name/README.md + 配套资源`
- 强行统一会让简单节点臃肿、复杂节点装不下，故按需升级

**家族 README.md 固定四块**（直接讲概念，**不另开 `00-concept.md`**）：

1. **一句话定位** — 这个家族解决了什么、被什么逼出来的
2. **概念本身** — "什么是 CNN / LSTM / Transformer …"，直觉解释 + 关键公式
3. **子时间线表** — 一张表：年份 / 名字 / 关键贡献 / 之前卡在哪
4. **依赖与延伸** — 前置 `foundations/` 清单 + 通向后面哪些家族

**子时间线节点固定四块**（无论 .md 还是文件夹）：

- **之前卡在哪**（一句话）
- **核心思想**（公式 + 直觉）
- **关键代码片段**（PyTorch 极简实现）
- **影响 / 后续**（被谁继承、被谁取代）

每个节点 markdown 顶部带 frontmatter，供 `TIMELINE.md` 自动生成脚本和 Web 端读取：

```yaml
---
name: ResNet
year: 2015
family: 01-cnn
order: 5
paper: "Deep Residual Learning for Image Recognition"
authors: [He Kaiming, ...]
key_idea: "残差连接让 152 层网络可训练"
---
```

### 4.3 foundations 切法

按"概念类别"分组，不按时间：

```
foundations/
├── README.md                  # 总览 + 引用关系（哪个家族用到哪些）
├── 01-neural-network-basics/  # 神经元、前向、反向传播、链式法则
├── 02-activations/            # ReLU、GELU、Swish、Softmax
├── 03-optimizers/             # SGD、Momentum、Adam、AdamW
├── 04-normalization/          # BatchNorm、LayerNorm、RMSNorm
├── 05-initialization/         # Xavier、He、正交初始化
├── 06-losses/                 # CE、MSE、对比损失、KL
├── 07-regularization/         # Dropout、权重衰减、早停
├── 08-attention-mechanism/    # 通用注意力（不绑定 Transformer）
└── 09-tokenization-embedding/ # BPE、WordPiece、SentencePiece
```

子目录形态规则与家族节点相同（默认 .md，复杂时升级文件夹）。

**Attention 留在 foundations 而非塞进 Transformer**：它不止 Transformer 用——Seq2Seq+Attention（02）、CLIP（09）等都引用，是横切的典型。

### 4.4 引用规则

**内容只去一个地方，不复制。**

- 家族节点讲到 BatchNorm，**链到 `foundations/04-normalization/`**，不重复讲
- 每个家族 README 顶部列出"前置依赖"清单（指向 foundations 的相对链接）
- 跨家族的引用（如 ViT 引用 Transformer）也走相对链接，正本只在主家族

## 5. 迁移路径

### 5.1 映射表（旧 → 新）

| 旧位置 | 新位置 |
|--------|--------|
| `timeline/2012/alexnet` | `01-cnn/02-alexnet.md` |
| `timeline/2015/resnet` | `01-cnn/05-resnet.md` |
| `timeline/2017/transformer` | `05-transformer/01-original.md`（或目录） |
| `timeline/<year>/<work>` | 拆进对应家族 |
| `tracks/vision/` | 拆进 `01-cnn/`、`04-gan/`、`08-vit/` |
| `tracks/language/` | 拆进 `02-rnn-lstm/`、`03-word-embedding/`、`05-transformer/`、`06-bert-family/`、`07-gpt-scaling/` |
| `tracks/scale-multimodal/` | 拆进 `07-gpt-scaling/`、`08-vit/`、`09-multimodal-clip/` |
| `tracks/alignment/` | 拆进 `11-peft-lora/`、`12-rlhf-alignment/` |
| `tracks/systems/` | 拆进 `13-moe-efficient/`、`14-rag-agent/` |
| 旧 `foundations/`（现存） | 按 4.3 重切进新 `foundations/` 各子目录 |

### 5.2 迁移哲学：重写而非搬运

旧 `timeline/` 和 `tracks/` 内容质量参差不齐。迁移过程是**重写**——旧内容当作素材参考，写得不行的直接丢，不机械复制。每个新家族节点按 4.2 的模板重新组织。

### 5.3 收尾

- 迁移完所有内容后，**删除 `timeline/` 和 `tracks/` 目录**
- 在仓库根生成 `TIMELINE.md`（脚本扫所有家族节点的 frontmatter 生成），作为按年索引的只读速查表
- 更新根 `README.md`：顶部放"总时间线表"（15 家族 + 关键年份 + 一句话定位），替换现有的年表
- 更新 `CLAUDE.md` 中的"项目结构"段，删除 `timeline/` 和 `tracks/` 的描述，替换为新结构

## 6. Web ↔ GitHub 联动（方向，细节后续单独 brainstorm）

主轴和 Web 一一对应：

- **主时间线视图** = 15 个家族节点横向排开，按年份定位
- **点击家族** = 进入该家族子时间线，列出代表作
- **点击代表作** = 渲染对应 .md（或构建时转 JSON）
- **URL 结构** = `/01-cnn` → `/01-cnn/05-resnet`，与仓库根下的文件路径一一对应（家族目录已扁平到根，URL 也不加 `families/` 前缀）
- 元数据来源：节点 frontmatter，GitHub 改动直接反映到 Web，不维护两套

Web 端的具体实现（路由、渲染、子时间线交互形式）留到后面单独立项。

## 7. 不在本次范围内的事

- Web 端的具体重构（仅敲定主轴对应方向，实现单独 brainstorm）
- `projects/` 的内部结构调整
- 各家族节点的具体内容质量（迁移时按需重写，不在本 spec 列清单）
- 双语策略（沿用现状：中文优先，按需补 EN）

## 8. 验收标准

重构完成后应满足：

1. 仓库根 `ls` 输出按编号排序，呈现一条 15 家族的时间线
2. `timeline/` 和 `tracks/` 目录不再存在
3. 每个家族目录符合 4.2 的模板（README.md 四块齐全，节点带 frontmatter）
4. 任何一个具体工作（如 ResNet）在仓库中只有一个正本
5. `foundations/` 按 4.3 切分，被家族节点正确引用
6. 根 `README.md` 总时间线表替代旧年表，`TIMELINE.md` 由脚本生成
7. `CLAUDE.md` 中的"项目结构"段已更新
