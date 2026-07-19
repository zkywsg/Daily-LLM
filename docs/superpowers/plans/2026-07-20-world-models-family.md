# 第 16 个家族(世界模型 / 视频生成)markdown 正本 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新建 `16-world-models/` 家族目录,写出 1 篇家族 README + 6 篇节点 markdown 正本(严格按年份:World Models 2018 → Video Diffusion Models 2022 → DreamerV3 2023 → Sora 2024 → Genie 2024 → GameNGen 2024),接入 `generate_timeline.py` / `FamilyId` / `tokens.css` 三处注册点,不做金标本交互页。

**Architecture:** 复用仓库已收敛 15 个家族的写作模板(frontmatter + 前作进展/核心思想+直觉/机制一二三/三件套协同/关键代码/性能数据/影响后续 九段式),每篇配 1-2 张手绘风格 SVG。内容生产走"我先给任务简报(真实论文事实+章节大纲)→ 派 subagent 写正文"的既有模式,而不是在 plan 里预先写死全文。

**Tech Stack:** 纯 markdown + SVG,Python 脚本 `scripts/generate_timeline.py` 生成 TIMELINE.md/families.json,TypeScript `FamilyId` 类型,CSS custom property。

---

## 参考:设计文档

本 plan 的所有决策依据 `docs/superpowers/specs/2026-07-20-world-models-family-design.md`,写节点前建议先读一遍该文件确认章节结构约定。

## 参考:写作模板锚点文件

写正文前先读这几个近期节点作为结构范例(不要照抄措辞,只借鉴章节骨架和"三件套协同"收尾的写法):
- `13-moe-efficient/04-deepseek-v3.md` —— flat mechanism 模式(机制一/二/三是顶层 H2)
- `13-moe-efficient/README.md` —— 家族 README 结构范例
- `11-peft-lora/02-prefix-tuning.md` —— 另一份 flat mechanism 范例,含"三件套协同"收尾

## 已知的两个必须规避的坑(本轮多次踩过)

1. **CommonMark 加粗定界符边界情况**:`**` 紧贴标点(引号/问号/括号)时,另一侧必须是空白或标点才能正确解析,不能直接接普通字符。例如 `**"xxx"**后面` 会解析失败,要写成 `**"xxx"** 后面`(加空格)或调整引号位置。写完每篇后人工过一遍 `**` 前后字符。
2. **`$` 货币符号与 remark-math 冲突**:涉及美元数字(训练成本、GPU 时租等)一律转义成 `\$`,例如 `\$5M`、`\$1.5/h`。

---

## Task 1: 家族基础设施注册

**Files:**
- Modify: `scripts/generate_timeline.py`
- Modify: `web/src/types/family.ts`
- Modify: `web/src/styles/tokens.css`

- [ ] **Step 1: 在 `scripts/generate_timeline.py` 的 `FAMILY_IDS` 列表末尾追加新家族 id**

打开 `scripts/generate_timeline.py`,找到:

```python
FAMILY_IDS = [
    "01-cnn", "02-rnn-lstm", "03-word-embedding", "04-gan",
    "05-transformer", "06-bert-family", "07-gpt-scaling",
    "08-vit", "09-multimodal-clip", "10-diffusion",
    "11-peft-lora", "12-rlhf-alignment", "13-moe-efficient",
    "14-rag-agent", "15-reasoning-o1-r1",
]
```

改成:

```python
FAMILY_IDS = [
    "01-cnn", "02-rnn-lstm", "03-word-embedding", "04-gan",
    "05-transformer", "06-bert-family", "07-gpt-scaling",
    "08-vit", "09-multimodal-clip", "10-diffusion",
    "11-peft-lora", "12-rlhf-alignment", "13-moe-efficient",
    "14-rag-agent", "15-reasoning-o1-r1", "16-world-models",
]
```

- [ ] **Step 2: 在 `web/src/types/family.ts` 的 `FamilyId` 联合类型末尾追加新家族 id**

找到以 `| "15-reasoning-o1-r1";` 结尾的联合类型定义,改成:

```typescript
  | "15-reasoning-o1-r1"
  | "16-world-models";
```

- [ ] **Step 3: 在 `web/src/styles/tokens.css` 新增家族色 token**

找到 `--family-15: #a855f7; /* o1/R1 推理 紫红 */` 这一行,在它之后新增一行:

```css
  --family-16: #d946ef; /* World Models/Video 洋红 */
```

- [ ] **Step 4: 提交**

```bash
git add scripts/generate_timeline.py web/src/types/family.ts web/src/styles/tokens.css
git commit -m "feat: 第 16 个家族(世界模型/视频生成)基础设施注册

追加 FAMILY_IDS / FamilyId / --family-16 三处,markdown 正本在后续
任务里逐篇写。此时家族目录还不存在,generate_timeline.py 暂不会扫到
新家族(FAMILY_DIR_RE 匹配不到不存在的目录会被脚本内部的
fam_dir.is_dir() 检查跳过,不报错)。"
```

---

## Task 2: 家族 README

**Files:**
- Create: `16-world-models/README.md`

- [ ] **Step 1: 写家族 README**

创建 `16-world-models/README.md`,内容如下(章节结构复用 `13-moe-efficient/README.md`,内容需要你在写作时用真实、准确的技术描述填充,以下给出每段的写作要点和必须包含的事实锚点):

```markdown
# 世界模型 / 视频生成

> **让模型学会预测"接下来会发生什么"——从在自己想象的世界里训练强化学习智能体,到生成分钟级连贯视频,再到把 diffusion 模型直接变成可交互的实时游戏引擎。**

## 一句话定位

这家族解决的是一个和"理解世界"直接相关的问题——**能不能让模型学出一个关于世界如何运作的内部模型,然后用这个模型做预测、做决策、甚至生成可交互的环境?** 2018 年 Ha & Schmidhuber 的 **World Models** 首次证明:一个强化学习智能体可以完全在自己学到的"梦境"(RNN 生成的想象轨迹)里训练策略,再迁移回真实环境。这条"学一个世界模型来做决策"的思路在 2023 年被 **DreamerV3** 规模化到 150+ 个跨领域任务、固定超参数不调参就能打平各领域的 model-free SOTA。与此同时,另一条独立发展的脉络是**用生成模型直接产出视频画面本身**——2022 年 **Video Diffusion Models** 把 DDPM 从图像推广到视频,2024 年 **Sora** 把 DiT 规模化到分钟级连贯视频,论文明确提出"video generation models as world simulators"的定位。这两条脉络在 2024 年汇合:**Genie** 证明可以从无标注的互联网视频里无监督学出逐帧可控制的生成式环境,**GameNGen** 则证明 diffusion 模型可以完全替代传统游戏引擎的渲染循环,实时生成可玩的 DOOM。这家族要回答的问题是:**从"在想象里训练策略"到"生成可交互的世界本身",世界模型这一支是怎么和视频生成技术合流的**。

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

(此处由 `scripts/generate_timeline.py` 自动生成,写作时留空表格结构由脚本填充逻辑決定,参考其他家族 README 里"子时间线"章节紧跟在其后的呈现方式,通常是脚本读取 frontmatter 后拼出的卡片列表,不需要手写)

## 依赖与延伸

- 前置依赖:[Transformer](../05-transformer/01-transformer.md)(Sora 的 DiT 主干)、[DDPM](../10-diffusion/01-ddpm.md)(Video Diffusion Models 直接扩展自 DDPM 的去噪框架)、[DiT](../10-diffusion/05-dit.md)(Sora 复用的 diffusion transformer 架构)、[LSTM](../02-rnn-lstm/02-lstm.md)(World Models 的 MDN-RNN 前身)
- 延伸方向:[Scaling Laws](../07-gpt-scaling/04-scaling-laws.md) 的 scaling 叙事在 Sora 身上再次印证——视频生成质量随算力/数据规模提升的规律
```

注意:"子时间线"章节的具体渲染逻辑由 `scripts/generate_timeline.py` 的 `parse_family_readme` 函数决定(读取 README 的"一句话定位"和标题作为 `label`/`blurb`,不需要手写子时间线表格本身,那是脚本从各节点 frontmatter 自动拼出来插入 `web/src/data/families.json` 的,不在 README.md 源文件里手写具体条目)。写 README 时重点写好"一句话定位"、"概念本身"、"依赖与延伸"三段,"子时间线"这个二级标题可以保留但不需要手写表格内容。

- [ ] **Step 2: 提交**

```bash
git add 16-world-models/README.md
git commit -m "feat: 世界模型/视频生成 家族 README"
```

---

## Task 3: 节点 01 —— World Models(2018)

**Files:**
- Create: `16-world-models/01-world-models.md`
- Create: `16-world-models/assets/01-world-models-architecture.svg`(至少 1 张)

- [ ] **Step 1: 写节点正文**

创建 `16-world-models/01-world-models.md`。Frontmatter:

```yaml
---
name: "World Models"
year: 2018
family: "16-world-models"
order: 1
paper: "World Models"
authors: ["David Ha", "Jürgen Schmidhuber"]
key_idea: "把智能体拆成 V(VAE 视觉压缩)+ M(MDN-RNN 时序预测)+ C(极小线性控制器)三部分,C 完全在 M 生成的'梦境'里用进化策略训练,首次证明智能体可以脱离真实环境、完全在自己学到的世界模型内部完成策略训练"
---
```

正文九段式结构,以下给出每段必须包含的真实事实(写作时先用 WebSearch/WebFetch 核实这些数字,若核实后与此处不同以核实结果为准,不要盲目照抄):

- **前作进展**:2018 年之前,RL 智能体要么直接在原始像素观测上做 model-free RL(如 DQN、A3C),样本效率低、训练慢;要么用 model-based RL 但世界模型通常和策略网络耦合在一起联合训练,难以独立评估世界模型质量。Ha & Schmidhuber 的洞察是把"学世界模型"和"学策略"彻底解耦成两个独立训练阶段。
- **核心思想 + 直觉**:V 负责把每一帧压缩成一个低维隐向量 z(用 VAE,典型维度如 32 维);M 是一个 MDN-RNN(mixture density network 输出的 RNN),建模 P(z_{t+1} | z_t, a_t, h_t) 为高斯混合分布,即"给定当前压缩表征和动作,预测下一时刻的压缩表征分布";C 是一个极小的线性模型(输入 [z_t, h_t],输出动作 a_t),因为足够小(数百个参数量级)所以可以用 CMA-ES(一种进化策略,不需要梯度)直接优化,不需要反向传播。
- **机制一(V:视觉压缩)**:VAE 把每帧原始像素压缩成低维隐向量,重建损失驱动学习到的表征保留视觉上的关键信息。
- **机制二(M:时序动态预测)**:MDN-RNN 学习环境动态,关键是输出**混合高斯分布**而不是单一确定性预测——因为环境未来往往有多种可能性(比如游戏里对手可能往左也可能往右),单一确定性预测会退化成"平均"这些可能性,而混合高斯能表达多峰不确定性。
- **机制三(C:极小控制器 + 在梦境里训练)**:C 极小,直接用进化策略(CMA-ES)优化,不需要梯度。最关键的一步是把 C 的训练完全放到 M 生成的"想象轨迹"(hallucinated rollouts,用 M 自己采样出的 z 序列模拟环境反馈,不接触真实环境)里进行,训完后再把策略迁移回真实环境测试。
- **三件套协同**:V 不准,C 学到的策略在真实环境里对不上;M 不准,在梦境里训出来的策略在真实环境里会失效(尤其是 M 如果学得"过于确定"、缺少不确定性建模,C 会利用 M 的预测漏洞刷分但那些漏洞在真实环境不存在);C 如果不够小或者用梯度法直接在梦境里训练,容易过拟合 M 的想象、在真实环境泛化差。三者必须同时到位。
- **关键代码**:给一段简化的 PyTorch/伪代码,展示 VAE encoder → MDN-RNN 预测下一步分布 → 从想象轨迹里采样训练 controller 的核心循环结构(不需要完整可运行,展示核心接口即可,参照仓库其他节点"关键代码"段的简化风格,如 `13-moe-efficient/04-deepseek-v3.md` 的"关键代码"一节的详略程度)。
- **性能数据**:在 CarRacing-v0 和 VizDoom(DoomTakeCover)两个基准上的结果需要核实后填写真实数字(核实来源:原论文摘要/结果表)。
- **影响 / 后续**:这篇论文确立了"V/M/C 三段式世界模型"的范式,直接启发了后续 PlaNet、Dreamer 系列(在本家族 02-video-diffusion-models 之后的 03-dreamerv3 节点里详细展开演化关系)。用 `→ 03-dreamerv3.md · 世界模型规模化到跨领域` 这样的格式加跨节点链接。

- [ ] **Step 2: 画配图 SVG**

创建 `16-world-models/assets/01-world-models-architecture.svg`,画出 V→M→C 的架构流程图(观测帧 → VAE → z → MDN-RNN(结合动作 a)→ 预测下一个 z 分布 → Controller 输出动作 → 反馈回 M)。SVG 必须是合法 XML——写完后用以下命令自查(仓库已有的 `web/src/test/svgAssets.test.ts` 会在后续 web 端测试里自动扫描此文件,但写作阶段就该保证合法):

```bash
python3 -c "
import xml.etree.ElementTree as ET
ET.parse('16-world-models/assets/01-world-models-architecture.svg')
print('OK')
"
```

Expected: `OK`(如果报错,检查 SVG 文字内容里是否有未转义的 `<`、`&` 等字符,或重复属性)

- [ ] **Step 3: 检查 `**` 加粗定界符与 `$` 转义**

```bash
grep -n '\*\*[^*]*[?"）)]\*\*[^ \n,。—-]' 16-world-models/01-world-models.md
grep -n '\$[0-9]' 16-world-models/01-world-models.md
```

Expected: 第一条命令若有输出,说明存在 `**` 紧贴标点又紧跟普通字符的边界情况,需要按 spec 里的规则手动修正(加空格或调整标点位置)。第二条命令若有输出,检查是否是未转义的货币符号,是的话改成 `\$`。

- [ ] **Step 4: 提交**

```bash
git add 16-world-models/01-world-models.md 16-world-models/assets/01-world-models-architecture.svg
git commit -m "feat: World Models(2018)节点正文"
```

---

## Task 4: 节点 02 —— Video Diffusion Models(2022)

**Files:**
- Create: `16-world-models/02-video-diffusion-models.md`
- Create: `16-world-models/assets/02-vdm-architecture.svg`(至少 1 张)

- [ ] **Step 1: 写节点正文**

Frontmatter:

```yaml
---
name: "Video Diffusion Models"
year: 2022
family: "16-world-models"
order: 2
paper: "Video Diffusion Models"
authors: ["Jonathan Ho", "Tim Salimans", "Alexey Gritsenko", "William Chan", "Mohammad Norouzi", "David J. Fleet"]
key_idea: "把 DDPM 的去噪框架从图像推广到视频:用时空分解卷积(2D 空间卷积 + 1D 时间卷积)代替昂贵的 3D 卷积,图像/视频联合训练复用大规模图像数据,是'用 diffusion 生成视频'这条路线的起点"
---
```

九段式结构写作要点(先核实真实数字再落笔):

- **前作进展**:2020 年 DDPM 在图像生成上证明了 diffusion 的可行性,但直接把图像 diffusion 的 2D U-Net 扩展成 3D(加一个时间维)会让计算量和显存爆炸式增长,而且视频数据集规模远小于图像数据集(标注视频昂贵)。
- **核心思想 + 直觉**:核心洞察是不需要真正的 3D 卷积——把每个卷积层拆成"2D 空间卷积(在每一帧内部做空间卷积,时间维度当作 batch 维度处理)+ 1D 时间卷积(在每个空间位置上沿时间维做卷积/attention)"的分解形式,大幅降低计算量,同时保留建模时空关系的能力。
- **机制一(时空分解架构)**:详细展开空间卷积和时间卷积如何交替堆叠,以及为什么这种分解比全 3D 卷积效率高得多但表达力损失很小。
- **机制二(图像/视频联合训练)**:把静止图像当作"单帧视频"混入训练数据,让模型可以复用远大于视频数据集的图像数据集,提升视觉质量和泛化。
- **机制三(条件生成的引导技术)**:引入 reconstruction guidance / classifier-free guidance 等引导技术提升样本质量和条件一致性,以及如何通过对已生成视频的末尾帧做条件生成来自回归地扩展视频长度。
- **三件套协同**:时空分解让训练/推理在算力上可行;联合训练解决数据稀缺问题;引导技术保证生成质量和长视频的一致性——三者缺一,"用 diffusion 生成高质量长视频"在 2022 年都无法成立。
- **关键代码**:展示分解卷积模块的简化伪代码(2D conv 后接 1D temporal conv 的模块结构)。
- **性能数据**:在 UCF-101 等视频生成基准上的定量结果(FVD 等指标),核实真实数字后填写。
- **影响 / 后续**:直接启发后续 Imagen Video、Make-A-Video 等作品,以及 2024 年 Sora 采用的架构范式(虽然 Sora 换成了 DiT 而不是 U-Net,但"视频当作时空数据统一建模"的思路一脉相承)。加跨节点链接 `→ ../10-diffusion/01-ddpm.md · 本文直接扩展的图像 diffusion 基础` 和 `→ 04-sora.md · 规模化到分钟级连贯视频`。

- [ ] **Step 2: 画配图 SVG**

创建 `16-world-models/assets/02-vdm-architecture.svg`,画出时空分解卷积模块结构图(输入视频帧序列 → 2D 空间卷积(逐帧独立)→ 1D 时间卷积(逐空间位置沿时间)→ 输出)。

- [ ] **Step 3: XML 合法性 + 加粗/货币符号自查**(同 Task 3 Step 2-3 的检查命令,路径替换成本任务对应文件)

- [ ] **Step 4: 提交**

```bash
git add 16-world-models/02-video-diffusion-models.md 16-world-models/assets/02-vdm-architecture.svg
git commit -m "feat: Video Diffusion Models(2022)节点正文"
```

---

## Task 5: 节点 03 —— DreamerV3(2023)

**Files:**
- Create: `16-world-models/03-dreamerv3.md`
- Create: `16-world-models/assets/03-dreamerv3-architecture.svg`(至少 1 张)

- [ ] **Step 1: 写节点正文**

Frontmatter:

```yaml
---
name: "DreamerV3"
year: 2023
family: "16-world-models"
order: 3
paper: "Mastering Diverse Domains through World Models"
authors: ["Danijar Hafner", "Jurgis Pasukonis", "Jimmy Ba", "Timothy Lillicrap"]
key_idea: "把 latent imagination 式的 model-based RL 规模化到跨领域通吃(Atari/DMC/Minecraft 等 150+ 任务),固定同一套超参数不调参就能匹配甚至超过各领域的 model-free SOTA,包括无需人类数据/课程学习拿到 Minecraft 钻石"
---
```

九段式结构写作要点(先核实真实数字):

- **前作进展**:World Models(2018)证明了"在梦境里训练策略"的可行性,但只在简单任务(CarRacing、VizDoom)上验证;后续 PlaNet、Dreamer、DreamerV2 逐步把这个思路扩展到更复杂的连续控制和 Atari 任务,但每个领域往往需要针对性调超参数,跨领域泛化性差。
- **核心思想 + 直觉**:DreamerV3 的核心洞察是"世界模型 + latent imagination"这套范式本身足够通用,真正阻碍跨领域泛化的是**不同领域的观测/奖励尺度差异巨大**(比如 Atari 的奖励和 Minecraft 的奖励量级完全不同),需要专门的归一化技术让同一套超参数在所有领域都稳定。
- **机制一(RSSM 世界模型:离散隐变量)**:用 Recurrent State-Space Model(RSSM)学习世界模型,隐状态用**离散分类变量**(而不是像 World Models 的连续高斯/MDN),离散表征让世界模型对不同领域的视觉复杂度更鲁棒。
- **机制二(symlog 归一化 + 跨尺度稳定训练)**:用 symlog 变换处理不同领域间奖励/价值量级差异悬殊的问题,不需要针对每个领域手工调整奖励缩放。
- **机制三(纯 latent imagination 训练 actor-critic)**:policy(actor)和 value function(critic)完全在世界模型生成的想象轨迹里训练,通过想象轨迹反传梯度更新 actor/critic,不需要在每一步都和真实环境交互。
- **三件套协同**:离散世界模型表征让不同领域的视觉输入都能被稳定建模;symlog 归一化让同一套超参数能应对不同量级的奖励;纯 imagination 训练让样本效率跨领域保持一致——三者共同实现了论文标题里的"mastering diverse domains"。
- **关键代码**:RSSM 前向传播 + imagination rollout 的简化伪代码。
- **性能数据**:核实真实数字后填写——论文报告的跨领域基准数量(如"150+ 任务"或论文实际报告的具体数字)、Minecraft 收集钻石这一具体成就的达成情况(是否首次不用人类数据/课程学习)。
- **影响 / 后续**:被认为是"model-based RL 通用化"的里程碑,与视频生成脉络在 Genie(2024)身上开始合流(Genie 的 latent action model 和 DreamerV3 的世界模型都用离散隐变量表征动态)。加跨节点链接 `→ 01-world-models.md · 本文继承的"想象里训练"范式起点`、`→ 05-genie.md · 离散隐变量表征动态的思路在此复用到无监督环境生成`。

- [ ] **Step 2: 画配图 SVG**

创建 `16-world-models/assets/03-dreamerv3-architecture.svg`,画出 RSSM 世界模型 + actor-critic 在想象轨迹里训练的循环图。

- [ ] **Step 3: XML 合法性 + 加粗/货币符号自查**

- [ ] **Step 4: 提交**

```bash
git add 16-world-models/03-dreamerv3.md 16-world-models/assets/03-dreamerv3-architecture.svg
git commit -m "feat: DreamerV3(2023)节点正文"
```

---

## Task 6: 节点 04 —— Sora(2024)

**Files:**
- Create: `16-world-models/04-sora.md`
- Create: `16-world-models/assets/04-sora-architecture.svg`(至少 1 张)

- [ ] **Step 1: 写节点正文**

Frontmatter:

```yaml
---
name: "Sora"
year: 2024
family: "16-world-models"
order: 4
paper: "Video generation models as world simulators"
authors: ["OpenAI"]
key_idea: "把 DiT 规模化到分钟级、多分辨率、多时长连贯视频:用 spacetime patches 统一表示不同长宽比/时长的时空数据,论文明确提出'video generation models are world simulators'的定位"
---
```

九段式结构写作要点:

- **前作进展**:Video Diffusion Models(2022)证明了 diffusion 可以生成视频,但受限于固定分辨率/时长的训练方式,以及 U-Net 架构在规模化(scaling)上不如 Transformer 干净的 scaling curve(呼应 [DiT](../10-diffusion/05-dit.md) 节点里"U-Net scaling 不如 Transformer 可预测"的论点)。
- **核心思想 + 直觉**:核心洞察是把视频压缩到低维时空 latent 空间后,不再强行 resize/crop 成固定分辨率,而是把原生分辨率/长宽比/时长的视频统一表示成"spacetime patches"(类似 ViT 把图像切成 patch 当 token,这里把时空 latent 切成时空 patch 当 token),用 Transformer(DiT)处理这些 patch 序列。
- **机制一(视频压缩网络 + spacetime patches)**:一个视频压缩网络把原始视频压缩到低维时空 latent,再把这个 latent 切分成 spacetime patches 序列,patch 数量随分辨率/时长自然变化(不需要 resize 到固定尺寸)。
- **机制二(DiT 主干规模化到视频)**:核实真实做法(diffusion transformer 处理 patch 序列,类似 [DiT](../10-diffusion/05-dit.md) 的 patchify + adaLN 条件注入范式,但条件从 class label 换成 text embedding)。
- **机制三(原生分辨率/时长训练)**:直接在不同分辨率、长宽比、时长的视频上训练(而不是统一 resize/crop),论文报告这样训练出的模型能更好地保持画面构图和取景的多样性。
- **三件套协同**:spacetime patches 统一表示让任意分辨率/时长的视频都能进 Transformer;DiT 主干提供干净的 scaling curve;原生分辨率训练保留了真实视频分布的多样性——三者共同让 Sora 能生成"分钟级、高保真、多样构图"的视频。
- **关键代码**:spacetime patchify 的简化伪代码(参考仓库 `10-diffusion/05-dit.md` 节点"关键代码"段的 patchify 实现风格)。
- **性能数据**:Sora 是技术报告而非同行评审论文,没有标准 benchmark 数字表,这一段改写成"论文展示的定性能力清单"(核实真实报告内容后填写,如最长生成时长、展示的物理一致性案例类型等),不要编造不存在的量化 benchmark 分数。
- **影响 / 后续**:直接催生"video generation as world simulator"这一研究方向的爆发,后续 Genie / GameNGen 都在这一叙事框架下工作。加跨节点链接 `→ ../10-diffusion/05-dit.md · Sora 复用的 DiT 主干架构`、`→ ../07-gpt-scaling/04-scaling-laws.md · scaling 叙事在视频生成上的再次印证`、`→ 05-genie.md · 从"生成视频"到"生成可交互环境"`。

- [ ] **Step 2: 画配图 SVG**

创建 `16-world-models/assets/04-sora-architecture.svg`,画出视频压缩 → spacetime patchify → DiT 处理 → 解压缩输出的流程图。

- [ ] **Step 3: XML 合法性 + 加粗/货币符号自查**

- [ ] **Step 4: 提交**

```bash
git add 16-world-models/04-sora.md 16-world-models/assets/04-sora-architecture.svg
git commit -m "feat: Sora(2024)节点正文"
```

---

## Task 7: 节点 05 —— Genie(2024)

**Files:**
- Create: `16-world-models/05-genie.md`
- Create: `16-world-models/assets/05-genie-architecture.svg`(至少 1 张)

- [ ] **Step 1: 写节点正文**

Frontmatter:

```yaml
---
name: "Genie"
year: 2024
family: "16-world-models"
order: 5
paper: "Genie: Generative Interactive Environments"
authors: ["Jake Bruce", "Michael Dennis", "Ashley Edwards"]
key_idea: "无监督地从海量无标注互联网视频里学出逐帧可控制的生成式环境:隐式学习出离散的 latent action 空间,不需要任何人工动作标注,用户可以用学到的离散动作逐帧'玩'生成出来的世界"
---
```

（注:authors 字段先列出论文的核心/一作代表,写作时核实完整作者列表后按需扩充,不要虚构不在作者列表里的人名。）

九段式结构写作要点:

- **前作进展**:Sora(2024)证明视频生成模型可以生成逼真、时空一致的视频,但生成的视频是"被动播放"的——不能像游戏一样让用户实时控制画面接下来发生什么。同时,DreamerV3 这条脉络虽然能学出可控制的世界模型,但依赖于已有明确动作空间的强化学习环境(比如 Atari 游戏本身定义好了按键动作),不能直接用在没有动作标注的海量互联网视频上。
- **核心思想 + 直觉**:核心洞察是把"学习可控制的世界模型"和"需要动作标注"这两件事解耦——即便视频里没有任何动作标注,只要视频里存在"某个东西的状态在连续帧之间发生了变化",就有可能反推出一个隐式的、离散的"动作"空间来解释这些变化,完全无监督学出来。
- **机制一(视频 tokenizer)**:把原始视频帧压缩成离散/连续的视觉 token 序列(核实具体用的是 VQ-VAE 还是其他 tokenizer)。
- **机制二(Latent Action Model,LAM)**:核心创新——一个模型观察相邻两帧,推断出一个很小的离散"latent action"集合(核实论文报告的具体数值,如动作空间大小),完全无监督(不看任何人工标注的按键/动作数据),只靠"预测下一帧需要什么隐式动作"这个自监督目标训练出来。
- **机制三(动态模型:给定 latent action 自回归生成下一帧)**:一个自回归/MaskGIT 风格的 transformer,给定过去帧的 token + LAM 推断出的 latent action,预测下一帧的 token,让用户在推理时手动挑选 latent action 来"操控"生成的视频。
- **三件套协同**:视频 tokenizer 把原始像素变成可处理的离散序列;LAM 无监督学出可控制的"操作把手";动态模型让这些操作把手真正能影响生成内容——三者共同让 Genie 能做到"从无标注视频里学出可玩的游戏"。
- **关键代码**:LAM 训练目标的简化伪代码(输入两帧,输出离散 latent action,用这个 latent action 重建/预测下一帧作为自监督信号)。
- **性能数据**:核实真实数字后填写——训练数据规模(视频时长)、模型参数量、latent action 空间大小等论文报告的具体数字,不确定的数字必须先核实再写,不要用记忆里模糊的数字。
- **影响 / 后续**:证明了"从视频里无监督学可控制性"这条路径可行,为后续通用世界模型 / 具身智能训练数据的自动化生成提供了新思路。加跨节点链接 `→ 03-dreamerv3.md · 离散隐变量表征思路的呼应`、`→ 06-gamengen.md · 同样是"生成可交互环境",GameNGen 走的是有监督/RL 数据路线而非无监督`。

- [ ] **Step 2: 画配图 SVG**

创建 `16-world-models/assets/05-genie-architecture.svg`,画出 video tokenizer → LAM(相邻帧推断 latent action)→ 动态模型(latent action + 历史帧 → 下一帧)的流程图。

- [ ] **Step 3: XML 合法性 + 加粗/货币符号自查**

- [ ] **Step 4: 提交**

```bash
git add 16-world-models/05-genie.md 16-world-models/assets/05-genie-architecture.svg
git commit -m "feat: Genie(2024)节点正文"
```

---

## Task 8: 节点 06 —— GameNGen(2024)

**Files:**
- Create: `16-world-models/06-gamengen.md`
- Create: `16-world-models/assets/06-gamengen-architecture.svg`(至少 1 张)

- [ ] **Step 1: 写节点正文**

Frontmatter:

```yaml
---
name: "GameNGen"
year: 2024
family: "16-world-models"
order: 6
paper: "Diffusion Models Are Real-Time Game Engines"
authors: ["Dani Valevski", "Yaniv Leviathan", "Moab Arar", "Shlomi Fruchter"]
key_idea: "用条件 diffusion 模型完全替代传统游戏引擎的渲染循环,实时交互式生成可玩的 DOOM 画面,证明神经网络可以端到端承担游戏引擎的职责"
---
```

九段式结构写作要点:

- **前作进展**:Genie(2024)证明了从无标注视频里能学出可控制的生成式环境,但生成的画面质量/分辨率、以及交互的实时性(是否能做到实时帧率的流畅体验)不是 Genie 论文的重点。传统游戏引擎(如 DOOM 的渲染引擎)则是完全手工编写的确定性程序,不是学出来的。
- **核心思想 + 直觉**:核心洞察是——如果有足够多的"游戏画面 + 对应操作"的配对数据,一个 diffusion 模型可以学会"给定历史帧 + 当前操作,预测下一帧长什么样",而这正是游戏引擎渲染循环在做的事情;如果这个预测足够快(实时)、足够准(长时间玩下去画面不崩坏),这个 diffusion 模型本身就能替代游戏引擎。
- **机制一(RL agent 自动生成训练数据)**:先训练一个 RL 智能体去玩 DOOM,收集大量"画面帧 + 操作"的配对轨迹数据,不需要人类玩家录制。
- **机制二(条件 diffusion 预测下一帧)**:diffusion 模型以过去若干帧 + 当前操作为条件,预测下一帧画面,替代传统游戏引擎的渲染步骤。
- **机制三(噪声增强条件帧,防止自回归漂移)**:核心工程细节——如果直接把模型自己生成的历史帧原样作为下一步的条件输入,误差会随着帧数增加而累积放大(自回归漂移),GameNGen 在训练时对条件帧人为加噪声增强,让模型对这种误差累积更鲁棒,支撑长时间稳定运行。
- **三件套协同**:RL agent 提供了足够规模且覆盖面广的训练数据;条件 diffusion 提供了"预测下一帧"的生成能力;噪声增强条件帧解决了长时间运行的稳定性问题——三者共同实现了"实时可玩、长时间不崩坏"的神经网络游戏引擎。
- **关键代码**:条件 diffusion 采样循环的简化伪代码(输入历史帧 buffer + 当前动作,输出下一帧,更新 buffer)。
- **性能数据**:核实真实数字后填写——报告的帧率(FPS)、使用的硬件(如 TPU 型号)、能维持"看起来可玩"的连续运行时长等。
- **影响 / 后续**:是"diffusion 模型完全替代游戏引擎"这一设想的第一个可工作原型,后续会不会规模化到更复杂的 3A 游戏、是否会催生"神经网络渲染"这一新方向,是这篇论文留下的开放问题。加跨节点链接 `→ 05-genie.md · 都是"生成可交互环境",本文换成了有监督 RL 轨迹数据而非无监督视频学习`、`→ 01-world-models.md · 呼应本家族开篇"在学到的世界模型里训练/运行"这一命题的最新形态`。

- [ ] **Step 2: 画配图 SVG**

创建 `16-world-models/assets/06-gamengen-architecture.svg`,画出 RL agent 生成数据 → 条件 diffusion 训练 → 推理时自回归生成(历史帧+动作→下一帧,噪声增强防漂移)的流程图。

- [ ] **Step 3: XML 合法性 + 加粗/货币符号自查**

- [ ] **Step 4: 提交**

```bash
git add 16-world-models/06-gamengen.md 16-world-models/assets/06-gamengen-architecture.svg
git commit -m "feat: GameNGen(2024)节点正文"
```

---

## Task 9: 生成 TIMELINE.md / families.json + 全项目验证

**Files:**
- Modify (自动生成,不手写): `TIMELINE.md`
- Modify (自动生成,不手写): `web/src/data/families.json`

- [ ] **Step 1: 运行生成脚本**

```bash
python3 scripts/generate_timeline.py
```

Expected: 无报错退出。

- [ ] **Step 2: 检查 TIMELINE.md 里新家族的 6 行是否按年份正确插入**

```bash
grep -n "World Models\|Video Diffusion Models\|DreamerV3\|Sora\|Genie\|GameNGen" TIMELINE.md
```

Expected: 6 行都出现,且 `\`16-world-models\`` 出现在对应行里,年份列(2018/2022/2023/2024/2024/2024)与其他行的年份顺序保持全表升序(TIMELINE.md 是全仓库所有家族按年份统一排序的表,不是分家族的表)。

- [ ] **Step 3: 检查 families.json 里新家族块**

```bash
python3 -c "
import json
data = json.load(open('web/src/data/families.json'))
fam = next((f for f in data['families'] if f['id'] == '16-world-models'), None)
assert fam is not None, '16-world-models 家族块缺失'
assert len(fam['nodes']) == 6, f'期望 6 个节点,实际 {len(fam[\"nodes\"])}'
assert fam['colorToken'] == '--family-16', f'colorToken 不对: {fam[\"colorToken\"]}'
print('OK', fam['label'])
"
```

Expected: 打印 `OK 世界模型 / 视频生成`(或你在 README 里写的实际标题)

- [ ] **Step 4: 全仓库 SVG 合法性自查**

```bash
python3 -c "
import xml.etree.ElementTree as ET
import glob
bad = []
for f in sorted(glob.glob('16-world-models/assets/*.svg')):
    try:
        ET.parse(f)
    except Exception as e:
        bad.append((f, str(e)))
print(f'扫描 {len(glob.glob(\"16-world-models/assets/*.svg\"))} 个,{len(bad)} 个非法')
for f, e in bad:
    print(f'  {f}: {e}')
"
```

Expected: `扫描 6 个(或更多,若某节点配了 2 张图),0 个非法`

- [ ] **Step 5: 全项目 tsc + vitest**

```bash
cd web
npx tsc --noEmit
npx vitest run
```

Expected: tsc 无输出(clean exit);vitest 全部通过,测试数量与新增家族前一致(新节点没有 `lib/prose.ts` 也没有注册进 `goldenSamples`,`AllGoldenSamples.smoke.test.tsx` / `ProseCompleteness.test.tsx` 都不会新增用例,但 `web/src/test/svgAssets.test.ts` 会新增 6+ 条,因为它扫描全仓库 `*/assets/*.svg`)

- [ ] **Step 6: 浏览器验证家族页面可正常访问**

用 preview_start 起 dev server,访问 `/families/16-world-models` 确认:
- 家族标题、"一句话定位"正文正常渲染
- 6 个节点卡片按年份显示,颜色为洋红 `#d946ef`
- 点进任一节点(如 `/families/16-world-models/04-sora`)确认走的是 `NodePage.tsx` 通用渲染路径(非金标本),标题/作者/正文/前后节点导航正常显示,console 无 error

- [ ] **Step 7: 提交**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add TIMELINE.md web/src/data/families.json
git commit -m "feat: 生成 TIMELINE.md / families.json,第 16 个家族接入完成

世界模型/视频生成家族全部 6 篇节点 markdown 正本 + README 就位,
python3 scripts/generate_timeline.py 重新生成产物,tsc + vitest 全项目
通过,浏览器验证家族页面与节点详情页渲染正常。金标本交互页留待
后续单独轮次补,本轮范围到此完成。"
```

---

## Self-Review 记录(写 plan 时已自查)

1. **Spec 覆盖**:spec 第 3 节(节点列表)→ Task 3-8;第 4 节(节点写作规范)→ 每个节点 Task 的 Step 1 写作要点;第 5 节(家族 README)→ Task 2;第 6 节(注册文件)→ Task 1 + Task 9;第 7 节(验收标准)→ Task 9 的检查命令逐条对应。
2. **Placeholder 扫描**:每个节点任务给出的是"必须包含的真实事实清单 + 结构大纲",不是空洞的"写好这一段"——这是内容创作类任务的等价物(内容本身留给执行者查证真实论文后落笔,而不是本 plan 编造完整正文;这一点已在 spec 第 4 节写明,是有意的设计决策,不是遗漏)。
3. **一致性检查**:6 个节点的 frontmatter `order` 字段(1-6)与年份升序(2018/2022/2023/2024/2024/2024)一致;跨节点链接指向的文件名(`01-world-models.md` `02-video-diffusion-models.md` `03-dreamerv3.md` `04-sora.md` `05-genie.md` `06-gamengen.md`)在各任务间保持一致拼写。
