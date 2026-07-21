# 第 17 个家族(图神经网络 GNN)markdown 正本 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新建 `17-graph-neural-networks/` 家族目录,写出 1 篇家族 README + 5 篇节点 markdown 正本(严格按年份:GCN 2017 → GraphSAGE 2017 → GAT 2018 → GIN 2019 → Graphormer 2021),接入 `generate_timeline.py` / `FamilyId` / `tokens.css` 三处注册点,不做金标本交互页。

**Architecture:** 复用仓库已收敛 16 个家族的写作模板(frontmatter + 前作进展/核心思想+直觉/机制一二三/三件套协同/关键代码/性能数据/影响后续 九段式),每篇配 1-2 张手绘风格 SVG。内容生产走"我先给任务简报(真实论文事实+章节大纲)→ 派 subagent 写正文"的既有模式。本轮吸取上一轮(16-world-models)的三个教训并在任务简报里逐条强调:SVG 命名必须匹配节点 markdown 完整 stem、家族 README 子时间线必须手写真实表格、familyHero.ts 缺口在第一个节点任务里就处理掉,不留到 review 阶段才发现。

**Tech Stack:** 纯 markdown + SVG,Python 脚本 `scripts/generate_timeline.py` 生成 TIMELINE.md/families.json,TypeScript `FamilyId` 类型,CSS custom property。

---

## 参考:设计文档

本 plan 的所有决策依据 `docs/superpowers/specs/2026-07-22-graph-neural-networks-family-design.md`,写节点前建议先读一遍该文件确认章节结构约定。

## 参考:写作模板锚点文件

写正文前先读这几个近期节点作为结构范例(不要照抄措辞,只借鉴章节骨架和"三件套协同"收尾的写法):
- `16-world-models/01-world-models.md`、`16-world-models/05-genie.md` —— 最近一轮已验证过的完整范例,含"关键代码"详略程度参考
- `16-world-models/README.md` —— 家族 README 结构范例,包括"子时间线"手写表格的真实格式
- `13-moe-efficient/04-deepseek-v3.md` —— flat mechanism 模式(机制一/二/三是顶层 H2)的另一个参考

## 已知的四个必须规避的坑(前两轮多次踩过,第四个是上一轮新发现的)

1. **CommonMark 加粗定界符边界情况**:`**` 紧贴标点(引号/问号/括号)时,另一侧必须是空白或标点才能正确解析,不能直接接普通字符。例如 `**"xxx"**后面` 会解析失败,要写成 `**"xxx"** 后面`(加空格)或调整引号位置。写完每篇后人工过一遍 `**` 前后字符。
2. **`$` 货币符号与 remark-math 冲突**:涉及美元数字一律转义成 `\$`,例如 `\$5M`。本家族大概率不涉及,但如果提到云算力成本要注意。
3. **跨节点链接必须用 markdown link 语法,不能写纯文本**:`→ 02-graphsage.md · ...` 会渲染成不可点击的纯文本,必须写成 `→ [02-graphsage.md](02-graphsage.md) · ...`。
4. **SVG 资产文件名必须以节点 markdown 的完整 stem 开头**(上一轮 Task 4 踩过的坑,当时配图叫 `02-vdm-architecture.svg` 而不是 `02-video-diffusion-models-*.svg`,导致 `scripts/generate_timeline.py` 的资产匹配规则 `{file_stem}-*.svg` 匹配不上,`families.json` 里该节点的 `assets` 数组是空的)。**例如节点 `01-gcn.md` 的配图必须命名为 `01-gcn-architecture.svg` 或 `01-gcn-xxx.svg`(以 `01-gcn-` 开头),不能用缩写或改写的名字。**
5. **跨节点事实一致性**:写"前作进展"或"影响/后续"提到已写好的兄弟节点时,必须重新读一遍那个节点自己的正文,确认自己写的 claim 与对方自述内容一致,不能凭训练知识里的一般印象凭空归因(上一轮 Sora 节点错误归因 Video Diffusion Models 的局限性,被 code review 抓到并修复)。

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
    "14-rag-agent", "15-reasoning-o1-r1", "16-world-models",
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
    "17-graph-neural-networks",
]
```

- [ ] **Step 2: 在 `web/src/types/family.ts` 的 `FamilyId` 联合类型末尾追加新家族 id**

找到以 `| "16-world-models";` 结尾的联合类型定义,改成:

```typescript
  | "16-world-models"
  | "17-graph-neural-networks";
```

- [ ] **Step 3: 在 `web/src/styles/tokens.css` 新增家族色 token**

找到 `--family-16: #d946ef; /* World Models/Video 洋红 */` 这一行,在它之后新增一行:

```css
  --family-17: #f43f5e; /* GNN 玫瑰红 */
```

- [ ] **Step 4: 验证 tsc(预期会因 familyHero.ts 报一个已知错误,不用现在修)**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web && npx tsc --noEmit
```

Expected: 报一个 `web/src/components/home/familyHero.ts` 的 `TS2741` 错误(`Record<FamilyId, string>` 缺 `"17-graph-neural-networks"` key)。这是预期的、已知的中间状态——上一轮(16-world-models)已经验证过这个错误在追加节点(Task 3)时一并修复是正确的做法,不需要现在处理,继续往下走。

- [ ] **Step 5: 提交**

```bash
git add scripts/generate_timeline.py web/src/types/family.ts web/src/styles/tokens.css
git commit -m "feat: 第 17 个家族(图神经网络 GNN)基础设施注册

追加 FAMILY_IDS / FamilyId / --family-17 三处,markdown 正本在后续
任务里逐篇写。tsc 此时会报 familyHero.ts 的已知错误,留到 Task 3(第一个
节点)一并修复,与上一轮(16-world-models)的做法一致。"
```

---

## Task 2: 家族 README

**Files:**
- Create: `17-graph-neural-networks/README.md`

- [ ] **Step 1: 写家族 README**

创建 `17-graph-neural-networks/README.md`,章节结构复用 `16-world-models/README.md`:

```markdown
# 图神经网络(GNN)

> **把深度学习从网格(图像)和序列(文本)推广到任意图结构——节点通过边互相"传消息",聚合邻居信息来学习每个节点的表征。**

## 一句话定位

CNN 处理网格结构(图像的像素网格),RNN/Transformer 处理序列结构(文本的 token 序列),但现实世界有大量数据天然是**图结构**——社交网络(节点=用户,边=关注关系)、分子结构(节点=原子,边=化学键)、知识图谱(节点=实体,边=关系)、推荐系统(节点=用户/商品,边=交互记录)。这些数据没有网格的规则邻域,也没有序列的固定顺序,每个节点的邻居数量还各不相同。图神经网络(GNN)要解决的核心问题是:**如何在这种不规则的拓扑结构上定义"卷积"或"消息传递",让每个节点能聚合邻居信息、学到有意义的表征?** 2017 年 Kipf & Welling 的 **GCN** 把谱图卷积简化成一阶邻域聚合,定义了"现代 GNN"这个范式的起点;同年 **GraphSAGE** 引入采样 + 可学习聚合函数,第一次让 GNN 能泛化到训练时没见过的节点;2018 年 **GAT** 用注意力机制替代 GCN 里固定的度数归一化系数;2019 年 **GIN** 从理论上证明了大多数 GNN 的表达力上限就是 Weisfeiler-Lehman 图同构测试,并给出能达到这一上限的架构;2021 年 **Graphormer** 则完全跳出"消息传递"框架,把标准 Transformer 搬到图上,用全局注意力 + 结构编码拿下分子性质预测挑战赛冠军。这家族要回答的问题是:**GNN 是怎么从"谱图卷积的一阶近似"演化到"图结构感知的全局注意力"的**。

## 概念本身

### "消息传递"这一统一视角

GCN、GraphSAGE、GAT、GIN 这四篇尽管公式各不相同,但都可以归纳进同一个统一框架——**消息传递神经网络(Message Passing)**:每一层,每个节点做两件事:

1. **聚合(Aggregate)**——收集所有邻居节点在上一层的表征,通过某种函数(求和/均值/最大值/注意力加权)汇总成一个"消息"
2. **更新(Update)**——把聚合到的消息和节点自己上一层的表征结合(通常过一个线性层 + 非线性激活),得到这个节点在这一层的新表征

四篇论文的核心差异,几乎都能归结为"聚合函数怎么设计"和"要不要额外的采样步骤":

- **GCN**:聚合函数是按节点度数加权的平均(谱图卷积一阶近似的直接推论),不采样,直推式(transductive)
- **GraphSAGE**:聚合函数可以是均值/LSTM/池化,聚合前先对邻居做固定大小采样,归纳式(inductive)
- **GAT**:聚合函数是学出来的注意力加权平均,不需要提前知道图的完整拓扑做矩阵运算
- **GIN**:聚合函数是求和(而非均值/最大值),理论上证明这是唯一能达到 WL test 表达力上限的聚合方式

### Graphormer:跳出消息传递框架

Graphormer(2021)不再逐层做"聚合邻居"这件事——它直接把每个节点当作 token,用标准 Transformer 的全局自注意力(每个节点看得到图里所有其他节点),再通过三种结构编码(节点度数中心性编码、节点对最短路径的空间编码、路径上边特征的编码)把图的拓扑信息直接注入 attention 的计算过程。这是"把 Transformer 迁移到新模态"这条元技术路线(呼应 [ViT](../08-vit/01-vit.md) 把 Transformer 搬到图像 patch 序列)在图数据上的又一次实践。

## 子时间线

| 年份 | 名字 | 关键贡献 | 之前卡在哪 |
|------|------|---------|-----------|
| 2017 | **GCN** | Kipf & Welling——把谱图卷积(Chebyshev 多项式近似图拉普拉斯)简化到一阶邻域聚合,一层 `D̃^(-1/2) Ã D̃^(-1/2) H W` 传播规则,定义了"现代 GNN"这个范式起点,在引文网络半监督节点分类上大幅超过此前基于图的方法 | 此前的谱图卷积(Defferrard et al. 2016 等)需要 K 阶 Chebyshev 多项式展开,计算量大、参数多,而且是全图谱分解,难以扩展到大图 |
| 2017 | **GraphSAGE** | Hamilton, Ying & Leskovec——SAmple + aggreGatE:固定大小邻域采样 + 可学习聚合函数(mean/LSTM/pooling),让 GNN 第一次能泛化到训练时没见过的节点/图(归纳式) | GCN 是直推式的,训练时需要完整图结构,新节点加入图后要重新训练整个模型,无法处理大规模动态图(如 Reddit 帖子流、社交网络新用户) |
| 2018 | **GAT** | Veličković et al.——用可学习的 attention 权重替代 GCN 里固定的度数归一化系数,配合多头注意力提升训练稳定性,在直推式和归纳式基准上都拿到 SOTA | GCN/GraphSAGE 的聚合权重要么固定(按度数)要么依赖显式图结构做矩阵运算,不能根据节点内容动态调整"哪个邻居更重要" |
| 2019 | **GIN** | Xu et al.——用 Weisfeiler-Lehman 图同构测试给 GNN 表达力定理上界:证明 mean/max 聚合(如 GraphSAGE)不如 WL test 强,提出 sum 聚合 + MLP 的 GIN,理论上证明达到 WL test 同等的最大可能表达力 | 此前的 GNN 架构(GCN/GraphSAGE/GAT)大多凭经验设计,没人系统回答过"这些聚合函数在理论上能不能区分所有不同构的图" |
| 2021 | **Graphormer** | Ying et al.——把标准 Transformer 搬到图上:中心性编码 + 空间编码(最短路径距离)+ 边编码把图结构信息直接注入 attention,用全局注意力替代逐跳消息传递,OGB 大规模分子性质预测挑战赛冠军 | 消息传递框架下的 GNN(GCN→GIN)有过平滑(over-smoothing)、长程依赖建模弱等问题——堆叠层数太多所有节点表征趋同,而 Transformer 已经在 NLP/CV 证明了全局注意力建模长程依赖的能力,GNN 领域还没人验证过这条路 |

## 依赖与延伸

- 前置依赖:[Transformer](../05-transformer/01-transformer.md)(Graphormer 直接复用的架构)、[ViT](../08-vit/01-vit.md)(同样是"把 Transformer 搬到新模态"的思路呼应)
- 延伸方向:GNN 在推荐系统(如 PinSAGE,GraphSAGE 的工业化版本)、知识图谱补全、药物分子性质预测(如 Graphormer 在 OGB 分子性质预测挑战赛的应用)等场景有广泛的工业落地,这些方向本仓库暂未单独收录
```

- [ ] **Step 2: 验证跨链接路径存在**

```bash
ls /Users/lauzanhing/Desktop/Daily-LLM/05-transformer/01-transformer.md /Users/lauzanhing/Desktop/Daily-LLM/08-vit/01-vit.md
```

Expected: 两个文件都存在,不报错。

- [ ] **Step 3: 提交**

```bash
git add 17-graph-neural-networks/README.md
git commit -m "feat: 图神经网络(GNN)家族 README"
```

---

## Task 3: 节点 01 —— GCN(2017)

**Files:**
- Create: `17-graph-neural-networks/01-gcn.md`
- Create: `17-graph-neural-networks/assets/01-gcn-architecture.svg`(至少 1 张,**文件名必须以 `01-gcn-` 开头**)
- Modify: `web/src/components/home/familyHero.ts`

Frontmatter:

```yaml
---
name: "GCN"
year: 2017
family: "17-graph-neural-networks"
order: 1
paper: "Semi-Supervised Classification with Graph Convolutional Networks"
authors: ["Thomas N. Kipf", "Max Welling"]
key_idea: "把谱图卷积(Chebyshev 多项式近似图拉普拉斯)简化到一阶邻域聚合,一层 D̃^(-1/2) Ã D̃^(-1/2) H W 传播规则定义了'现代 GNN'这个范式的起点,在引文网络半监督节点分类上大幅超过此前基于图的方法"
---
```

九段式结构写作要点(**先用 WebSearch 核实真实数字再落笔**;若 WebSearch 不可用,用你的训练知识里最有把握的说法,并在文档内加编辑备注,遵循上一轮 `16-world-models` 家族已经建立的备注写法惯例——参照 `16-world-models/03-dreamerv3.md` 的性能数据段落写法):

- **前作进展**:2016 年之前,图上的深度学习主要有两条路:一是谱方法(spectral methods),把图信号变换到图拉普拉斯的特征空间做"卷积",但要么需要显式特征分解(计算量随节点数三次方增长,无法扩展到大图),要么像 Defferrard et al. 2016 的 ChebNet 用 K 阶 Chebyshev 多项式近似避免特征分解,但仍需要多阶邻域信息、参数量较大;二是非谱方法(如早期的 Graph Neural Network,Scarselli et al. 2009),用递归神经网络在图上传播信息,但训练不稳定、难以扩展。GCN 的洞察是把 ChebNet 的多项式近似进一步简化到一阶(K=1),用一个线性传播规则替代复杂的多阶展开。
- **核心思想 + 直觉**:核心洞察是——如果只保留 Chebyshev 多项式近似的最低阶项(一阶邻域),整个谱图卷积可以简化成一个非常简单的传播规则:每个节点的新表征 = 自己和直接邻居的表征做加权平均(权重由度数决定),再过一个线性变换和非线性激活。这个简化牺牲了"看多阶邻居"的能力,但换来了极大的计算效率——可以堆叠多层来间接扩大感受野(每层看一阶邻居,K 层就能看到 K 阶邻居),这和 CNN 堆叠卷积层扩大感受野的思路是类似的。
- **机制一(重整化技巧:Ã = A + I)**:直接用邻接矩阵 A 做聚合会丢失节点自己的信息(A 对角线是 0,自己不算自己的邻居),GCN 给邻接矩阵加自环(Ã = A + I,I 是单位矩阵),让每个节点在聚合时也把自己算进去。
- **机制二(对称归一化:D̃^(-1/2) Ã D̃^(-1/2))**:直接用 Ã 做聚合会让度数大的节点(邻居多)表征被过度放大,GCN 用度数矩阵 D̃ 做对称归一化,让聚合操作在数值上稳定,不会因为节点度数差异悬殊而导致某些节点表征爆炸或消失。
- **机制三(逐层传播规则:H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l)))**:把前两个机制组合成一个简洁的逐层传播公式,每层做一次"聚合 + 线性变换 + 非线性激活",堆叠多层实现多阶邻域信息传播,整个网络可以端到端用反向传播训练(半监督节点分类任务上,只需要一部分节点有标签,loss 只在有标签节点上算,梯度通过图结构传播到所有节点)。
- **三件套协同**:只有自环(机制一)没有归一化(机制二),节点表征会因度数差异被放大或缩小,数值不稳定,训练容易发散;只有归一化没有自环,节点看不到自己的信息,退化成纯粹的邻居平均;只有传播规则(机制三)没有前两者,无法定义出这个简洁的线性传播公式。三者组合起来才是 GCN 那个广为人知的传播规则。
- **关键代码**:一个 GCN 层的简化 PyTorch 伪代码(邻接矩阵归一化 + 线性变换 + 激活),参照 `13-moe-efficient/04-deepseek-v3.md` 的"关键代码"一节的详略程度,不需要完整可运行。
- **性能数据**:在 Cora / Citeseer / Pubmed 三个引文网络数据集上的半监督节点分类准确率,与此前方法(如 DeepWalk、Planetoid、ChebNet)的对比,核实真实数字后填写(不确定就写方向性描述加编辑备注)。
- **影响 / 后续**:GCN 是"现代 GNN"研究爆发的起点,直接启发了 GraphSAGE(解决 GCN 的直推式限制)和 GAT(解决 GCN 固定权重的限制)。加跨节点链接 `→ [02-graphsage.md](02-graphsage.md) · 解决 GCN 直推式训练、无法泛化到新节点的限制`。

## Context

**Important additional fix needed in this task** (与上一轮 16-world-models 家族 Task 1→Task 3 的处理方式完全一致):Task 1 widened the `FamilyId` TypeScript union type, which will break `web/src/components/home/familyHero.ts` — it has an exhaustive `Record<FamilyId, string>` map (`FAMILY_HERO`) picking one representative node per family. Add this line to the `FAMILY_HERO` object (follow the existing pattern in the file):

```typescript
  "17-graph-neural-networks": "17-graph-neural-networks/01-gcn.md",
```

- [ ] **Step 1: 写节点正文,创建配图,验证 SVG 合法性**

```bash
python3 -c "
import xml.etree.ElementTree as ET
ET.parse('17-graph-neural-networks/assets/01-gcn-architecture.svg')
print('OK')
"
```

- [ ] **Step 2: 修复 familyHero.ts,验证 tsc 干净**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web && npx tsc --noEmit
```

Expected: 无输出(clean exit)。

- [ ] **Step 3: 提交**

```bash
git add 17-graph-neural-networks/01-gcn.md 17-graph-neural-networks/assets/01-gcn-architecture.svg web/src/components/home/familyHero.ts
git commit -m "feat: GCN(2017)节点正文

补上 Task 1 遗留的 familyHero.ts FAMILY_HERO 缺口(第 17 个家族现在
有了第一个节点,可以选它作为家族卡片的代表作品)。"
```

---

## Task 4: 节点 02 —— GraphSAGE(2017)

**Files:**
- Create: `17-graph-neural-networks/02-graphsage.md`
- Create: `17-graph-neural-networks/assets/02-graphsage-architecture.svg`(至少 1 张,**文件名必须以 `02-graphsage-` 开头**)

Frontmatter:

```yaml
---
name: "GraphSAGE"
year: 2017
family: "17-graph-neural-networks"
order: 2
paper: "Inductive Representation Learning on Large Graphs"
authors: ["William L. Hamilton", "Rex Ying", "Jure Leskovec"]
key_idea: "SAmple + aggreGatE:固定大小邻域采样 + 可学习聚合函数(mean/LSTM/pooling),让 GNN 第一次能泛化到训练时没见过的节点/图(归纳式,而非 GCN 的直推式)"
---
```

九段式结构写作要点(**先用 WebSearch 核实真实数字再落笔**;若不可用,遵循已建立的编辑备注惯例):

- **前作进展**:GCN(2017)证明了简化的谱图卷积可行,但有一个根本限制——它是**直推式(transductive)**的:训练时需要完整的图邻接矩阵参与矩阵运算,模型学到的是"这个特定图里每个特定节点"的表征,没法直接用在训练时没见过的新节点上(比如社交网络新注册的用户、蛋白质相互作用网络里新发现的蛋白质)。要处理这种新节点,GCN 需要把新节点加入图后重新训练整个模型,在大规模、持续增长的真实图上不现实(如 Reddit 帖子流每天都有新内容、新用户)。
- **核心思想 + 直觉**:核心洞察是——不要为每个节点学一个固定的嵌入向量,而是学一个**通用的聚合函数**,这个函数接收"任意节点的邻居特征"作为输入,输出这个节点的表征。只要这个聚合函数学得足够好,即使是训练时完全没见过的新节点,只要知道它的邻居是谁、邻居的原始特征是什么,就能用同一个聚合函数算出它的表征——这就是"归纳式(inductive)"学习的关键。
- **机制一(邻域采样)**:真实大图里,一个节点可能有成千上万个邻居(如社交网络的大 V),如果每次聚合都用上全部邻居,计算量和内存都扛不住。GraphSAGE 对每个节点的邻居做固定大小的随机采样(比如第一层采样 25 个邻居,第二层再采样每个邻居的 10 个邻居),把不规则、大小不一的邻域统一成固定大小,方便批量化训练。
- **机制二(可学习聚合函数:mean / LSTM / pooling)**:GraphSAGE 提出了三种聚合函数供选择——简单的邻居特征均值池化(mean aggregator)、把邻居序列(随机打乱顺序)喂给 LSTM 取最后隐状态(LSTM aggregator,能力更强但需要人为定义一个不存在的顺序)、或者对每个邻居先过一个全连接层再做逐元素最大池化(pooling aggregator)。这些函数的参数是学出来的,不依赖某个特定图的固定结构。
- **机制三(逐层采样-聚合-拼接,K 层堆叠)**:和 GCN 一样,GraphSAGE 堆叠多层来扩大感受野,但每一层做的是"采样固定数量邻居 → 用聚合函数汇总邻居特征 → 和节点自己上一层的表征拼接(concat)→ 过线性层 + 激活"。拼接(而非像 GCN 那样直接加权平均)保留了"我自己的信息"和"邻居传来的信息"的区分,不会混在一起。
- **三件套协同**:只有采样没有可学习聚合函数,退化成对固定数量邻居做简单平均,表达力不够;只有聚合函数没有采样,大图上算不动;只有前两者没有逐层拼接结构,无法堆叠多层来扩大感受野、也丢失了区分"自身信息"与"邻居信息"的能力。三者组合起来才让 GraphSAGE 既能处理大规模图、又能泛化到新节点。
- **关键代码**:邻域采样 + mean aggregator + 拼接更新的简化伪代码,参照 `13-moe-efficient/04-deepseek-v3.md` 的详略程度。
- **性能数据**:核实真实数字——在引文网络(直推式对比)、Reddit 帖子分类(大规模)、蛋白质相互作用网络 PPI(归纳式,多图场景)三个数据集上相对 GCN 等基线的提升幅度。
- **影响 / 后续**:GraphSAGE 的归纳式框架成为工业界大规模图神经网络应用的基础(如 Pinterest 的 PinSAGE 推荐系统),证明了 GNN 可以脱离"训练时见过完整图"这一限制,在持续增长的真实世界大图上落地。加跨节点链接 `→ [01-gcn.md](01-gcn.md) · 本文解决的直推式训练限制`、`→ [03-gat.md](03-gat.md) · 用可学习注意力替代本文里手工设计的聚合函数选择`。

- [ ] **Step 1: 写节点正文,创建配图,验证 SVG 合法性**(命令同上,路径替换成本任务对应文件)

- [ ] **Step 2: 检查三个已知坑**(加粗定界符、`$` 转义、跨节点链接语法)

- [ ] **Step 3: 提交**

```bash
git add 17-graph-neural-networks/02-graphsage.md 17-graph-neural-networks/assets/02-graphsage-architecture.svg
git commit -m "feat: GraphSAGE(2017)节点正文"
```

---

## Task 5: 节点 03 —— GAT(2018)

**Files:**
- Create: `17-graph-neural-networks/03-gat.md`
- Create: `17-graph-neural-networks/assets/03-gat-architecture.svg`(至少 1 张,**文件名必须以 `03-gat-` 开头**)

Frontmatter:

```yaml
---
name: "GAT"
year: 2018
family: "17-graph-neural-networks"
order: 3
paper: "Graph Attention Networks"
authors: ["Petar Veličković", "Guillem Cucurull", "Arantxa Casanova", "Adriana Romero", "Pietro Liò", "Yoshua Bengio"]
key_idea: "用可学习的 attention 权重替代 GCN 里固定的度数归一化系数,让模型隐式学会'哪个邻居更重要',不需要提前知道完整图结构做矩阵运算"
---
```

九段式结构写作要点(**先用 WebSearch 核实真实数字再落笔**;若不可用,遵循已建立的编辑备注惯例):

- **前作进展**:GCN 用节点度数决定聚合权重(度数越大的邻居贡献权重相对越小,纯粹基于图的拓扑结构),GraphSAGE 的 mean/pooling 聚合函数也没有区分"哪个邻居更重要"——所有邻居(或采样到的邻居)在聚合时地位相同或仅由结构决定,模型没法根据节点的实际内容/特征来判断某个邻居是否真的携带更多有用信息。
- **核心思想 + 直觉**:核心洞察是——借鉴 Transformer 里 self-attention 的思路,让模型自己学习"给每个邻居分配多少权重",而不是用固定的图结构统计量(度数)决定。具体做法是:对每一对相连的节点,用一个共享的小型注意力机制(参数量很小,一个单层前馈网络)计算出一个注意力分数,再对一个节点的所有邻居的分数做 softmax 归一化,得到最终的聚合权重。这个注意力机制的参数是端到端学出来的,同一套参数应用在图里的每一条边上。
- **机制一(自注意力系数计算)**:对每条边 (i, j),把两个节点的特征拼接后过一个共享的单层前馈网络(权重向量 a)加 LeakyReLU 非线性,得到一个未归一化的注意力分数 e_ij,衡量"节点 j 对节点 i 有多重要"。
- **机制二(Softmax 归一化 + 加权聚合)**:对一个节点 i 的所有邻居 j,把 e_ij 在邻居范围内做 softmax 归一化(而不是全图),得到归一化的注意力系数 α_ij,再用这些系数对邻居特征做加权求和,得到节点 i 的新表征。这一步只需要局部邻居的信息,不需要知道全图结构或做矩阵求逆,是"masked attention"(只在图的边上计算注意力,而不是像标准 Transformer 那样对所有节点两两计算)。
- **机制三(多头注意力)**:和 Transformer 一样,GAT 用多个独立的注意力头分别计算,每个头学到不同的"重要性"判断标准,最后把多个头的输出拼接(中间层)或取平均(最后一层),提升训练稳定性和表达力。
- **三件套协同**:只有注意力系数计算没有 softmax 归一化,权重不构成一个合理的加权平均(可能为负或总和不为 1);只有归一化聚合没有多头注意力,单一注意力机制容易学到有偏或不稳定的重要性判断;只有多头没有前两者,无从谈起"注意力"这件事本身。三者组合起来才是 GAT 能稳定训练、又比 GCN/GraphSAGE 更灵活地判断邻居重要性的完整机制。
- **关键代码**:masked self-attention + 多头聚合的简化伪代码,参照 `13-moe-efficient/04-deepseek-v3.md` 的详略程度。
- **性能数据**:核实真实数字——在 Cora/Citeseer/Pubmed(直推式)和 PPI(归纳式)上相对 GCN、GraphSAGE 的准确率提升。
- **影响 / 后续**:GAT 的注意力机制成为后续大量 GNN 变种的标准组件,也是"把 Transformer 的核心思想(注意力)迁移到非序列结构数据"的早期成功案例之一,为 Graphormer(2021)彻底抛弃消息传递、全用 Transformer 埋下伏笔。加跨节点链接 `→ [02-graphsage.md](02-graphsage.md) · 本文替代的手工聚合函数设计`、`→ [05-graphormer.md](05-graphormer.md) · 注意力机制在图数据上的思路延伸到全局自注意力`。

- [ ] **Step 1: 写节点正文,创建配图,验证 SVG 合法性**

- [ ] **Step 2: 检查三个已知坑**

- [ ] **Step 3: 提交**

```bash
git add 17-graph-neural-networks/03-gat.md 17-graph-neural-networks/assets/03-gat-architecture.svg
git commit -m "feat: GAT(2018)节点正文"
```

---

## Task 6: 节点 04 —— GIN(2019)

**Files:**
- Create: `17-graph-neural-networks/04-gin.md`
- Create: `17-graph-neural-networks/assets/04-gin-architecture.svg`(至少 1 张,**文件名必须以 `04-gin-` 开头**)

Frontmatter:

```yaml
---
name: "GIN"
year: 2019
family: "17-graph-neural-networks"
order: 4
paper: "How Powerful are Graph Neural Networks?"
authors: ["Keyulu Xu", "Weihua Hu", "Jure Leskovec", "Stefanie Jegelka"]
key_idea: "用 Weisfeiler-Lehman 图同构测试给 GNN 表达力定理上界:证明 mean/max 聚合(如 GraphSAGE)不如 WL test,提出 sum 聚合 + MLP 的 GIN,理论上证明达到 WL test 同等的最大可能表达力"
---
```

九段式结构写作要点(**先用 WebSearch 核实真实数字再落笔**;若不可用,遵循已建立的编辑备注惯例):

- **前作进展**:GCN、GraphSAGE、GAT 都在实践上取得了成功,但都是"凭经验设计"的架构——没人系统回答过一个根本问题:**这些 GNN 到底有多强?能不能区分任意两个不同构的图(或子图结构)?** 图论里早就有一个经典的、简单但强大的图同构检测算法——Weisfeiler-Lehman(WL)测试,通过反复聚合邻居的"颜色标签"来判断两个图是否可能同构。Xu 等人的问题是:GNN 的消息传递机制和 WL test 有什么关系?
- **核心思想 + 直觉**:核心洞察是——**GNN 的消息传递过程和 WL test 的邻居标签聚合过程在数学结构上是同一件事**,因此 GNN 的表达力天花板就是 WL test 的表达力天花板;但要真正达到这个天花板,聚合函数和更新函数必须是**单射(injective)**的——也就是说,两个不同的邻居多重集合(multiset,同一个特征可能出现多次)必须被映射到两个不同的聚合结果,不能因为聚合函数本身"损失信息"而把明明不同的邻居结构映射成相同的输出。
- **机制一(为什么 mean/max 聚合不是单射的)**:论文证明,均值聚合(GraphSAGE mean aggregator、GCN 的加权平均)无法区分"邻居特征分布相同但数量不同"的情况(比如一个节点有 2 个特征都是 [1,0] 的邻居,另一个节点有 1 个特征是 [1,0] 的邻居,均值聚合结果相同);最大值聚合(GraphSAGE pooling aggregator)则完全丢失了"有多少个邻居具有某个特征"的信息,只保留"是否存在"。这两种聚合方式都不是单射的,因此严格弱于 WL test。
- **机制二(GIN 的求和聚合 + MLP)**:论文证明,在邻居特征是可数集合(离散或可数无穷)的前提下,存在一个函数能把任意多重集合单射地映射到一个向量,这个函数可以被参数化为"对多重集合内所有元素求和,再过一个多层感知机(MLP,而不是单层线性变换,因为单层线性变换不足以逼近任意的单射函数)"。GIN 的核心更新公式是:h_v^(k) = MLP^(k)((1+ε^(k))·h_v^(k-1) + Σ_{u∈N(v)} h_u^(k-1)),其中 ε 是一个可学习(或固定)的标量,用来区分节点自己的表征和邻居聚合的表征。
- **机制三(图级别读出函数:多层拼接而非只用最后一层)**:对于图分类任务(需要把整张图的所有节点表征汇总成一个图级别的向量),GIN 论文还证明:只用最后一层的节点表征做全图求和(readout)会丢失浅层结构信息,更好的做法是把每一层的图级别表征都求和后拼接起来(而不只是用最深一层),因为不同层能捕捉到不同"跳数"范围的子结构信息,浅层信息(局部结构)对某些任务同样重要。
- **三件套协同**:只有求和聚合没有 MLP(用单层线性变换),达不到单射所需的函数复杂度,理论上限达不到;只有 MLP 没有求和聚合(比如用均值/最大值 + MLP),聚合这一步本身就已经损失了信息,后面接多复杂的 MLP 都补不回来;只有前两者没有多层拼接读出,图分类任务丢失浅层结构信息。三者组合起来,GIN 才被证明是消息传递框架下理论上表达力最强的架构(等价于 WL test)。
- **关键代码**:GIN 层的求和聚合 + MLP 更新 + 多层拼接读出的简化伪代码,参照 `13-moe-efficient/04-deepseek-v3.md` 的详略程度。
- **性能数据**:核实真实数字——在图分类基准(生物信息学数据集如 MUTAG/PROTEINS,社交网络数据集如 IMDB-BINARY 等)上相对 GraphSAGE、其他基线的准确率提升,以及论文里报告的"训练集准确率能否拟合到 100%"这类验证表达力上限的实验结果。
- **影响 / 后续**:GIN 的理论分析确立了消息传递 GNN 表达力的天花板(WL test),后续大量工作要么试图突破这个天花板(用更高阶的 WL test 变种,超出本家族范围),要么像 Graphormer 一样干脆放弃消息传递框架、改用全局注意力。加跨节点链接 `→ [03-gat.md](03-gat.md) · GIN 的理论分析同样适用于分析本文的聚合方式`、`→ [05-graphormer.md](05-graphormer.md) · 跳出消息传递表达力上限的另一条路径`。

- [ ] **Step 1: 写节点正文,创建配图,验证 SVG 合法性**

- [ ] **Step 2: 检查三个已知坑**

- [ ] **Step 3: 提交**

```bash
git add 17-graph-neural-networks/04-gin.md 17-graph-neural-networks/assets/04-gin-architecture.svg
git commit -m "feat: GIN(2019)节点正文"
```

---

## Task 7: 节点 05 —— Graphormer(2021)

**Files:**
- Create: `17-graph-neural-networks/05-graphormer.md`
- Create: `17-graph-neural-networks/assets/05-graphormer-architecture.svg`(至少 1 张,**文件名必须以 `05-graphormer-` 开头**)

Frontmatter:

```yaml
---
name: "Graphormer"
year: 2021
family: "17-graph-neural-networks"
order: 5
paper: "Do Transformers Really Perform Bad for Graph Representation?"
authors: ["Chengxuan Ying", "Tianle Cai", "Shengjie Luo", "Shuxin Zheng", "Guolin Ke", "Di He", "Yanming Shen", "Tie-Yan Liu"]
key_idea: "把标准 Transformer 搬到图上:中心性编码 + 空间编码(最短路径距离)+ 边编码把图结构信息直接注入 attention,用全局注意力替代逐跳消息传递,OGB 大规模分子性质预测挑战赛冠军"
---
```

九段式结构写作要点(**先用 WebSearch 核实真实数字再落笔**;若不可用,遵循已建立的编辑备注惯例。**写"前作进展"section 提到 GIN 时,必须重新读一遍 `17-graph-neural-networks/04-gin.md` 自己的正文,确认 claim 与 GIN 节点自述内容一致,这是本家族和上一轮家族反复强调的规则**):

- **前作进展**:GCN → GraphSAGE → GAT → GIN 这条消息传递(message passing)路线,尽管理论(GIN)和实践(GAT 的注意力)都在不断改进,但消息传递框架本身有结构性局限:每一层只能让信息传播一跳(一个节点只能"看到"直接邻居),要建模长程依赖(比如图上相距很远的两个节点之间的关系)需要堆叠很多层,而堆叠层数太多会导致"过平滑"(over-smoothing)问题——所有节点的表征逐渐趋同,失去区分度。与此同时,标准 Transformer 的全局自注意力天然不受"一跳"限制,任意两个 token 之间可以直接建立联系,已经在 NLP 和视觉(呼应 [ViT](../08-vit/01-vit.md))领域证明了这一优势,但此前没有工作系统地把标准 Transformer 应用到图数据上并证明其有效性——图数据不像文本/图像那样有天然的序列/网格顺序,Transformer 的注意力机制本身也不感知图的拓扑结构。
- **核心思想 + 直觉**:核心洞察是——不需要为图数据设计全新的架构,标准 Transformer 的自注意力机制本身就足够强大,真正缺的是**把图的结构信息(节点度数、节点对之间的距离、边上的特征)编码进 attention 的计算过程**,让模型在做全局注意力时能"知道"图的拓扑,而不是把图当成一个无结构的节点集合。
- **机制一(中心性编码,Centrality Encoding)**:在标准 Transformer 里,每个 token 的重要性完全由内容和上下文决定;但在图里,节点的度数(连接了多少条边)本身就是一个重要的结构信号(度数高的节点往往是图里的"枢纽")。Graphormer 给每个节点的输入表征加上一个可学习的、按节点度数索引的嵌入向量(度数中心性编码),让模型在一开始就能感知节点在图里的"重要程度"。
- **机制二(空间编码,Spatial Encoding)**:核心创新——在计算注意力分数时,不只依赖两个节点的特征相似度,还给每一对节点 (i, j) 加上一个偏置项,这个偏置项由 i 和 j 之间的**最短路径距离**决定(距离越远,偏置越倾向于降低注意力权重),这个偏置是可学习的(按距离分桶,每个桶一个可学习标量),让模型在全局注意力的同时,仍然保留"图上距离近的节点更相关"这一归纳偏置。
- **机制三(边编码,Edge Encoding)**:如果图的边上还带有特征(比如分子图里化学键的类型),Graphormer 沿着两节点间最短路径上的所有边,把边特征也编码进注意力偏置项里,让边的信息也能影响全局注意力的计算,而不只是节点特征。
- **三件套协同**:只有中心性编码没有空间/边编码,模型知道"这个节点有多重要"但不知道"两个节点之间的图上关系",退化成普通 Transformer 加一点节点属性;只有空间编码没有中心性编码,模型知道节点间距离但不知道每个节点自身的结构重要性;只有前两者没有边编码,分子图这类边携带关键信息(化学键类型)的场景会丢失信息。三者组合起来,标准 Transformer 才能在完全保留全局注意力优势的同时,充分利用图的拓扑结构信息。
- **关键代码**:中心性编码 + 空间编码偏置矩阵计算 + 标准多头注意力(加偏置)的简化伪代码,参照 `../10-diffusion/05-dit.md` 或 `13-moe-efficient/04-deepseek-v3.md` 的详略程度。
- **性能数据**:核实真实数字——在 OGB(Open Graph Benchmark)大规模分子性质预测挑战赛(如 PCQM4M-LSC)上的排名/分数,相对此前 GNN 基线(包括 GIN)的提升幅度。
- **影响 / 后续**:Graphormer 证明了"图数据不需要专门设计消息传递架构,标准 Transformer 加合适的结构编码就能表现优异",是"Transformer 统一多模态架构"这一更大叙事(呼应 ViT 统一视觉、Whisper/wav2vec 统一语音)在图数据上的又一例证,也让"GNN vs Transformer 谁更适合图数据"成为后续几年的热门研究方向。加跨节点链接 `→ [04-gin.md](04-gin.md) · 本文跳出的消息传递表达力框架`、`→ [../05-transformer/01-transformer.md](../05-transformer/01-transformer.md) · 本文直接复用的 Transformer 架构`。

- [ ] **Step 1: 重新读一遍 `17-graph-neural-networks/04-gin.md` 全文,确认"前作进展"里对 GIN 的描述与该节点自述内容一致**

- [ ] **Step 2: 写节点正文,创建配图,验证 SVG 合法性**

- [ ] **Step 3: 检查三个已知坑**

- [ ] **Step 4: 提交**

```bash
git add 17-graph-neural-networks/05-graphormer.md 17-graph-neural-networks/assets/05-graphormer-architecture.svg
git commit -m "feat: Graphormer(2021)节点正文"
```

---

## Task 8: 生成 TIMELINE.md / families.json + 全项目验证

**Files:**
- Modify (自动生成,不手写): `TIMELINE.md`
- Modify (自动生成,不手写): `web/src/data/families.json`

- [ ] **Step 1: 运行生成脚本**

```bash
python3 scripts/generate_timeline.py
```

Expected: 无报错退出,输出类似 `wrote .../TIMELINE.md (79 nodes)` 和 `wrote .../families.json (17 families, 79 nodes)`(74 + 5 = 79)。

- [ ] **Step 2: 检查 TIMELINE.md 里新家族的 5 行是否按年份正确插入**

```bash
grep -n "GCN\|GraphSAGE\|GAT\|GIN\|Graphormer" TIMELINE.md
```

Expected: 5 行都出现,`\`17-graph-neural-networks\`` 出现在对应行里(注意 GAT/GIN 这两个缩写可能和其他家族节点的普通文本有误匹配,人工过一遍确认是本家族的 5 行)。

- [ ] **Step 3: 检查 families.json 里新家族块,尤其每个节点 assets 数组非空**

```bash
python3 -c "
import json
data = json.load(open('web/src/data/families.json'))
fam = next((f for f in data['families'] if f['id'] == '17-graph-neural-networks'), None)
assert fam is not None, '17-graph-neural-networks 家族块缺失'
assert len(fam['nodes']) == 5, f'期望 5 个节点,实际 {len(fam[\"nodes\"])}'
assert fam['colorToken'] == '--family-17', f'colorToken 不对: {fam[\"colorToken\"]}'
for n in fam['nodes']:
    assert len(n['assets']) > 0, f'{n[\"name\"]} 的 assets 数组是空的!检查 SVG 文件名是否匹配 {n[\"path\"]} 的 stem'
print('OK', fam['label'], fam['yearRange'])
for n in fam['nodes']:
    print(' -', n['order'], n['year'], n['name'], n['assets'])
"
```

Expected: 打印 `OK 图神经网络(GNN) [2017, 2021]`,随后 5 行,每行 `assets` 数组都非空。**如果某个节点 assets 为空,说明 SVG 文件名没有匹配上该节点 markdown 的 stem(见"已知的四个必须规避的坑"第 4 条),回去改文件名重新生成。**

- [ ] **Step 4: 全仓库 SVG 合法性自查(本家族范围)**

```bash
python3 -c "
import xml.etree.ElementTree as ET
import glob
files = sorted(glob.glob('17-graph-neural-networks/assets/*.svg'))
bad = []
for f in files:
    try:
        ET.parse(f)
    except Exception as e:
        bad.append((f, str(e)))
print(f'扫描 {len(files)} 个,{len(bad)} 个非法')
for f, e in bad:
    print(f'  {f}: {e}')
"
```

Expected: `扫描 5 个(或更多,若某节点配了 2 张图),0 个非法`

- [ ] **Step 5: 全项目 tsc + vitest**

```bash
cd web
npx tsc --noEmit
npx vitest run
```

Expected: tsc 无输出;vitest 全部通过,测试数量比之前(307)增加(新家族 5+ 个 SVG 会被 `web/src/test/svgAssets.test.ts` 扫描到,新增对应数量的测试用例)。

- [ ] **Step 6: 浏览器验证家族页面与节点详情页**

用 preview_start 起 dev server(用 `preview_logs` 确认实际绑定端口,工具报告的端口可能不准),访问 `/families/17-graph-neural-networks` 确认:
- 家族标题、"一句话定位"正文正常渲染
- 5 个节点卡片按年份显示,颜色为玫瑰红 `#f43f5e`
- 逐一点进 5 个节点详情页(`/families/17-graph-neural-networks/01-gcn` 等),确认走的是 `NodePage.tsx` 通用渲染路径(非金标本),标题/作者/正文/配图/前后节点导航正常显示,console 无 error
- 访问首页确认头部计数变成"17 家族 · 79 节点",侧栏家族列表和时间线可视化里都能看到"图神经网络(GNN)"及其 5 个节点

- [ ] **Step 7: 提交**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add TIMELINE.md web/src/data/families.json
git commit -m "feat: 生成 TIMELINE.md / families.json,第 17 个家族接入完成

图神经网络(GNN)家族全部 5 篇节点 markdown 正本 + README 就位,
python3 scripts/generate_timeline.py 重新生成产物,tsc + vitest 全项目
通过,浏览器验证家族页面与节点详情页渲染正常,首页计数更新为
17 家族 79 节点。金标本交互页按计划留待后续单独轮次补,本轮完成。"
```

---

## Self-Review 记录(写 plan 时已自查)

1. **Spec 覆盖**:spec 第 3 节(节点列表)→ Task 3-7;第 4 节(节点写作规范,含 SVG 命名规则)→ 每个节点 Task 的 Step 1 强调;第 5 节(家族 README,含手写子时间线表格要求)→ Task 2 的完整表格内容;第 6 节(注册文件,含 familyHero.ts 提前处理)→ Task 1 + Task 3;第 7 节(验收标准,含 assets 非空断言)→ Task 8 的检查脚本逐条对应;第 8 节(Out of scope)→ 全程未涉及金标本/foundations 改动。
2. **Placeholder 扫描**:每个节点任务给出的是"必须包含的真实事实清单 + 结构大纲"(内容创作类任务的等价物),不是"TBD"式占位符——这是延续 spec 第 4 节和上一轮(16-world-models)已验证有效的做法。
3. **一致性检查**:5 个节点的 frontmatter `order` 字段(1-5)与年份升序(2017/2017/2018/2019/2021)一致;跨节点链接指向的文件名(`01-gcn.md` `02-graphsage.md` `03-gat.md` `04-gin.md` `05-graphormer.md`)在各任务间保持一致拼写;所有跨节点链接示例均已采用 `[filename.md](filename.md)` 括号语法(吸取上一轮 Task 3 的教训,不再需要事后修复);familyHero.ts 的修复被前移到 Task 3(第一个节点任务)而非事后补丁,吸取上一轮的教训;家族 README 的"子时间线"是完整手写的 5 行真实表格,不是留空占位,吸取上一轮 Task 2 的教训。
