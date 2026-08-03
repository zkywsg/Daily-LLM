# 第 19 个家族(推荐系统 Recommendation)markdown 正本 Implementation Plan

**Goal:** 新建 `19-recommendation/` 家族目录,写出 1 篇家族 README + 5 篇节点 markdown 正本(Wide & Deep 2016 → YouTube DNN 2016 → DeepFM 2017 → DIN 2018 → PinSAGE 2018,教学顺序,见下方"排序说明"),接入 `generate_timeline.py` / `FamilyId` / `tokens.css` 三处注册点,不做金标本交互页。

**Architecture:** 复用仓库已收敛 18 个家族的写作模板(frontmatter + 前作进展/核心思想+直觉/机制一二三/三件套协同/关键代码/性能数据/影响后续 九段式,**扁平 `##` 标题,不嵌套**,与 17/18 家族一致),每篇配 1-2 张手绘风格 SVG。吸取前四轮沉淀的全部教训:SVG 命名必须匹配节点 markdown 完整 stem、家族 README 子时间线必须手写真实表格、familyHero.ts 缺口在第一个节点任务里就处理掉、跨节点链接用 markdown 语法、跨节点事实引用前重读对方原文。

**Tech Stack:** 纯 markdown + SVG,Python 脚本 `scripts/generate_timeline.py` 生成 TIMELINE.md/families.json,TypeScript `FamilyId` 类型,CSS custom property。

---

## 参考:设计文档

本 plan 的所有决策依据 `docs/superpowers/specs/2026-08-02-recommendation-family-design.md`,写节点前建议先读一遍该文件确认章节结构约定与排序说明。

## 参考:写作模板锚点文件

写正文前先读这几个近期节点作为结构范例(不要照抄措辞,只借鉴章节骨架和"三件套协同"收尾的写法):
- `18-speech-audio/01-wav2vec2.md`、`18-speech-audio/05-musicgen.md` —— 最近一轮已验证过的完整范例,扁平 `##` 标题结构
- `18-speech-audio/README.md` —— 家族 README 结构范例,包括"子时间线"手写表格的真实格式
- `13-moe-efficient/04-deepseek-v3.md` —— "关键代码"一节的详略程度参考
- `17-graph-neural-networks/02-graphsage.md` —— PinSAGE 节点"前作进展"必须重读的直接前作

## 已知的五个必须规避的坑(前四轮反复踩过)

1. **CommonMark 加粗定界符边界情况**:`**` 紧贴标点(引号/问号/括号)时,另一侧必须是空白或标点才能正确解析。写完每篇后人工过一遍 `**` 前后字符。
2. **`$` 货币符号与 remark-math 冲突**:涉及美元数字一律转义成 `\$`。本家族提到"千亿级样本""数十亿参数""数十亿边"等规模数字时要注意,若提到具体金额同样处理。
3. **跨节点链接必须用 markdown link 语法,不能写纯文本**:必须写成 `→ [02-youtube-dnn.md](02-youtube-dnn.md) · ...`。
4. **SVG 资产文件名必须以节点 markdown 的完整 stem 开头**:例如节点 `01-wide-deep.md` 的配图必须命名为 `01-wide-deep-architecture.svg` 或 `01-wide-deep-xxx.svg`,不能用缩写。
5. **跨节点事实一致性**:写"前作进展"或"影响/后续"提到已写好的兄弟节点时(尤其 PinSAGE 提到 `17-graph-neural-networks/02-graphsage.md` 时),必须重新读一遍那个节点自己的正文,确认自己写的 claim 与对方自述内容一致,不能凭训练知识里的一般印象凭空归因。

---

## Task 1: 家族基础设施注册

**Files:**
- Modify: `scripts/generate_timeline.py`
- Modify: `web/src/types/family.ts`
- Modify: `web/src/styles/tokens.css`

- [ ] **Step 1: 在 `scripts/generate_timeline.py` 的 `FAMILY_IDS` 列表末尾追加新家族 id**

打开 `scripts/generate_timeline.py`,找到以 `"18-speech-audio",` 结尾的 `FAMILY_IDS` 列表,改成:

```python
FAMILY_IDS = [
    "01-cnn", "02-rnn-lstm", "03-word-embedding", "04-gan",
    "05-transformer", "06-bert-family", "07-gpt-scaling",
    "08-vit", "09-multimodal-clip", "10-diffusion",
    "11-peft-lora", "12-rlhf-alignment", "13-moe-efficient",
    "14-rag-agent", "15-reasoning-o1-r1", "16-world-models",
    "17-graph-neural-networks", "18-speech-audio", "19-recommendation",
]
```

- [ ] **Step 2: 在 `web/src/types/family.ts` 的 `FamilyId` 联合类型末尾追加新家族 id**

找到以 `| "18-speech-audio";` 结尾的联合类型定义,改成:

```typescript
  | "18-speech-audio"
  | "19-recommendation";
```

- [ ] **Step 3: 在 `web/src/styles/tokens.css` 新增家族色 token**

找到 `--family-18: #fb7185; /* Speech/Audio 浅玫瑰红 */` 这一行,在它之后新增一行:

```css
  --family-19: #f87171; /* Recommendation 红色 */
```

- [ ] **Step 4: 验证 tsc(预期会因 familyHero.ts 报一个已知错误,不用现在修)**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web && npx tsc --noEmit
```

Expected: 报一个 `web/src/components/home/familyHero.ts` 的 `TS2741` 错误(`Record<FamilyId, string>` 缺 `"19-recommendation"` key)。这是预期的、已知的中间状态——前四轮已验证过这个错误在追加节点(Task 3)时一并修复是正确的做法,不需要现在处理,继续往下走。

- [ ] **Step 5: 提交**

```bash
git add scripts/generate_timeline.py web/src/types/family.ts web/src/styles/tokens.css
git commit -m "feat: 第 19 个家族(推荐系统 Recommendation)基础设施注册

追加 FAMILY_IDS / FamilyId / --family-19 三处,markdown 正本在后续
任务里逐篇写。tsc 此时会报 familyHero.ts 的已知错误,留到 Task 3(第一个
节点)一并修复,与前四轮做法一致。"
```

---

## Task 2: 家族 README

**Files:**
- Create: `19-recommendation/README.md`

- [ ] **Step 1: 写家族 README**

创建 `19-recommendation/README.md`,章节结构复用 `18-speech-audio/README.md`:

```markdown
# 推荐系统(Recommendation Systems)

> **从"人工设计特征叉乘做记忆"到"用注意力动态建模用户兴趣",再到"把用户-物品关系直接建模成图做归纳式表征学习"——推荐系统这条主线全部由工业界驱动,每一步演化都直接对应生产环境里踩过的真实的坑。**

## 一句话定位

推荐系统要解决的核心问题是:给定一个用户和海量候选物品(商品/广告/视频/图片),预测用户对每个候选的偏好程度并排序。这条主线与本仓库其他家族最大的不同是——全部 5 篇论文都来自工业界(Google、Huawei、Alibaba、Pinterest),每一次架构演化都直接对应一个真实的生产环境痛点,而不是单纯的学术研究驱动。2016 年 **Wide & Deep** 首次系统性解决"记忆(memorization)与泛化(generalization)"的权衡——线性模型靠特征叉乘记住历史共现规律,深度网络靠 embedding 泛化到没见过的特征组合;同年 **YouTube DNN** 确立了"候选生成(从数亿物品里快速捞出几百个候选)+ 排序(精细排序这几百个候选)"两阶段漏斗架构,成为工业界大规模推荐系统的标准范式;2017 年 **DeepFM** 用因子分解机(FM)自动建模特征交互,不再需要 Wide & Deep 那样人工设计交叉特征;2018 年 **DIN** 用注意力机制让用户历史行为的权重根据当前候选广告动态变化,解决了"把用户兴趣压缩成一个定长向量"表达力不足的问题;同年 **PinSAGE** 把 [GraphSAGE](../17-graph-neural-networks/02-graphsage.md) 的归纳式图卷积扩展到 Pinterest 30 亿节点、180 亿边的工业级图规模,是 GNN 在推荐系统里最早的大规模工业落地。这家族要回答的问题是:**推荐系统是怎么从"人工特征工程 + 线性/浅层模型"演化到"端到端深度学习自动建模特征交互与用户兴趣"的**。

## 概念本身

### 记忆与泛化的权衡,贯穿前四篇的统一视角

Wide & Deep、YouTube DNN、DeepFM、DIN 这四篇尽管具体架构不同,但都在同一个核心矛盾上做文章:

- **记忆(memorization)**:模型能不能记住"用户 A 点击过和商品 X 强相关的商品 Y"这类具体的、稀疏的共现规律?线性模型 + 特征叉乘天然擅长这个,但需要人工设计交叉特征,组合数量爆炸,而且完全无法泛化到没出现过的组合。
- **泛化(generalization)**:模型能不能从"用户 A 喜欢的商品和商品 Z 有些相似的 embedding"推断出用户 A 可能也喜欢商品 Z,即使两者从未共现过?深度网络的 embedding + 非线性变换天然擅长这个,但可能"过度泛化",在数据稀疏区域推荐不相关的物品。

Wide & Deep 把两者显式拼接联合训练;DeepFM 用 FM 替代 Wide 部分的人工特征叉乘,让"低阶记忆"也变成自动学习;DIN 则更进一步,不满足于"用户兴趣是一个固定向量",而是让这个向量本身根据候选物品动态变化——本质上仍是在优化"如何更精确地记忆用户的具体兴趣,同时不丢失泛化能力"这个母题。YouTube DNN 是唯一一篇不直接讨论这个权衡的论文,它解决的是另一个正交问题:如何把"数亿候选物品"的检索问题工程化拆解成可以落地生产的两阶段漏斗。

### PinSAGE:跳出"表格特征"框架,改用图结构建模

前四篇论文全部把推荐问题建模成"用户特征 + 物品特征 + 上下文特征"的表格特征交互问题,PinSAGE 则完全跳出这个框架——直接把 Pinterest 的用户-图片-画板关系建模成一张二部图,用 [GraphSAGE](../17-graph-neural-networks/02-graphsage.md) 式的归纳式图卷积在图结构上直接学习物品的表征,不需要人工设计"用户历史行为统计特征"这类中间表格特征。这是"用图结构直接建模关系数据"这条思路(呼应 GNN 家族的核心叙事)在推荐系统里的工业级实践。

## 子时间线

| 年份 | 名字 | 关键贡献 | 之前卡在哪 |
|------|------|---------|-----------|
| 2016 | **Wide & Deep** | Cheng et al.(Google)——把线性模型(Wide,靠特征叉乘"记忆"共现规律)和深度神经网络(Deep,靠 embedding"泛化"到没见过的特征组合)联合训练成一个模型,首次系统性解决"记忆与泛化"的权衡问题,在 Google Play 应用商店上线并取得线上效果提升 | 纯线性模型(逻辑回归 + 人工交叉特征)能记住具体共现规律但完全无法泛化;纯深度模型(embedding+DNN)泛化能力强但在数据稀疏、特征交互复杂的场景容易"过度泛化"、推荐出不相关物品 |
| 2016 | **YouTube DNN** | Covington, Adams & Sargin(Google)——用"候选生成(从数亿视频里检索出几百个候选,建模成极端多分类问题)+ 排序(用更丰富特征精细排序这几百个候选)"两阶段深度神经网络架构,确立了工业界大规模推荐系统的标准漏斗范式 | 此前的推荐系统(矩阵分解等)难以融合视频元数据、用户上下文等异构特征,也难以在数亿规模的候选池上做实时高效检索 |
| 2017 | **DeepFM** | Guo et al.(Huawei)——用因子分解机(FM)替代 Wide & Deep 里需要人工设计的特征叉乘部分,FM 和 DNN 共享同一套特征 embedding 端到端训练,不再需要特征工程就能同时建模低阶和高阶特征交互 | Wide & Deep 的 Wide 部分仍然需要人工设计交叉特征(特征工程成本高、依赖领域知识),FM 本身虽然能自动建模二阶交互但缺乏高阶交互的建模能力 |
| 2018 | **DIN** | Zhou et al.(Alibaba)——用注意力机制让模型根据候选广告动态计算用户历史行为序列里每个行为的权重,解决了此前把用户兴趣压缩成单一定长向量、无法表达兴趣多样性的问题 | Wide & Deep、DeepFM 等模型把用户的全部历史行为通过求和/平均池化压缩成一个固定长度的向量,不管候选广告是什么,这个向量都不变,无法表达"用户对不同候选有不同兴趣侧重"这一现实 |
| 2018 | **PinSAGE** | Ying et al.(Pinterest/Stanford)——把 GraphSAGE 的归纳式图卷积扩展到 30 亿节点、180 亿边的工业级二部图(用户-物品图),用基于随机游走的重要性采样 + 生产者-消费者流水线训练,是 GNN 在推荐系统里最早的大规模工业落地 | GraphSAGE 验证了归纳式图卷积在中等规模图上可行,但没有解决在 Pinterest 这种数十亿节点/边规模的图上如何高效采样、训练、批量推理这些工程问题 |

## 依赖与延伸

- 前置依赖:[GraphSAGE](../17-graph-neural-networks/02-graphsage.md)(PinSAGE 直接扩展的归纳式图卷积架构)
- 延伸方向:DIEN/DSIN(DIN 的序列建模后续工作)、xDeepFM(DeepFM 的显式高阶交互后续工作)、双塔模型/大规模向量检索(YouTube DNN 候选生成阶段的工业化延伸),这些方向本仓库暂未单独收录
```

- [ ] **Step 2: 验证跨链接路径存在**

```bash
ls /Users/lauzanhing/Desktop/Daily-LLM/17-graph-neural-networks/02-graphsage.md
```

Expected: 文件存在,不报错。

- [ ] **Step 3: 提交**

```bash
git add 19-recommendation/README.md
git commit -m "feat: 推荐系统(Recommendation)家族 README"
```

---

## Task 3: 节点 01 —— Wide & Deep(2016)

**Files:**
- Create: `19-recommendation/01-wide-deep.md`
- Create: `19-recommendation/assets/01-wide-deep-architecture.svg`(至少 1 张,**文件名必须以 `01-wide-deep-` 开头**)
- Modify: `web/src/components/home/familyHero.ts`

Frontmatter:

```yaml
---
name: "Wide & Deep"
year: 2016
family: "19-recommendation"
order: 1
paper: "Wide & Deep Learning for Recommender Systems"
authors: ["Heng-Tze Cheng", "Levent Koc", "Jeremiah Harmsen", "Tal Shaked", "Tushar Chandra", "Hrishi Aradhye", "Glen Anderson", "Greg Corrado", "Wei Chai", "Mustafa Ispir", "Rohan Anil", "Zakaria Haque", "Lichan Hong", "Vihan Jain", "Xiaobing Liu", "Hemal Shah"]
key_idea: "把线性模型(Wide,靠特征叉乘'记忆'共现规律)和深度神经网络(Deep,靠 embedding'泛化'到没见过的特征组合)联合训练成一个模型,首次系统性解决'记忆与泛化'的权衡问题"
---
```

九段式结构写作要点(**先用 WebSearch 核实真实数字再落笔**;若 WebSearch 不可用,用你的训练知识里最有把握的说法,并在文档内加编辑备注,遵循前几轮已建立的备注写法惯例):

- **前作进展**:在 Wide & Deep 之前,推荐/排序系统主要靠逻辑回归这类线性模型 + 人工设计的交叉特征(cross-product feature)来做,比如把"用户安装过的 app 类型"和"当前曝光的 app 类型"做叉乘生成新特征。这种方法能很好地记住具体的、稀疏的共现规律("装过美颜类 app 的用户点击其他美颜类 app 的概率高"),但交叉特征需要领域专家手工设计,组合数量随特征数指数增长,而且对训练数据里从未出现过的特征组合完全没有泛化能力。另一条路是纯深度学习模型(embedding + DNN),能自动学到特征间的隐式高阶交互并泛化到新组合,但在用户-物品交互矩阵非常稀疏时,可能"过度泛化"、给用户推荐一些看似相关但实际不匹配的物品。
- **核心思想 + 直觉**:核心洞察是——与其在"记忆"和"泛化"之间二选一,不如把两种模型联合训练成一个整体:Wide 部分继续用线性模型 + 交叉特征处理"记忆",Deep 部分用 embedding + 前馈网络处理"泛化",两部分的输出在最后一层加权求和后过 sigmoid 联合优化,让模型同时具备两种能力。这个思路后来被证明是工业级推荐系统里"效果和可解释性都要兼顾"的一个通用范式。
- **机制一(Wide 组件:广义线性模型 + 特征叉乘)**:Wide 部分是一个 `y = w^T x + b` 形式的线性模型,输入 x 除了原始特征外,还包括人工设计的交叉积变换特征(cross-product transformation),例如"用户已安装 app=netflix AND 当前曝光 app=hulu"这种组合特征,用来显式记忆特定特征组合与目标的强相关性。
- **机制二(Deep 组件:embedding + 前馈神经网络)**:把类别型特征(如 app id、用户人口统计特征)映射成低维稠密 embedding 向量,拼接后输入多层前馈网络(ReLU 激活),让模型自动学习特征间的隐式、高阶非线性交互,具备对训练时未见过的特征组合的泛化能力。
- **机制三(联合训练:加权求和 + 端到端反向传播)**:把 Wide 部分和 Deep 部分的输出做加权求和,过 sigmoid 得到最终预测概率,整个模型用同一个 loss(如 logistic loss)端到端联合训练——注意这是"联合训练(joint training)"而不是"集成(ensemble)":两部分共享同一个训练信号,反向传播时 Wide 部分通常用 FTRL + L1 正则优化(鼓励稀疏),Deep 部分用 AdaGrad 优化,两种优化器同时更新各自负责的参数。
- **三件套协同**:只有 Wide 没有 Deep,退化成传统的线性模型 + 人工特征工程,泛化能力有限;只有 Deep 没有 Wide,丢失了显式记忆具体共现规律的能力,在稀疏场景下效果可能不如加了交叉特征的线性模型;只有前两者的架构没有"联合训练"这个机制(比如分别训练再简单加权),两部分学到的表示无法针对同一个目标协同优化。三者组合起来,Wide & Deep 才能在 Google Play 这种超大规模、特征稀疏的真实推荐场景里同时兼顾记忆和泛化。
- **关键代码**:Wide 线性层 + Deep embedding-MLP + 加权求和 sigmoid 输出的简化 PyTorch 风格伪代码,参照 `13-moe-efficient/04-deepseek-v3.md` 的"关键代码"一节的详略程度,不需要完整可运行。
- **性能数据**:核实真实数字——论文报告的 Google Play 应用商店线上 A/B 测试的应用获取率(acquisition rate)提升幅度,以及离线 AUC 相对纯 Wide 模型、纯 Deep 模型的对比。不确定的具体数字写方向性描述加编辑备注,参照 `16-world-models/03-dreamerv3.md` 的性能数据段落写法。
- **影响 / 后续**:Wide & Deep 确立了"线性记忆 + 深度泛化"联合训练这一至今仍被广泛使用的工业级推荐系统设计范式,直接影响了 DeepFM(用 FM 自动化 Wide 部分的特征交叉)等后续工作。加跨节点链接 `→ [03-deepfm.md](03-deepfm.md) · 用 FM 自动学习特征交叉,替代本文需要人工设计的交叉特征`。

## Context

**Important additional fix needed in this task**(与前四轮 Task 1→Task 3 的处理方式完全一致):Task 1 widened the `FamilyId` TypeScript union type, which will break `web/src/components/home/familyHero.ts` — it has an exhaustive `Record<FamilyId, string>` map (`FAMILY_HERO`) picking one representative node per family. Add this line to the `FAMILY_HERO` object (follow the existing pattern in the file):

```typescript
  "19-recommendation": "19-recommendation/01-wide-deep.md",
```

- [ ] **Step 1: 写节点正文,创建配图,验证 SVG 合法性**

```bash
python3 -c "
import xml.etree.ElementTree as ET
ET.parse('19-recommendation/assets/01-wide-deep-architecture.svg')
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
git add 19-recommendation/01-wide-deep.md 19-recommendation/assets/01-wide-deep-architecture.svg web/src/components/home/familyHero.ts
git commit -m "feat: Wide & Deep(2016)节点正文

补上 Task 1 遗留的 familyHero.ts FAMILY_HERO 缺口(第 19 个家族现在
有了第一个节点,可以选它作为家族卡片的代表作品)。"
```

---

## Task 4: 节点 02 —— YouTube DNN(2016)

**Files:**
- Create: `19-recommendation/02-youtube-dnn.md`
- Create: `19-recommendation/assets/02-youtube-dnn-architecture.svg`(至少 1 张,**文件名必须以 `02-youtube-dnn-` 开头**)

Frontmatter:

```yaml
---
name: "YouTube DNN"
year: 2016
family: "19-recommendation"
order: 2
paper: "Deep Neural Networks for YouTube Recommendations"
authors: ["Paul Covington", "Jay Adams", "Emre Sargin"]
key_idea: "用'候选生成(candidate generation)+ 排序(ranking)'两阶段深度神经网络架构处理数亿视频规模的推荐,候选生成阶段把推荐建模成极端多分类问题,是工业界大规模深度推荐系统的奠基性架构"
---
```

九段式结构写作要点(**先用 WebSearch 核实真实数字再落笔**;若不可用,遵循已建立的编辑备注惯例。**写"前作进展"提到 Wide & Deep 时,必须重新读一遍 `19-recommendation/01-wide-deep.md` 自己的正文,确认 claim 与该节点自述内容一致**):

- **前作进展**:Wide & Deep(2016)证明了联合训练线性模型和深度网络能同时兼顾记忆和泛化,但那篇论文关注的是"给定一小批候选,如何精确排序"这个问题;YouTube 面临的是完全不同量级的挑战——从数亿条视频里,毫秒级地为每个用户找出最相关的几百个候选,再精细排序。此前基于矩阵分解等经典协同过滤方法的推荐系统,难以融合视频的丰富元数据、用户观看历史、上下文特征等异构信息,也难以支撑这种规模的实时检索。
- **核心思想 + 直觉**:核心洞察是把推荐问题拆解成两个复杂度递减的阶段——候选生成阶段先用一个相对简单、高效的模型从数亿视频里快速筛出几百个粗粒度相关的候选(牺牲一些精度换取速度),排序阶段再用一个更复杂、特征更丰富的模型对这几百个候选做精细排序(此时候选集合已经很小,可以承受更高的计算成本)。这种"先粗筛后精排"的两阶段漏斗架构,后来成为几乎所有工业级推荐系统的标准范式。
- **机制一(候选生成:极端多分类 + embedding 近邻检索)**:把"预测用户接下来会观看哪个视频"建模成一个类别数等于视频总数(数百万级)的多分类问题,用户的观看历史、搜索历史等特征通过 embedding 后取平均得到用户向量,过几层全连接网络后与 softmax 权重矩阵(每个视频一行,相当于视频 embedding)做内积,取内积最大(即分类概率最高)的若干视频作为候选;工程上训练时用负采样近似 softmax 避免对数百万类别做归一化,线上服务阶段则把"预测最可能的视频"简化成在视频 embedding 空间里做最近邻检索,不需要重新跑一遍全网络。
- **机制二(排序:逻辑回归预测期望观看时长)**:候选生成阶段产出的几百个候选进入排序阶段,这里用更丰富的特征(视频本身特征、用户与视频的交互历史特征、上下文特征)训练一个逻辑回归模型,但优化目标不是简单的点击率,而是预测期望观看时长(用加权逻辑回归实现,正样本权重为观看时长),因为 YouTube 更关心用户观看时长而非单纯的点击。
- **机制三(工程细节:样本年龄特征与训练数据处理)**:视频的流行度会随时间迅速变化(新上传的热门视频),模型如果不感知这一点会系统性低估新视频的推荐概率;论文引入"样本年龄"(example age)作为一个显式特征,让模型能学习并补偿这种"越新的内容后续观看概率相对越高"的时间偏置。此外训练数据的构造(如何生成正负样本、如何处理观看历史序列)也是这篇论文里重要的工程细节。
- **三件套协同**:只有候选生成没有排序,几百个候选里相关性差异很大但没有精细排序,用户体验差;只有排序没有候选生成,无法应对数亿视频规模的实时检索,计算量无法承受;只有前两者没有样本年龄等工程细节,模型会系统性偏向旧内容、跟不上内容生态的动态变化。三者组合起来,YouTube DNN 才能在数亿视频、数十亿用户规模下做到实时、准确、且能跟上内容新鲜度的推荐。
- **关键代码**:候选生成阶段的 embedding 平均池化 + 多层网络 + 负采样 softmax,以及排序阶段加权逻辑回归的简化伪代码,参照 `13-moe-efficient/04-deepseek-v3.md` 的详略程度。
- **性能数据**:核实真实数字——论文报告的离线 holdout 数据集上的 mAP(mean average precision)提升,以及线上 A/B 测试的观看时长等指标变化(相对此前系统的提升幅度)。不确定的具体数字写方向性描述加编辑备注。
- **影响 / 后续**:YouTube DNN 确立的"候选生成 + 排序"两阶段漏斗架构成为此后几乎所有大规模工业推荐系统(电商、广告、社交)的标准范式,候选生成阶段"用户/物品 embedding 近邻检索"的思路也直接启发了后续的双塔模型等大规模向量召回架构。加跨节点链接 `→ [01-wide-deep.md](01-wide-deep.md) · 同年发布,聚焦排序问题的互补工作`、`→ [04-din.md](04-din.md) · 排序阶段用户历史行为建模的后续改进`。

- [ ] **Step 1: 写节点正文,创建配图,验证 SVG 合法性**(命令同上,路径替换成本任务对应文件)

- [ ] **Step 2: 检查五个已知坑**(加粗定界符、`$` 转义、跨节点链接语法、SVG 命名、跨节点事实一致性)

- [ ] **Step 3: 提交**

```bash
git add 19-recommendation/02-youtube-dnn.md 19-recommendation/assets/02-youtube-dnn-architecture.svg
git commit -m "feat: YouTube DNN(2016)节点正文"
```

---

## Task 5: 节点 03 —— DeepFM(2017)

**Files:**
- Create: `19-recommendation/03-deepfm.md`
- Create: `19-recommendation/assets/03-deepfm-architecture.svg`(至少 1 张,**文件名必须以 `03-deepfm-` 开头**)

Frontmatter:

```yaml
---
name: "DeepFM"
year: 2017
family: "19-recommendation"
order: 3
paper: "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"
authors: ["Huifeng Guo", "Ruiming Tang", "Yunming Ye", "Zhenguo Li", "Xiuqiang He"]
key_idea: "用因子分解机(FM)替代 Wide & Deep 里需要人工设计的特征叉乘部分,FM 和 DNN 共享同一套特征 embedding 端到端训练,不再需要特征工程就能同时建模低阶和高阶特征交互"
---
```

九段式结构写作要点(**先用 WebSearch 核实真实数字再落笔**;若不可用,遵循已建立的编辑备注惯例。**写"前作进展"提到 Wide & Deep 时,必须重新读一遍 `19-recommendation/01-wide-deep.md` 自己的正文确认一致**):

- **前作进展**:Wide & Deep 用联合训练的方式兼顾了记忆和泛化,但它的 Wide 部分仍然依赖人工设计的交叉特征(cross-product transformation)——这意味着工程师需要凭领域知识猜测"哪些特征组合值得交叉",既费人力又容易遗漏重要的交叉组合。同时期广泛使用的因子分解机(Factorization Machine, FM)能自动学习任意两个特征之间的二阶交互(通过隐向量内积),不需要人工设计,但 FM 本身只能建模到二阶交互,无法捕捉更复杂的高阶特征组合模式。
- **核心思想 + 直觉**:核心洞察是——把 Wide & Deep 里的"Wide 部分"直接替换成 FM,让二阶特征交互也变成自动学习而不是人工设计,同时让 FM 和 Deep(DNN)两部分共享同一套特征 embedding(而不是像 Wide & Deep 那样 Wide 和 Deep 使用不同的输入特征),这样模型可以同时、端到端地学习低阶(二阶,来自 FM)和高阶(来自 DNN 的隐式非线性组合)特征交互,完全不需要额外的特征工程。
- **机制一(FM 组件:一阶 + 二阶特征交互)**:FM 部分的输出由两部分组成——一阶线性项(各个特征自身的加权和)和二阶交互项(所有特征两两之间隐向量内积的加权和),二阶交互项通过隐向量内积隐式计算,不需要显式枚举所有特征对,计算复杂度是特征数的线性而非平方级别。
- **机制二(Deep 组件:共享 embedding 的高阶交互)**:把 FM 二阶交互项里用到的同一套特征隐向量(embedding)拼接起来,输入到一个多层前馈网络里,让 DNN 自动学习二阶以上的高阶非线性特征交互——关键是这套 embedding 与 FM 组件共享,而不是像 Wide & Deep 那样 Wide 和 Deep 分别用独立的特征表示。
- **机制三(端到端联合训练:FM 输出 + DNN 输出相加过 sigmoid)**:把 FM 部分的输出(一阶+二阶)和 Deep 部分的输出直接相加,过 sigmoid 得到最终 CTR 预测,整个模型(包括共享的 embedding 层)端到端用同一个 loss 联合训练,不需要像 Wide & Deep 那样对 Wide 和 Deep 两部分用不同的优化器分别处理。
- **三件套协同**:只有 FM 没有 Deep,退化成经典 FM,只能捕捉二阶交互,表达力有限;只有 Deep 没有 FM,失去了显式建模二阶交互的归纳偏置,DNN 需要更多数据和参数才能隐式学到类似的模式;只有前两者没有"共享 embedding",FM 和 Deep 各自学一套表示,既增加参数量又可能学到不一致甚至冲突的特征表示。三者组合起来,DeepFM 才能在完全不需要人工特征工程的前提下,同时精确建模低阶和高阶特征交互。
- **关键代码**:FM 二阶交互的隐向量内积计算 + 共享 embedding 输入 DNN + 两路输出相加的简化伪代码,参照 `13-moe-efficient/04-deepseek-v3.md` 的详略程度。
- **性能数据**:核实真实数字——论文在 Criteo 数据集、Company(华为内部)数据集上,DeepFM 相对 FM、Wide & Deep、纯 DNN 等基线的 AUC 和 Logloss 提升幅度。不确定的具体数字写方向性描述加编辑备注。
- **影响 / 后续**:DeepFM 成为工业界 CTR 预估任务里被广泛采用的基线模型之一,"自动特征交叉 + 共享 embedding"的设计思路后续启发了 xDeepFM(显式建模任意阶交互)等一系列工作,证明了"去除人工特征工程"是这条主线持续演进的方向。加跨节点链接 `→ [01-wide-deep.md](01-wide-deep.md) · 本文自动化替代的人工特征叉乘设计`、`→ [04-din.md](04-din.md) · 同样追求减少人工设计、让模型自动适应用户行为的后续工作`。

- [ ] **Step 1: 重新读一遍 `19-recommendation/01-wide-deep.md` 全文,确认"前作进展"里对 Wide & Deep 的描述与该节点自述内容一致**

- [ ] **Step 2: 写节点正文,创建配图,验证 SVG 合法性**

- [ ] **Step 3: 检查五个已知坑**

- [ ] **Step 4: 提交**

```bash
git add 19-recommendation/03-deepfm.md 19-recommendation/assets/03-deepfm-architecture.svg
git commit -m "feat: DeepFM(2017)节点正文"
```

---

## Task 6: 节点 04 —— DIN(2018)

**Files:**
- Create: `19-recommendation/04-din.md`
- Create: `19-recommendation/assets/04-din-architecture.svg`(至少 1 张,**文件名必须以 `04-din-` 开头**)

Frontmatter:

```yaml
---
name: "DIN"
year: 2018
family: "19-recommendation"
order: 4
paper: "Deep Interest Network for Click-Through Rate Prediction"
authors: ["Guorui Zhou", "Xiaoqiang Zhu", "Chenru Song", "Ying Fan", "Han Zhu", "Xiao Ma", "Yanghui Yan", "Junqi Jin", "Han Li", "Kun Gai"]
key_idea: "用注意力机制让模型根据候选广告动态计算用户历史行为序列里每个行为的权重,解决了此前把用户兴趣压缩成单一定长向量、无法表达兴趣多样性的问题"
---
```

九段式结构写作要点(**先用 WebSearch 核实真实数字再落笔**;若不可用,遵循已建立的编辑备注惯例。**写"前作进展"提到 Wide & Deep/DeepFM 时,必须重新读一遍两者各自的正文,确认 claim 一致**):

- **前作进展**:Wide & Deep、DeepFM 这类模型在处理用户历史行为序列(如用户最近点击/购买过的商品列表)时,通常的做法是把序列里每个商品的 embedding 做求和或平均池化,压缩成一个固定长度的"用户兴趣向量"。这个向量一旦算出来,不管接下来要预测用户对哪个候选广告的点击率,都是同一个向量——但现实中用户的兴趣是多样的:一个既买过运动鞋又买过婴儿用品的用户,当候选广告是运动装备时,历史里"运动鞋"相关的行为应该权重更高;当候选广告是婴儿用品时,"婴儿用品"相关的行为应该权重更高。固定池化的用户向量无法表达这种"兴趣的哪个侧面与当前候选相关"的动态性。
- **核心思想 + 直觉**:核心洞察是——借鉴注意力机制的思路,不要把用户历史行为一视同仁地池化成一个固定向量,而是针对当前要预测的候选广告,动态计算历史里每个行为与这个候选的相关性权重,再做加权求和。这样同一个用户面对不同候选广告时,得到的"兴趣表示"是不同的,能够聚焦在与当前候选真正相关的历史行为上。
- **机制一(局部激活单元,Local Activation Unit)**:对用户历史行为序列里的每一个行为(如某次点击的商品),把这个行为的 embedding 和候选广告的 embedding 一起输入一个小型前馈网络(局部激活单元),输出一个标量,代表这个历史行为与当前候选广告的相关性权重(注意力分数)。
- **机制二(加权求和池化替代固定池化)**:把局部激活单元算出的权重对每个历史行为的 embedding 做加权求和(而不是简单的求和/平均池化),得到这一次预测专用的、随候选广告变化的动态用户兴趣表示,再和候选广告 embedding、其他特征一起送入后续的全连接网络预测点击率。
- **机制三(训练稳定性的工程改进:Dice 激活函数与自适应正则化)**:论文还提出了两个工程改进——Dice 激活函数(根据每层输入数据的分布自适应调整激活函数的形态,替代固定形态的 PReLU)和 mini-batch 感知的自适应正则化(针对 CTR 预估里类别特征长尾分布的特点,对出现频率不同的特征采用不同强度的正则化,避免高频特征过拟合、低频特征欠拟合)。
- **三件套协同**:只有局部激活单元没有加权求和,算出的相关性权重无处使用;只有加权求和没有局部激活单元,不知道该怎么给每个历史行为分配权重,退化回固定池化;只有前两者没有 Dice 激活函数和自适应正则化这些工程改进,在真实的、长尾分布严重的广告点击数据上模型训练容易不稳定或过拟合。三者组合起来,DIN 才能在保持强表达力的同时,在工业级长尾数据上稳定训练。
- **关键代码**:局部激活单元(候选 embedding 与历史行为 embedding 计算注意力分数)+ 加权求和池化的简化伪代码,参照 `13-moe-efficient/04-deepseek-v3.md` 的详略程度。
- **性能数据**:核实真实数字——论文在阿里巴巴展示广告数据集上,DIN 相对不带注意力机制的基线模型(如简单池化的 Wide&Deep/DeepFM 式模型)的 AUC 提升幅度,以及线上 A/B 测试的 CTR 提升幅度。不确定的具体数字写方向性描述加编辑备注。
- **影响 / 后续**:DIN 确立了"注意力机制动态建模用户兴趣"这一序列推荐的主流范式,直接启发了后续 DIEN(引入 GRU 建模兴趣演化过程)、DSIN(建模用户会话结构)等一系列工作,序列建模+注意力从此成为工业级推荐排序模型的标准组件之一。加跨节点链接 `→ [03-deepfm.md](03-deepfm.md) · 本文改进的固定池化用户兴趣表示方式`、`→ [05-pinsage.md](05-pinsage.md) · 同年发布,从表格特征交互转向图结构建模的互补路径`。

- [ ] **Step 1: 重新读一遍 `19-recommendation/01-wide-deep.md` 和 `19-recommendation/03-deepfm.md` 全文,确认"前作进展"里对两者的描述与各自节点自述内容一致**

- [ ] **Step 2: 写节点正文,创建配图,验证 SVG 合法性**

- [ ] **Step 3: 检查五个已知坑**

- [ ] **Step 4: 提交**

```bash
git add 19-recommendation/04-din.md 19-recommendation/assets/04-din-architecture.svg
git commit -m "feat: DIN(2018)节点正文"
```

---

## Task 7: 节点 05 —— PinSAGE(2018)

**Files:**
- Create: `19-recommendation/05-pinsage.md`
- Create: `19-recommendation/assets/05-pinsage-architecture.svg`(至少 1 张,**文件名必须以 `05-pinsage-` 开头**)

Frontmatter:

```yaml
---
name: "PinSAGE"
year: 2018
family: "19-recommendation"
order: 5
paper: "Graph Convolutional Neural Networks for Web-Scale Recommender Systems"
authors: ["Rex Ying", "Ruining He", "Kaifeng Chen", "Pong Eksombatchai", "William L. Hamilton", "Jure Leskovec"]
key_idea: "把 GraphSAGE 的归纳式图卷积扩展到 30 亿节点、180 亿边的工业级二部图(用户-物品图),用随机游走采样 + 生产者-消费者流水线训练,是 GNN 在推荐系统里最早的大规模工业落地"
---
```

九段式结构写作要点(**先用 WebSearch 核实真实数字再落笔**;若不可用,遵循已建立的编辑备注惯例。**写"前作进展"提到 GraphSAGE 时,必须重新读一遍 `17-graph-neural-networks/02-graphsage.md` 自己的正文,确认 claim 与该节点自述内容一致——这是本家族和前几轮反复强调的规则**):

- **前作进展**:[GraphSAGE](../17-graph-neural-networks/02-graphsage.md) 证明了归纳式图卷积(采样固定数量邻居 + 可学习聚合函数)能让 GNN 泛化到训练时没见过的新节点,但论文验证的图规模远小于 Pinterest 真实业务场景——Pinterest 的用户-图片-画板关系图有约 30 亿节点、180 亿边,直接套用 GraphSAGE 论文里的训练流程(单机、内存里随机采样)在这个规模下完全不可行,需要解决"如何采样最重要的邻居而不是随机采样""如何设计能在分布式集群上高效运行的训练流水线""如何给数十亿节点批量生成 embedding 而不重复计算"这些新的工程问题。
- **核心思想 + 直觉**:核心洞察是——保留 GraphSAGE"采样邻居 + 聚合"的核心框架,但把两个关键环节替换成能应对海量图规模的工程方案:用基于随机游走的重要性分数替代均匀随机采样(让模型优先聚合"确实重要"的邻居,而不是随机选择),用生产者-消费者(CPU 负责构造训练批次、GPU 负责训练)的流水线替代单机训练循环,让数据准备和模型训练可以并行重叠执行。
- **机制一(基于随机游走的重要性采样)**:从目标节点出发做多次短随机游走,统计每个被访问节点的归一化访问次数,作为这个节点对目标节点的"重要性分数";采样邻居时优先选择重要性分数最高的若干节点(而不是均匀随机采样),这样既控制了每层计算量,又让模型聚合到的是真正对目标节点有意义的邻居(重要性分数后续也直接用作聚合时的权重)。
- **机制二(高效卷积与困难负样本的课程学习)**:图卷积本身沿用 GraphSAGE 式的采样-聚合结构,但论文引入课程学习(curriculum learning)策略,训练过程中逐步加入更难区分的负样本(即和正样本有一定相关性但实际不是目标的物品),让模型在训练后期学会区分更细粒度的相关性差异,而不只是区分"完全不相关"和"相关"这种粗粒度差异。
- **机制三(生产者-消费者分布式训练流水线与 MapReduce 批量推理)**:训练时用生产者(CPU 集群,负责从图中采样构造 minibatch)和消费者(GPU,负责前向反向传播更新参数)分离的流水线架构,让数据准备和模型训练并行重叠,避免 GPU 等待数据;推理阶段(给全图数十亿节点生成 embedding)用类似 MapReduce 的批量处理方式,确保每个节点的中间计算结果只需要计算一次、可以被多个下游节点复用,避免了朴素实现里的大量重复计算。
- **三件套协同**:只有重要性采样没有高效卷积+课程学习,模型在稠密的重要邻居上训练但缺乏区分细粒度相关性的能力;只有课程学习没有分布式流水线,单机训练速度扛不住 30 亿节点的规模;只有前两者没有 MapReduce 式批量推理,虽然能训练出模型,但无法在合理时间内为全图节点生成 embedding、也就无法部署上线。三者组合起来,PinSAGE 才能在 Pinterest 真实的数十亿规模图上训练出可用且能落地部署的推荐模型。
- **关键代码**:随机游走重要性采样 + 加权聚合更新的简化伪代码,参照 `17-graph-neural-networks/02-graphsage.md`(如果该节点有类似的伪代码可参考详略程度)或 `13-moe-efficient/04-deepseek-v3.md` 的详略程度。
- **性能数据**:核实真实数字——论文报告的离线评估指标(如 hit-rate)相对纯内容特征方法、协同过滤等基线的提升,以及 Pinterest 生产环境上线后的用户参与度(engagement)提升幅度。不确定的具体数字写方向性描述加编辑备注。
- **影响 / 后续**:PinSAGE 是 GNN 技术在工业级推荐系统里最早、影响最大的落地案例之一,证明了图卷积网络可以扩展到数十亿节点/边的真实生产规模,为后续大量"用图神经网络做推荐"的工业实践(电商、社交网络等场景)提供了可复用的工程范式。这是本家族按教学顺序收录的最后一篇节点,完整走完"人工特征叉乘(Wide & Deep)→ 自动特征交互(DeepFM)→ 动态兴趣建模(DIN)→ 图结构建模(PinSAGE)"这条推荐系统主线。加跨节点链接 `→ [../17-graph-neural-networks/02-graphsage.md](../17-graph-neural-networks/02-graphsage.md) · 本文直接扩展的归纳式图卷积架构`、`→ [04-din.md](04-din.md) · 同年发布,从表格特征交互转向图结构建模的互补路径`。

- [ ] **Step 1: 重新读一遍 `17-graph-neural-networks/02-graphsage.md` 全文,确认"前作进展"里对 GraphSAGE 的描述与该节点自述内容一致**

- [ ] **Step 2: 写节点正文,创建配图,验证 SVG 合法性**

- [ ] **Step 3: 检查五个已知坑**

- [ ] **Step 4: 提交**

```bash
git add 19-recommendation/05-pinsage.md 19-recommendation/assets/05-pinsage-architecture.svg
git commit -m "feat: PinSAGE(2018)节点正文"
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

Expected: 无报错退出,输出类似 `wrote .../TIMELINE.md (89 nodes)` 和 `wrote .../families.json (19 families, 89 nodes)`(84 + 5 = 89)。

- [ ] **Step 2: 检查 TIMELINE.md 里新家族的 5 行是否正确插入**

```bash
grep -n "Wide & Deep\|YouTube DNN\|DeepFM\|DIN\|PinSAGE" TIMELINE.md
```

Expected: 5 行都出现,`\`19-recommendation\`` 出现在对应行里(注意"DIN"这个缩写可能和其他家族节点的普通文本有误匹配,人工过一遍确认是本家族的 5 行)。

- [ ] **Step 3: 检查 families.json 里新家族块,尤其每个节点 assets 数组非空**

```bash
python3 -c "
import json
data = json.load(open('web/src/data/families.json'))
fam = next((f for f in data['families'] if f['id'] == '19-recommendation'), None)
assert fam is not None, '19-recommendation 家族块缺失'
assert len(fam['nodes']) == 5, f'期望 5 个节点,实际 {len(fam[\"nodes\"])}'
assert fam['colorToken'] == '--family-19', f'colorToken 不对: {fam[\"colorToken\"]}'
for n in fam['nodes']:
    assert len(n['assets']) > 0, f'{n[\"name\"]} 的 assets 数组是空的!检查 SVG 文件名是否匹配 {n[\"path\"]} 的 stem'
print('OK', fam['label'], fam['yearRange'])
for n in fam['nodes']:
    print(' -', n['order'], n['year'], n['name'], n['assets'])
"
```

Expected: 打印 `OK 推荐系统(Recommendation Systems) [2016, 2018]`,随后 5 行,每行 `assets` 数组都非空。**如果某个节点 assets 为空,说明 SVG 文件名没有匹配上该节点 markdown 的 stem,回去改文件名重新生成。**

- [ ] **Step 4: 全仓库 SVG 合法性自查(本家族范围)**

```bash
python3 -c "
import xml.etree.ElementTree as ET
import glob
files = sorted(glob.glob('19-recommendation/assets/*.svg'))
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

Expected: tsc 无输出;vitest 全部通过,测试数量与之前(349)基本持平(本轮不新增金标本/组件测试,只有 `web/src/test/svgAssets.test.ts` 会扫到新增的 5+ 张 SVG 并新增对应测试用例)。

- [ ] **Step 6: 浏览器验证家族页面与节点详情页**

用 preview_start 起 dev server(用 `preview_logs` 确认实际绑定端口,工具报告的端口可能不准),访问 `/families/19-recommendation` 确认:
- 家族标题、"一句话定位"正文正常渲染
- 5 个节点卡片按 Task 2 README 里的顺序显示(Wide & Deep → YouTube DNN → DeepFM → DIN → PinSAGE),颜色为红色 `#f87171`
- 逐一点进 5 个节点详情页(`/families/19-recommendation/01-wide-deep` 等),确认走的是 `NodePage.tsx` 通用渲染路径(非金标本),标题/作者/正文/配图/前后节点导航正常显示,console 无 error
- 访问首页确认头部计数变成"19 家族 · 89 节点",侧栏家族列表和时间线可视化里都能看到"推荐系统(Recommendation Systems)"及其 5 个节点

- [ ] **Step 7: 提交**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add TIMELINE.md web/src/data/families.json
git commit -m "feat: 生成 TIMELINE.md / families.json,第 19 个家族接入完成

推荐系统(Recommendation)家族全部 5 篇节点 markdown 正本 + README 就位,
python3 scripts/generate_timeline.py 重新生成产物,tsc + vitest 全项目
通过,浏览器验证家族页面与节点详情页渲染正常,首页计数更新为
19 家族 89 节点。金标本交互页按计划留待后续单独轮次补,本轮完成。"
```

---

## Self-Review 记录(写 plan 时已自查)

1. **Spec 覆盖**:spec 第 3 节(节点列表,含排序说明)→ Task 3-7,README 里同步注明排序说明;第 4 节(节点写作规范,含 SVG 命名规则)→ 每个节点 Task 的 Step 强调;第 5 节(家族 README,含手写子时间线表格要求)→ Task 2 的完整表格内容;第 6 节(注册文件,含 familyHero.ts 提前处理)→ Task 1 + Task 3;第 7 节(验收标准,含 assets 非空断言)→ Task 8 的检查脚本逐条对应;第 8 节(Out of scope)→ 全程未涉及金标本/foundations 改动。
2. **Placeholder 扫描**:每个节点任务给出的是"必须包含的真实事实清单 + 结构大纲",不是"TBD"式占位符,延续 spec 第 4 节和前四轮已验证有效的做法。
3. **一致性检查**:5 个节点的 frontmatter `order` 字段(1-5)与 Task 3-7 顺序一致(Wide&Deep/YouTube DNN/DeepFM/DIN/PinSAGE);跨节点链接指向的文件名(`01-wide-deep.md` `02-youtube-dnn.md` `03-deepfm.md` `04-din.md` `05-pinsage.md`)在各任务间保持一致拼写,全部采用 `[filename.md](filename.md)` 括号语法;跨节点事实一致性检查步骤(重读前置节点原文)在 Task 5/6/7 里逐一列出,PinSAGE 节点特别强调需要重读 `17-graph-neural-networks/02-graphsage.md`;familyHero.ts 的修复前移到 Task 3(第一个节点任务);家族 README 的"子时间线"是完整手写的 5 行真实表格,并在正文里显式注明 DIN/PinSAGE 的教学顺序例外(与 spec 第 3 节"排序说明"一致)。
