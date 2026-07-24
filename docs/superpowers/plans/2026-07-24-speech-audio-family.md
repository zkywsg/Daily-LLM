# 第 18 个家族(语音/音频 Speech/Audio)markdown 正本 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新建 `18-speech-audio/` 家族目录,写出 1 篇家族 README + 5 篇节点 markdown 正本(Wav2Vec 2.0 2020 → HuBERT 2021 → Whisper 2022 → AudioLM 2022 → MusicGen 2023,教学顺序,见下方"排序说明"),接入 `generate_timeline.py` / `FamilyId` / `tokens.css` 三处注册点,不做金标本交互页。

**Architecture:** 复用仓库已收敛 17 个家族的写作模板(frontmatter + 前作进展/核心思想+直觉/机制一二三/三件套协同/关键代码/性能数据/影响后续 九段式,扁平 `##` 标题,不嵌套),每篇配 1-2 张手绘风格 SVG。吸取前三轮(16-world-models、17-graph-neural-networks)沉淀的全部教训:SVG 命名必须匹配节点 markdown 完整 stem、家族 README 子时间线必须手写真实表格、familyHero.ts 缺口在第一个节点任务里就处理掉、跨节点链接用 markdown 语法、跨节点事实引用前重读对方原文。

**Tech Stack:** 纯 markdown + SVG,Python 脚本 `scripts/generate_timeline.py` 生成 TIMELINE.md/families.json,TypeScript `FamilyId` 类型,CSS custom property。

---

## 参考:设计文档

本 plan 的所有决策依据 `docs/superpowers/specs/2026-07-24-speech-audio-family-design.md`,写节点前建议先读一遍该文件确认章节结构约定与排序说明。

## 参考:写作模板锚点文件

写正文前先读这几个近期节点作为结构范例(不要照抄措辞,只借鉴章节骨架和"三件套协同"收尾的写法):
- `17-graph-neural-networks/01-gcn.md`、`17-graph-neural-networks/05-graphormer.md` —— 最近一轮已验证过的完整范例,扁平 `##` 标题结构
- `17-graph-neural-networks/README.md` —— 家族 README 结构范例,包括"子时间线"手写表格的真实格式
- `13-moe-efficient/04-deepseek-v3.md` —— "关键代码"一节的详略程度参考

## 已知的五个必须规避的坑(前三轮反复踩过)

1. **CommonMark 加粗定界符边界情况**:`**` 紧贴标点(引号/问号/括号)时,另一侧必须是空白或标点才能正确解析。写完每篇后人工过一遍 `**` 前后字符。
2. **`$` 货币符号与 remark-math 冲突**:涉及美元数字一律转义成 `\$`。本家族提到"云算力成本""数据采集成本"时要注意。
3. **跨节点链接必须用 markdown link 语法,不能写纯文本**:必须写成 `→ [02-hubert.md](02-hubert.md) · ...`。
4. **SVG 资产文件名必须以节点 markdown 的完整 stem 开头**:例如节点 `01-wav2vec2.md` 的配图必须命名为 `01-wav2vec2-architecture.svg` 或 `01-wav2vec2-xxx.svg`,不能用缩写。
5. **跨节点事实一致性**:写"前作进展"或"影响/后续"提到已写好的兄弟节点时,必须重新读一遍那个节点自己的正文,确认自己写的 claim 与对方自述内容一致,不能凭训练知识里的一般印象凭空归因。

---

## Task 1: 家族基础设施注册

**Files:**
- Modify: `scripts/generate_timeline.py`
- Modify: `web/src/types/family.ts`
- Modify: `web/src/styles/tokens.css`

- [ ] **Step 1: 在 `scripts/generate_timeline.py` 的 `FAMILY_IDS` 列表末尾追加新家族 id**

打开 `scripts/generate_timeline.py`,找到以 `"17-graph-neural-networks",` 结尾的 `FAMILY_IDS` 列表,改成:

```python
FAMILY_IDS = [
    "01-cnn", "02-rnn-lstm", "03-word-embedding", "04-gan",
    "05-transformer", "06-bert-family", "07-gpt-scaling",
    "08-vit", "09-multimodal-clip", "10-diffusion",
    "11-peft-lora", "12-rlhf-alignment", "13-moe-efficient",
    "14-rag-agent", "15-reasoning-o1-r1", "16-world-models",
    "17-graph-neural-networks", "18-speech-audio",
]
```

- [ ] **Step 2: 在 `web/src/types/family.ts` 的 `FamilyId` 联合类型末尾追加新家族 id**

找到以 `| "17-graph-neural-networks";` 结尾的联合类型定义,改成:

```typescript
  | "17-graph-neural-networks"
  | "18-speech-audio";
```

- [ ] **Step 3: 在 `web/src/styles/tokens.css` 新增家族色 token**

找到 `--family-17: #f43f5e; /* GNN 玫瑰红 */` 这一行,在它之后新增一行:

```css
  --family-18: #fb7185; /* Speech/Audio 浅玫瑰红 */
```

- [ ] **Step 4: 验证 tsc(预期会因 familyHero.ts 报一个已知错误,不用现在修)**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web && npx tsc --noEmit
```

Expected: 报一个 `web/src/components/home/familyHero.ts` 的 `TS2741` 错误(`Record<FamilyId, string>` 缺 `"18-speech-audio"` key)。这是预期的、已知的中间状态——前两轮已验证过这个错误在追加节点(Task 3)时一并修复是正确的做法,不需要现在处理,继续往下走。

- [ ] **Step 5: 提交**

```bash
git add scripts/generate_timeline.py web/src/types/family.ts web/src/styles/tokens.css
git commit -m "feat: 第 18 个家族(语音/音频 Speech/Audio)基础设施注册

追加 FAMILY_IDS / FamilyId / --family-18 三处,markdown 正本在后续
任务里逐篇写。tsc 此时会报 familyHero.ts 的已知错误,留到 Task 3(第一个
节点)一并修复,与前两轮做法一致。"
```

---

## Task 2: 家族 README

**Files:**
- Create: `18-speech-audio/README.md`

- [ ] **Step 1: 写家族 README**

创建 `18-speech-audio/README.md`,章节结构复用 `17-graph-neural-networks/README.md`:

```markdown
# 语音/音频模型(Speech/Audio Models)

> **把语音识别从"手工特征 + 声学模型流水线"推向"自监督表征 + 大规模弱监督端到端",再把音频生成从"波形合成"推向"离散 token 序列上的语言建模"。**

## 一句话定位

语音/音频是与文本、图像并列的第三条经典输入模态——原始波形是连续、高采样率(16kHz+)的信号,直接建模计算量极大,历史上长期依赖"手工特征(MFCC)+ 声学模型 + 语言模型"的流水线。这条主线要回答两个递进的问题:**如何用自监督/弱监督的方式让模型直接从原始波形学到好的语音表征或识别能力**(Wav2Vec 2.0 → HuBERT → Whisper),以及**如何把音频压缩成离散 token 后,复用文本领域已经验证过的语言建模范式来做音频生成**(AudioLM → MusicGen)。2020 年 **Wav2Vec 2.0** 用对比学习从原始波形自监督学出可迁移的语音表征,让下游 ASR 只需极少标注数据;2021 年 **HuBERT** 用离线聚类生成伪标签替代不稳定的对比学习目标;2022 年 **Whisper** 证明"68 万小时弱监督多语言数据 + 标准 Transformer"能在完全不微调的情况下达到接近监督 SOTA 的鲁棒性,把"数据规模碾压架构精巧"这条 scaling 经验从文本搬到了语音;同年 **AudioLM** 把音频离散化成语义 + 声学两级 token,用语言模型做 next-token 预测生成连贯音频;2023 年 **MusicGen** 把 AudioLM 的多阶段级联简化成单阶段 Transformer + 码本交错,用文本/旋律条件做可控音乐生成。这家族要回答的问题是:**语音/音频模型是怎么从"自监督表征学习"演化到"把音频当成另一种语言来生成"的**。

## 概念本身

### 两条子线索共享的前提:先把连续音频离散化

这 5 篇论文可以分成两条子线索,但都建立在同一个前提之上——**把连续、高采样率的原始波形压缩/离散化成一段更短、更抽象的 token 序列**,再在这个 token 序列上做后续建模:

- **表征学习 / 识别这条线(Wav2Vec 2.0 → HuBERT → Whisper)**:目标是学到能直接支撑下游任务(ASR、说话人识别等)的表征。Wav2Vec 2.0 用可学习的量化模块(product quantization)把连续特征离散化后作为对比学习的目标;HuBERT 用离线 k-means 聚类生成离散伪标签,把自监督问题转化成标准的掩码分类任务;Whisper 则完全绕开自监督预训练,直接用大规模弱监督数据端到端训练一个标准 Transformer encoder-decoder。
- **生成这条线(AudioLM → MusicGen)**:目标是生成新的音频。两篇论文都依赖神经音频编解码器(SoundStream / EnCodec)把音频压缩成若干层残差量化(RVQ)离散 token,再用自回归语言模型在这些 token 上做 next-token 预测——这是"把 [GPT 的自回归语言建模范式](../07-gpt-scaling/03-gpt3.md)搬到音频模态"的直接实践,呼应 [ViT](../08-vit/01-vit.md) 把 Transformer 搬到图像 patch 序列、[Graphormer](../17-graph-neural-networks/05-graphormer.md) 把 Transformer 搬到图数据的"架构统一"叙事。

### Whisper 是这条主线里的一个转折点

Wav2Vec 2.0 和 HuBERT 都遵循"自监督预训练 + 下游任务微调"这一 2018 年后 NLP/CV 领域(BERT、MAE)反复验证过的范式;Whisper 则证明——只要弱监督数据的规模和多样性足够大(68 万小时、覆盖 96 种语言、多任务格式),完全监督式端到端训练也能达到甚至超过"自监督预训练+微调"路线的鲁棒性,且不需要针对每个下游场景微调。这与 [GPT-3](../07-gpt-scaling/03-gpt3.md) 证明"规模本身就是能力"的叙事是同一条逻辑在语音识别上的重演。

## 子时间线

| 年份 | 名字 | 关键贡献 | 之前卡在哪 |
|------|------|---------|-----------|
| 2020 | **Wav2Vec 2.0** | Baevski et al.(Meta AI)——用对比学习从原始波形自监督学习语音表征:CNN 特征编码器 + 可学习量化模块生成离散对比目标 + Transformer 掩码预测,让下游 ASR 只需 10 分钟到 100 小时标注数据就能微调到有竞争力的 WER | 此前的语音识别严重依赖大量标注数据训练声学模型,标注语音数据成本高(需要专业转写),低资源语言/场景难以覆盖;早期自监督尝试(wav2vec、vq-wav2vec)表征质量和下游任务效果有限 |
| 2021 | **HuBERT** | Hsu et al.(Meta AI)——用离线 k-means 聚类对声学特征生成离散伪标签,再做 BERT 式掩码预测(分类任务而非对比学习),并通过迭代式重新聚类(用前一轮模型的隐藏层特征重新聚类)不断提纯伪标签的音素区分度 | Wav2Vec 2.0 的对比学习目标(量化码本)与被训练的表征联合演化,训练早期目标本身不稳定;对比学习还需要精心设计负样本采样策略,容易受相似发音片段干扰 |
| 2022 | **Whisper** | Radford et al.(OpenAI)——68 万小时弱监督多语言多任务数据(网络爬取的音频-文本对)+ 标准 Transformer encoder-decoder,单一模型端到端支持转写、翻译、语言识别,零样本鲁棒性接近或超过针对特定数据集微调的模型 | 自监督预训练模型(Wav2Vec2/HuBERT)仍需针对下游任务用标注数据微调,跨领域/口音/背景噪音的泛化能力有限,微调后的模型往往对训练分布外的场景脆弱 |
| 2022 | **AudioLM** | Borsos et al.(Google)——把音频离散化成语义 token(来自自监督模型,捕捉长程一致性)和声学 token(来自神经编解码器 SoundStream,捕捉音色/说话人细节)两级表示,用语言模型对语义→粗声学→细声学做层级式 next-token 预测,不需要文本条件就能生成语义连贯、说话人一致的语音/钢琴续写 | 此前的神经音频合成(如 WaveNet)擅长生成局部逼真的波形,但缺乏长程语义一致性(几十秒后内容/说话人容易跑偏);自监督表征学习(Wav2Vec2/HuBERT)擅长做理解任务,没人把它系统用于生成 |
| 2023 | **MusicGen** | Copet et al.(Meta AI)——用 EnCodec 神经编解码器产生的多层残差量化(RVQ)token,配合码本交错(codebook interleaving)技巧,把多个并行码本流摊平成一条序列,单阶段 Transformer decoder 自回归建模,支持文本(T5 编码)和旋律(chromagram)双重条件控制 | AudioLM/MusicLM 这类级联多阶段模型(语义 token → 粗声学 token → 细声学 token 分别用独立模型生成)结构复杂、推理慢、误差会在阶段间累积传播 |

## 依赖与延伸

- 前置依赖:[Transformer](../05-transformer/01-transformer.md)(五篇论文全部基于的核心架构)、[GPT-3](../07-gpt-scaling/03-gpt3.md)(AudioLM/MusicGen 直接复用的自回归语言建模范式)
- 延伸方向:VALL-E、SpeechT5 等更晚近的语音合成/克隆模型,以及 EnCodec/SoundStream 这类神经音频编解码器本身的压缩技术细节,本仓库暂未单独收录
```

- [ ] **Step 2: 验证跨链接路径存在**

```bash
ls /Users/lauzanhing/Desktop/Daily-LLM/05-transformer/01-transformer.md /Users/lauzanhing/Desktop/Daily-LLM/07-gpt-scaling/03-gpt3.md /Users/lauzanhing/Desktop/Daily-LLM/08-vit/01-vit.md /Users/lauzanhing/Desktop/Daily-LLM/17-graph-neural-networks/05-graphormer.md
```

Expected: 四个文件都存在,不报错。

- [ ] **Step 3: 提交**

```bash
git add 18-speech-audio/README.md
git commit -m "feat: 语音/音频模型(Speech/Audio)家族 README"
```

---

## Task 3: 节点 01 —— Wav2Vec 2.0(2020)

**Files:**
- Create: `18-speech-audio/01-wav2vec2.md`
- Create: `18-speech-audio/assets/01-wav2vec2-architecture.svg`(至少 1 张,**文件名必须以 `01-wav2vec2-` 开头**)
- Modify: `web/src/components/home/familyHero.ts`

Frontmatter:

```yaml
---
name: "Wav2Vec 2.0"
year: 2020
family: "18-speech-audio"
order: 1
paper: "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"
authors: ["Alexei Baevski", "Henry Zhou", "Abdelrahman Mohamed", "Michael Auli"]
key_idea: "CNN 特征编码器 + 可学习量化模块生成离散对比目标 + Transformer 掩码预测,用对比学习从原始波形自监督学到可迁移的语音表征,让下游 ASR 只需极少标注数据就能微调"
---
```

九段式结构写作要点(**先用 WebSearch 核实真实数字再落笔**;若 WebSearch 不可用,用训练知识里最有把握的说法,并加编辑备注注明"未经实时核实,建议核对原论文",遵循前几轮已建立的备注写法惯例——参照 `16-world-models/03-dreamerv3.md` 的性能数据段落写法):

- **前作进展**:语音识别长期依赖"手工特征(如 MFCC)+ 声学模型 + 语言模型"的流水线,需要大量人工转写的标注数据训练声学模型,标注成本高,低资源语言覆盖差。早期自监督尝试(wav2vec 2019、vq-wav2vec 2019,同一批作者)已经证明"从无标注音频预训练再做下游任务"这条路可行,但表征质量、下游任务提升幅度都比较有限,量化和上下文建模是分开两阶段做的(vq-wav2vec 先离散化,再单独训练 BERT)。
- **核心思想 + 直觉**:核心洞察是——把"学习连续表征"和"学习离散量化目标"放进同一个端到端训练过程,让模型在掩码预测的同时联合学习一套量化码本作为对比学习的目标,不再需要分两阶段训练。直觉类似 BERT 的掩码语言建模,但语音是连续信号,没有天然的离散"词",所以需要额外一步——用可学习的量化模块把连续特征转成离散单元,再让 Transformer 在这些离散目标上做对比学习意义下的"完形填空"。
- **机制一(CNN 特征编码器)**:多层一维卷积网络把原始波形(16kHz 采样)转换成较低频率(约 50Hz,即每 20ms 一帧)的潜在特征序列 Z,压缩掉冗余的高频细节,保留语音相关的结构信息。
- **机制二(量化模块 + 对比学习目标)**:用 Gumbel-softmax 乘积量化(product quantization)把连续的 Z 映射到离散的量化表示 Q,作为对比学习的"正确答案"。训练时随机 mask 掉一部分连续帧,Transformer 需要从上下文预测被 mask 位置对应的量化表示——对比损失要求模型能从一批候选(真实量化目标 + 若干干扰项)里正确识别出真实目标,同时加一个多样性损失鼓励量化码本被充分利用(避免所有帧都量化到同一个码本项)。
- **机制三(Transformer 上下文编码 + 掩码预测)**:CNN 输出的潜在特征序列被随机 mask 掉若干连续片段(span masking,而非单帧),送入标准 Transformer encoder 得到上下文表征 C,再用 C 在被 mask 的位置预测机制二里定义的量化目标,梯度同时更新 CNN 编码器、量化模块、Transformer 三部分。
- **三件套协同**:只有 CNN 编码器没有量化模块,对比学习没有离散、稳定的目标可用,退化成在连续空间做回归,容易学到平凡解;只有量化模块没有 Transformer 掩码预测,量化码本学不到有意义的语音结构,只是单纯的特征压缩;只有 Transformer 没有前两者,没有原始波形到离散目标的转换流程,无法构造出自监督训练所需的"完形填空"任务。三者组合起来,才是 Wav2Vec 2.0 能端到端联合学习表征和量化目标的完整机制。
- **关键代码**:CNN 特征编码器 + Gumbel-softmax 量化 + Transformer 掩码对比学习损失的简化 PyTorch 伪代码,参照 `13-moe-efficient/04-deepseek-v3.md` 的详略程度,不需要完整可运行。
- **性能数据**:核实真实数字——在 Librispeech 上用 10 分钟 / 1 小时 / 100 小时 / 960 小时不同规模的标注数据微调后的 WER(词错误率),尤其是论文强调的"仅 10 分钟标注数据"这一极低资源场景下的结果,以及相对此前监督基线的提升幅度。不确定的具体数字写方向性描述加编辑备注。
- **影响 / 后续**:Wav2Vec 2.0 证明了自监督预训练在语音领域同样可行且效果显著,催生了大量后续自监督语音表征工作,直接推动了 HuBERT 的诞生——HuBERT 正是为了解决 Wav2Vec 2.0 联合学习量化目标带来的训练不稳定问题而提出的。加跨节点链接 `→ [02-hubert.md](02-hubert.md) · 解决本文对比学习目标联合演化导致的训练不稳定问题`。

## Context

**Important additional fix needed in this task**(与前两轮 Task 1→Task 3 的处理方式完全一致):Task 1 widened the `FamilyId` TypeScript union type, which will break `web/src/components/home/familyHero.ts` — it has an exhaustive `Record<FamilyId, string>` map (`FAMILY_HERO`) picking one representative node per family. Add this line to the `FAMILY_HERO` object (follow the existing pattern in the file):

```typescript
  "18-speech-audio": "18-speech-audio/01-wav2vec2.md",
```

- [ ] **Step 1: 写节点正文,创建配图,验证 SVG 合法性**

```bash
python3 -c "
import xml.etree.ElementTree as ET
ET.parse('18-speech-audio/assets/01-wav2vec2-architecture.svg')
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
git add 18-speech-audio/01-wav2vec2.md 18-speech-audio/assets/01-wav2vec2-architecture.svg web/src/components/home/familyHero.ts
git commit -m "feat: Wav2Vec 2.0(2020)节点正文

补上 Task 1 遗留的 familyHero.ts FAMILY_HERO 缺口(第 18 个家族现在
有了第一个节点,可以选它作为家族卡片的代表作品)。"
```

---

## Task 4: 节点 02 —— HuBERT(2021)

**Files:**
- Create: `18-speech-audio/02-hubert.md`
- Create: `18-speech-audio/assets/02-hubert-architecture.svg`(至少 1 张,**文件名必须以 `02-hubert-` 开头**)

Frontmatter:

```yaml
---
name: "HuBERT"
year: 2021
family: "18-speech-audio"
order: 2
paper: "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units"
authors: ["Wei-Ning Hsu", "Benjamin Bolte", "Yao-Hung Hubert Tsai", "Kushal Lakhotia", "Ruslan Salakhutdinov", "Abdelrahman Mohamed"]
key_idea: "用离线 k-means 聚类对声学特征生成离散伪标签,再做 BERT 式掩码预测(分类而非对比学习),配合迭代式重新聚类不断提纯伪标签的音素区分度,解决 Wav2Vec 2.0 联合学习量化目标带来的训练不稳定问题"
---
```

九段式结构写作要点(**先用 WebSearch 核实真实数字再落笔**;若不可用,遵循已建立的编辑备注惯例。**写"前作进展"section 提到 Wav2Vec 2.0 时,必须重新读一遍 `18-speech-audio/01-wav2vec2.md` 自己的正文,确认 claim 与该节点自述内容一致**):

- **前作进展**:Wav2Vec 2.0 把"学连续表征"和"学离散量化目标"放进同一个端到端训练过程联合优化,但这种联合演化本身带来一个问题——量化码本在训练早期还很随机、质量差,用它作为对比学习的目标会给 Transformer 提供噪声信号,导致训练早期不稳定;此外对比学习需要精心设计负样本采样策略(从同一句话的其他帧里采样干扰项),容易受到发音相似片段的干扰,增加了优化难度。
- **核心思想 + 直觉**:核心洞察是——把"生成离散目标"和"学习上下文表征"这两步彻底解耦成两个阶段:先用一个独立的、离线的聚类步骤(不依赖被训练模型本身)生成一套固定的伪标签,再让模型去做标准的、类似 BERT 的掩码分类任务(预测每个被 mask 位置的伪标签类别)。因为目标在训练过程中是固定不变的,不会和被训练的表征"共谋"导致目标退化,训练目标比 Wav2Vec 2.0 的联合优化更稳定。
- **机制一(离线聚类生成伪标签)**:第一轮迭代直接对传统声学特征(如 MFCC)做 k-means 聚类,把每一帧语音映射到某个聚类中心的类别标签,作为这一帧的伪标签;这些伪标签虽然和真实音素边界不完全对齐,但已经携带了粗粒度的语音结构信息。
- **机制二(BERT 式掩码预测)**:把输入特征序列的若干片段 mask 掉(span masking,和 Wav2Vec 2.0 类似),送入 Transformer 编码器,在被 mask 的位置用一个分类头预测机制一生成的伪标签类别,用标准交叉熵损失训练——这是一个纯粹的分类任务,不需要对比学习里的负采样。
- **机制三(迭代式重新聚类)**:训练完第一轮 HuBERT 模型后,不是就此停止,而是用这个模型某一中间隐藏层的表征重新做一次 k-means 聚类,生成质量更高、更贴近音素边界的新伪标签,再重新训练一版 HuBERT——这个"训练模型→用模型自身特征重新聚类→用新标签重新训练"的迭代过程通常进行 2-3 轮,每一轮伪标签的音素区分度都比上一轮更好。
- **三件套协同**:只有离线聚类没有掩码预测,伪标签本身不会被用来训练出更好的表征,聚类步骤是无意义的;只有掩码预测没有离线聚类,没有离散的训练目标可用;只有前两者没有迭代式重新聚类,伪标签质量停留在初始 MFCC 聚类的粗糙水平,无法随着表征质量提升而同步提纯。三者组合起来,HuBERT 才能用一个完全解耦、逐轮自举提升的训练流程,达到比 Wav2Vec 2.0 更稳定的训练过程和相当或更好的下游效果。
- **关键代码**:k-means 聚类生成伪标签 + Transformer 掩码分类损失的简化伪代码,参照 `13-moe-efficient/04-deepseek-v3.md` 的详略程度。
- **性能数据**:核实真实数字——在 Librispeech 不同标注数据规模下微调的 WER,与 Wav2Vec 2.0 在相同实验设置下的对比,以及论文报告的训练稳定性/收敛速度方面的定性结论。
- **影响 / 后续**:HuBERT 的离散伪标签思路后续被广泛借鉴(包括用作 AudioLM 语义 token 的技术路线之一),证明了"离线聚类 + 掩码分类"这一更简单、更稳定的自监督范式在语音领域同样有效,是 Wav2Vec 2.0 之后自监督语音表征学习的另一个主流分支。加跨节点链接 `→ [01-wav2vec2.md](01-wav2vec2.md) · 本文解决的对比学习训练不稳定问题`、`→ [04-audiolm.md](04-audiolm.md) · 本文的离散伪标签思路呼应该文语义 token 的技术路线`。

- [ ] **Step 1: 写节点正文,创建配图,验证 SVG 合法性**(命令同上,路径替换成本任务对应文件)

- [ ] **Step 2: 检查五个已知坑**(加粗定界符、`$` 转义、跨节点链接语法、SVG 命名、跨节点事实一致性)

- [ ] **Step 3: 提交**

```bash
git add 18-speech-audio/02-hubert.md 18-speech-audio/assets/02-hubert-architecture.svg
git commit -m "feat: HuBERT(2021)节点正文"
```

---

## Task 5: 节点 03 —— Whisper(2022)

**Files:**
- Create: `18-speech-audio/03-whisper.md`
- Create: `18-speech-audio/assets/03-whisper-architecture.svg`(至少 1 张,**文件名必须以 `03-whisper-` 开头**)

Frontmatter:

```yaml
---
name: "Whisper"
year: 2022
family: "18-speech-audio"
order: 3
paper: "Robust Speech Recognition via Large-Scale Weak Supervision"
authors: ["Alec Radford", "Jong Wook Kim", "Tao Xu", "Greg Brockman", "Christine McLeavey", "Ilya Sutskever"]
key_idea: "68 万小时弱监督多语言多任务数据 + 标准 Transformer encoder-decoder,单一模型端到端支持转写/翻译/语言识别,零样本鲁棒性接近或超过针对特定数据集微调的模型,证明'数据规模碾压架构精巧'这条 scaling 经验在语音识别上同样成立"
---
```

九段式结构写作要点(**先用 WebSearch 核实真实数字再落笔**;若不可用,遵循已建立的编辑备注惯例。**写"前作进展"提到 Wav2Vec2/HuBERT 时,必须重新读一遍两者各自的正文,确认 claim 一致**):

- **前作进展**:Wav2Vec 2.0 和 HuBERT 都遵循"自监督预训练 + 下游任务微调"的范式——预训练阶段不需要标注数据,但要真正用于语音识别,仍然需要在特定数据集(如 Librispeech)上用标注数据微调,微调后的模型对训练分布之外的口音、背景噪音、录音条件等场景的泛化能力有限,一旦部署场景和微调数据分布不一致,识别效果会明显下降。
- **核心思想 + 直觉**:核心洞察是——与其追求"预训练表征质量",不如直接用海量、多样、弱监督(不追求转写质量完美,靠规模弥补噪声)的音频-文本配对数据端到端训练一个标准的序列到序列模型,让模型在训练阶段就见过足够多样的口音、语言、录音条件、背景噪音组合,从而在不做任何针对性微调的情况下(zero-shot)就有很强的鲁棒性。这本质上是"用规模换鲁棒性",与 GPT-3 用规模换少样本泛化能力是同一条逻辑在语音上的重演。
- **机制一(标准 Transformer encoder-decoder + log-mel 频谱输入)**:输入是音频转换成的 log-mel 频谱图(而非原始波形),encoder 用标准 Transformer 处理频谱特征,decoder 自回归生成文本 token,整体架构没有任何语音专用的特殊设计,刻意选用标准架构以验证"数据规模而非架构精巧"是关键因素。
- **机制二(大规模弱监督数据收集与过滤)**:从互联网收集约 68 万小时音频及其对应的文字(字幕、转写等),覆盖约 96 种语言;由于这些配对数据质量参差不齐(部分是机器生成的低质量转写),用一系列启发式规则和分类器过滤掉可能是机器翻译/生成而非人工转写的样本,尽量保留高质量的自然语言监督信号。
- **机制三(多任务统一格式)**:用特殊 token 把转写(同语言语音转文本)、翻译(任意语言语音转英文文本)、语言识别、时间戳预测、语音活动检测等多个任务统一编码进同一个 sequence-to-sequence 格式里——decoder 的第一个 token 指定任务类型和目标语言,模型根据这个前缀 token 决定接下来生成什么样的输出,一个模型同时具备多种能力,不需要为每个任务单独训练。
- **三件套协同**:只有标准架构没有大规模数据,退化成普通的小规模监督 ASR,没有零样本鲁棒性优势;只有大规模数据没有多任务格式,模型只能做单一转写任务,浪费了数据里蕴含的翻译、语言识别等多样信号;只有前两者没有过滤流程,大量低质量的机器生成"伪转写"会污染训练信号,削弱最终效果。三者组合起来,Whisper 才能在完全不微调的情况下达到接近监督 SOTA 的鲁棒性。
- **关键代码**:log-mel 频谱预处理 + 标准 Transformer encoder-decoder + 多任务特殊 token 前缀的简化伪代码,参照 `13-moe-efficient/04-deepseek-v3.md` 的详略程度。
- **性能数据**:核实真实数字——在多个跨领域测试集(而非单一 Librispeech test-clean)上的零样本 WER,与专门在这些数据集上微调的监督模型对比的鲁棒性差异,以及不同模型规模(tiny 到 large)的效果梯度。不确定的具体数字写方向性描述加编辑备注。
- **影响 / 后续**:Whisper 成为事实上的开源 ASR 标准基线,催生了大量衍生工作(推理加速、蒸馏小模型、微调到特定领域等),也进一步验证了"规模化弱监督"这条路径在语音之外的其他感知模态上的可推广性,是这条主线从"自监督表征学习"转向"离散 token 语言建模做生成"(AudioLM/MusicGen)之前的最后一个识别类节点。加跨节点链接 `→ [02-hubert.md](02-hubert.md) · 本文放弃的自监督预训练+微调范式`、`→ [04-audiolm.md](04-audiolm.md) · 同样基于 Transformer 但转向生成任务的下一篇`。

- [ ] **Step 1: 重新读一遍 `18-speech-audio/01-wav2vec2.md` 和 `18-speech-audio/02-hubert.md` 全文,确认"前作进展"里对两者的描述与各自节点自述内容一致**

- [ ] **Step 2: 写节点正文,创建配图,验证 SVG 合法性**

- [ ] **Step 3: 检查五个已知坑**

- [ ] **Step 4: 提交**

```bash
git add 18-speech-audio/03-whisper.md 18-speech-audio/assets/03-whisper-architecture.svg
git commit -m "feat: Whisper(2022)节点正文"
```

---

## Task 6: 节点 04 —— AudioLM(2022)

**Files:**
- Create: `18-speech-audio/04-audiolm.md`
- Create: `18-speech-audio/assets/04-audiolm-architecture.svg`(至少 1 张,**文件名必须以 `04-audiolm-` 开头**)

Frontmatter:

```yaml
---
name: "AudioLM"
year: 2022
family: "18-speech-audio"
order: 4
paper: "AudioLM: a Language Modeling Approach to Audio Generation"
authors: ["Zalán Borsos", "Raphaël Marinier", "Damien Vincent", "Eugene Kharitonov", "Olivier Pietquin", "Matt Sharifi", "Dominik Roblek", "Olivier Teboul", "David Grangier", "Marco Tagliasacchi", "Neil Zeghidour"]
key_idea: "把音频离散化成语义 token(捕捉长程一致性)和声学 token(捕捉音色/说话人细节)两级表示,用语言模型对两级 token 做层级式 next-token 预测,不需要文本条件也能生成语义连贯、说话人一致的语音/音乐续写"
---
```

九段式结构写作要点(**先用 WebSearch 核实真实数字再落笔**;若不可用,遵循已建立的编辑备注惯例。**写"前作进展"提到 Whisper/HuBERT 时,必须重新读一遍各自正文确认一致**):

- **前作进展**:Wav2Vec 2.0、HuBERT、Whisper 这条线索都在解决"理解"类任务(识别、表征学习),而神经音频"生成"这件事此前主要靠 WaveNet 这类自回归波形合成模型,擅长在局部生成逼真的音频细节,但缺乏长程语义一致性——生成几十秒之后,内容或说话人身份容易"跑偏",因为逐采样点建模的自回归模型很难在如此细粒度的时间尺度上维持长程结构。同时,自监督语音表征学习(Wav2Vec2/HuBERT)已经证明可以学到蕴含长程语义结构的离散/连续表征,但此前没有工作系统性地把这类表征用于指导生成。
- **核心思想 + 直觉**:核心洞察是——把"保证长程语义连贯"和"保证局部声学细节逼真"这两个目标分解成两级不同粒度的离散 token,分别由不同的模型/机制生成,再用语言模型对这两级 token 做层级式的自回归预测。粗粒度的语义 token 负责"接下来该说/演奏什么内容、是谁在说/演奏",细粒度的声学 token 负责"这段内容具体听起来是什么样的音色和声学细节"——用语言模型在离散 token 序列上做 next-token 预测,和文本 GPT 系列的自回归生成是同一套范式。
- **机制一(语义 token:来自自监督音频模型)**:用一个预训练好的自监督模型(w2v-BERT)提取音频的中间层表征,再离散化(聚类)成语义 token 序列,采样率较低(粗粒度,每个 token 覆盖更长的时间跨度),这套 token 携带的是内容和说话人身份等长程结构信息,而非精细的声学细节。
- **机制二(声学 token:来自神经编解码器,残差量化)**:用神经音频编解码器 SoundStream 把音频压缩成多层残差向量量化(RVQ)token——第一层码本捕捉粗粒度的声学信息(如整体音色),后续层逐层补充更精细的声学细节,这套 token 采样率更高(细粒度),负责重建出高保真的波形。
- **机制三(层级式级联生成)**:整个生成过程分三个阶段级联:第一阶段自回归生成语义 token 序列(决定"内容和说话人是什么");第二阶段以语义 token 为条件,自回归生成声学 token 的粗粒度码本层(决定"大致听起来是什么样");第三阶段以前两阶段的结果为条件,生成声学 token 的精细码本层(补充声学细节)。三个阶段分别用独立的 Transformer decoder 训练,推理时按顺序级联执行。
- **三件套协同**:只有语义 token 没有声学 token,能保证内容连贯但生成不出具体的高保真波形;只有声学 token 没有语义 token,局部声学细节可能很逼真但几十秒后内容/说话人容易漂移(退化回 WaveNet 类模型的老问题);只有前两者没有层级式级联结构,粗细粒度的 token 之间没有清晰的条件依赖关系,模型无法先确定"说什么"再确定"听起来怎么样"。三者组合起来,AudioLM 才能在不需要文本条件的情况下,生成既语义连贯又声学逼真的音频续写。
- **关键代码**:语义 token 提取 + 声学 token(RVQ)提取 + 三阶段级联 Transformer decoder 的简化伪代码,参照 `13-moe-efficient/04-deepseek-v3.md` 的详略程度。
- **性能数据**:核实真实数字——论文报告的语音续写在说话人身份保持、语义连贯性上的人工评估结果(如可懂度、自然度打分),以及钢琴音乐续写在旋律连贯性上的定性/定量结果。不确定的具体数字写方向性描述加编辑备注。
- **影响 / 后续**:AudioLM 确立了"离散化音频 token + 语言模型自回归生成"这一后续音频/音乐生成工作的主流范式,MusicLM(同年,基于 AudioLM 框架加文本条件做音乐生成)和 MusicGen 都直接受其启发;但三阶段级联结构复杂、推理慢、误差会在阶段间累积,这正是 MusicGen 要解决的问题。加跨节点链接 `→ [02-hubert.md](02-hubert.md) · 本文语义 token 提取思路的技术源头之一`、`→ [05-musicgen.md](05-musicgen.md) · 简化本文级联结构为单阶段生成的后续工作`。

- [ ] **Step 1: 重新读一遍 `18-speech-audio/02-hubert.md` 和 `18-speech-audio/03-whisper.md` 全文,确认相关描述与各自节点自述内容一致**

- [ ] **Step 2: 写节点正文,创建配图,验证 SVG 合法性**

- [ ] **Step 3: 检查五个已知坑**

- [ ] **Step 4: 提交**

```bash
git add 18-speech-audio/04-audiolm.md 18-speech-audio/assets/04-audiolm-architecture.svg
git commit -m "feat: AudioLM(2022)节点正文"
```

---

## Task 7: 节点 05 —— MusicGen(2023)

**Files:**
- Create: `18-speech-audio/05-musicgen.md`
- Create: `18-speech-audio/assets/05-musicgen-architecture.svg`(至少 1 张,**文件名必须以 `05-musicgen-` 开头**)

Frontmatter:

```yaml
---
name: "MusicGen"
year: 2023
family: "18-speech-audio"
order: 5
paper: "Simple and Controllable Music Generation"
authors: ["Jade Copet", "Felix Kreuk", "Itai Gat", "Tal Remez", "David Kant", "Gabriel Synnaeve", "Yossi Adi", "Alexandre Défossez"]
key_idea: "单阶段 Transformer decoder + EnCodec 码本交错(codebook interleaving)技巧,把多个残差量化码本流摊平成一条序列自回归生成,支持文本/旋律双重条件控制,把 AudioLM/MusicLM 的多阶段级联简化成单阶段模型"
---
```

九段式结构写作要点(**先用 WebSearch 核实真实数字再落笔**;若不可用,遵循已建立的编辑备注惯例。**写"前作进展"提到 AudioLM 时,必须重新读一遍 `18-speech-audio/04-audiolm.md` 自己的正文,确认 claim 与该节点自述内容一致——这是本家族和前几轮反复强调的规则**):

- **前作进展**:AudioLM 用三阶段级联(语义 token → 粗声学 token → 细声学 token)生成音频,证明了"离散化 + 语言建模"这套范式的可行性,但级联结构本身有代价:需要训练和维护三个独立的 Transformer decoder,推理时要依次执行三个阶段,速度慢,而且前一阶段的误差会传播、累积到后续阶段,影响最终生成质量;同期的 MusicLM 沿用了类似的级联结构给音乐生成加上文本条件(通过联合文本-音乐嵌入 MuLan),同样继承了级联结构的复杂性问题。
- **核心思想 + 直觉**:核心洞察是——不必用多个独立模型分阶段生成不同粒度的 token,只要设计一种巧妙的方式把神经编解码器输出的多层并行残差量化(RVQ)码本"摊平"成一条能被单个自回归 Transformer 直接建模的序列,就可以用一个模型、一次推理过程生成所有层级的音频 token,大幅简化系统复杂度、加快推理速度。
- **机制一(EnCodec 残差量化)**:用神经音频编解码器 EnCodec 把音频压缩成 K 层残差量化(RVQ)码本——每一帧时间步上,K 个码本并行地各自贡献一个离散 token,第一层捕捉粗粒度信息,后续层逐层用残差方式补充更精细的声学细节,K 层组合起来能重建出接近原始质量的音频。
- **机制二(码本交错 codebook interleaving)**:标准自回归 Transformer 一次只能预测一个 token,但 EnCodec 每个时间步有 K 个并行码本 token 需要生成。MusicGen 提出几种交错模式(如"延迟模式" delay pattern),把 K 个并行码本流按一定的时间错位规则重新排列成一条单一序列,让单个 decoder-only Transformer 能按固定顺序逐个预测所有码本的 token,而不需要为每层码本单独训练模型或额外增加阶段。
- **机制三(文本 + 旋律双重条件控制)**:文本条件通过预训练的 T5 文本编码器提取文本嵌入,以 cross-attention 方式注入 Transformer decoder,控制生成音乐的风格/描述内容;旋律条件则从参考音频里提取色度图(chromagram,反映音高/和声走向而非具体音色),作为额外条件输入,让模型可以按指定旋律生成不同编曲风格的音乐,两种条件可以单独或组合使用。
- **三件套协同**:只有 RVQ 编解码没有码本交错,多层并行 token 无法被单个自回归模型直接建模,只能退回 AudioLM 式的多阶段级联;只有码本交错没有 RVQ 提供的分层残差结构,交错的对象本身就不存在;只有前两者没有条件控制机制,模型只能做无条件的音频续写,不具备"按文本描述或参考旋律生成音乐"这一实用能力。三者组合起来,MusicGen 才能用单阶段模型同时实现"生成快、质量高、可控"。
- **关键代码**:EnCodec RVQ token 提取 + 延迟交错模式重排 + T5 文本条件 cross-attention 的简化伪代码,参照 `13-moe-efficient/04-deepseek-v3.md` 的详略程度。
- **性能数据**:核实真实数字——论文报告的不同模型规模(如 300M / 1.5B / 3.3B)在文本到音乐生成任务上的人工评估结果(音质、与文本描述的贴合度),以及相对 MusicLM 在自回归生成步数/推理速度上的简化幅度。不确定的具体数字写方向性描述加编辑备注。
- **影响 / 后续**:MusicGen 成为广泛使用的开源文本到音乐生成基线,证明了"码本交错"这一技巧可以把多阶段级联的音频语言模型简化成单阶段模型而不显著牺牲质量,这个思路后续也被其他音频/语音生成工作借鉴。这是本家族按教学顺序收录的最后一篇节点,完整走完"自监督表征学习(Wav2Vec2/HuBERT)→ 大规模弱监督识别(Whisper)→ 音频离散化 + 语言建模生成(AudioLM/MusicGen)"这条主线。加跨节点链接 `→ [04-audiolm.md](04-audiolm.md) · 本文简化的多阶段级联结构`。

- [ ] **Step 1: 重新读一遍 `18-speech-audio/04-audiolm.md` 全文,确认"前作进展"里对 AudioLM 的描述与该节点自述内容一致**

- [ ] **Step 2: 写节点正文,创建配图,验证 SVG 合法性**

- [ ] **Step 3: 检查五个已知坑**

- [ ] **Step 4: 提交**

```bash
git add 18-speech-audio/05-musicgen.md 18-speech-audio/assets/05-musicgen-architecture.svg
git commit -m "feat: MusicGen(2023)节点正文"
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

Expected: 无报错退出,输出类似 `wrote .../TIMELINE.md (84 nodes)` 和 `wrote .../families.json (18 families, 84 nodes)`(79 + 5 = 84)。

- [ ] **Step 2: 检查 TIMELINE.md 里新家族的 5 行是否正确插入**

```bash
grep -n "Wav2Vec 2.0\|HuBERT\|Whisper\|AudioLM\|MusicGen" TIMELINE.md
```

Expected: 5 行都出现,`\`18-speech-audio\`` 出现在对应行里。

- [ ] **Step 3: 检查 families.json 里新家族块,尤其每个节点 assets 数组非空**

```bash
python3 -c "
import json
data = json.load(open('web/src/data/families.json'))
fam = next((f for f in data['families'] if f['id'] == '18-speech-audio'), None)
assert fam is not None, '18-speech-audio 家族块缺失'
assert len(fam['nodes']) == 5, f'期望 5 个节点,实际 {len(fam[\"nodes\"])}'
assert fam['colorToken'] == '--family-18', f'colorToken 不对: {fam[\"colorToken\"]}'
for n in fam['nodes']:
    assert len(n['assets']) > 0, f'{n[\"name\"]} 的 assets 数组是空的!检查 SVG 文件名是否匹配 {n[\"path\"]} 的 stem'
print('OK', fam['label'], fam['yearRange'])
for n in fam['nodes']:
    print(' -', n['order'], n['year'], n['name'], n['assets'])
"
```

Expected: 打印 `OK 语音/音频模型(Speech/Audio Models) [2020, 2023]`,随后 5 行,每行 `assets` 数组都非空。**如果某个节点 assets 为空,说明 SVG 文件名没有匹配上该节点 markdown 的 stem,回去改文件名重新生成。**

- [ ] **Step 4: 全仓库 SVG 合法性自查(本家族范围)**

```bash
python3 -c "
import xml.etree.ElementTree as ET
import glob
files = sorted(glob.glob('18-speech-audio/assets/*.svg'))
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

Expected: tsc 无输出;vitest 全部通过,测试数量与之前(322)基本持平(本轮不新增金标本/组件测试,只有 `web/src/test/svgAssets.test.ts` 会扫到新增的 5+ 张 SVG 并新增对应测试用例)。

- [ ] **Step 6: 浏览器验证家族页面与节点详情页**

用 preview_start 起 dev server(用 `preview_logs` 确认实际绑定端口,工具报告的端口可能不准),访问 `/families/18-speech-audio` 确认:
- 家族标题、"一句话定位"正文正常渲染
- 5 个节点卡片按 Task 2 README 里的顺序显示(Wav2Vec2 → HuBERT → Whisper → AudioLM → MusicGen),颜色为浅玫瑰红 `#fb7185`
- 逐一点进 5 个节点详情页(`/families/18-speech-audio/01-wav2vec2` 等),确认走的是 `NodePage.tsx` 通用渲染路径(非金标本),标题/作者/正文/配图/前后节点导航正常显示,console 无 error
- 访问首页确认头部计数变成"18 家族 · 84 节点",侧栏家族列表和时间线可视化里都能看到"语音/音频模型(Speech/Audio Models)"及其 5 个节点

- [ ] **Step 7: 提交**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add TIMELINE.md web/src/data/families.json
git commit -m "feat: 生成 TIMELINE.md / families.json,第 18 个家族接入完成

语音/音频模型(Speech/Audio)家族全部 5 篇节点 markdown 正本 + README 就位,
python3 scripts/generate_timeline.py 重新生成产物,tsc + vitest 全项目
通过,浏览器验证家族页面与节点详情页渲染正常,首页计数更新为
18 家族 84 节点。金标本交互页按计划留待后续单独轮次补,本轮完成。"
```

---

## Self-Review 记录(写 plan 时已自查)

1. **Spec 覆盖**:spec 第 3 节(节点列表,含排序说明)→ Task 3-7,README 里同步注明排序说明;第 4 节(节点写作规范,含 SVG 命名规则)→ 每个节点 Task 的 Step 强调;第 5 节(家族 README,含手写子时间线表格要求)→ Task 2 的完整表格内容;第 6 节(注册文件,含 familyHero.ts 提前处理)→ Task 1 + Task 3;第 7 节(验收标准,含 assets 非空断言)→ Task 8 的检查脚本逐条对应;第 8 节(Out of scope)→ 全程未涉及金标本/foundations 改动。
2. **Placeholder 扫描**:每个节点任务给出的是"必须包含的真实事实清单 + 结构大纲"(内容创作类任务的等价物),不是"TBD"式占位符,延续 spec 第 4 节和前两轮已验证有效的做法。
3. **一致性检查**:5 个节点的 frontmatter `order` 字段(1-5)与 Task 3-7 顺序一致(Wav2Vec2/HuBERT/Whisper/AudioLM/MusicGen);跨节点链接指向的文件名(`01-wav2vec2.md` `02-hubert.md` `03-whisper.md` `04-audiolm.md` `05-musicgen.md`)在各任务间保持一致拼写,全部采用 `[filename.md](filename.md)` 括号语法;跨节点事实一致性检查步骤(重读前置节点原文)在 Task 5/6/7 里逐一列出,对应涉及的具体前置节点;familyHero.ts 的修复前移到 Task 3(第一个节点任务);家族 README 的"子时间线"是完整手写的 5 行真实表格,并在正文里显式注明 Whisper/AudioLM 的教学顺序例外(与 spec 第 3 节"排序说明"一致)。
