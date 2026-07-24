# 第 18 个家族:语音/音频模型(Speech/Audio)· 设计

**日期**:2026-07-24
**作者**:通过 brainstorming 共同确定
**状态**:Design Approved,等待写实施计划

---

## 1. 背景与动机

仓库目前有 17 个架构家族(`01-cnn` … `17-graph-neural-networks`),覆盖 CV / NLP / 生成模型 / PEFT / RLHF / MoE / RAG-Agent / 推理模型 / 世界模型-视频生成 / 图神经网络等主线。语音/音频是与文本、图像并列的第三条经典输入模态,自监督语音表征(Wav2Vec 2.0)、大规模弱监督 ASR(Whisper)、把音频生成建模成语言模型(AudioLM/MusicGen)这几条线索本身就是"预训练范式如何跨模态复用"的一个很好案例,值得单独开第 18 个家族收录。

本轮范围**只写 markdown 正本**(5 篇节点 + 1 篇家族 README + 配图 SVG),不做交互式金标本页面——延续上两轮(16-world-models、17-graph-neural-networks)确立的分阶段惯例:先把内容打好,金标本留到后续单独轮次按"开新一档金标本节点"的既有流程逐个补。

## 2. 核心决策

| # | 决策 | 选项 |
|---|------|-----|
| 1 | 家族目录 id | `18-speech-audio` |
| 2 | 家族标题 | 语音/音频模型(Speech/Audio Models) |
| 3 | 家族色 token | `--family-18: #fb7185`(浅玫瑰红,延续现有 17 个 token 的色相环,与 17 号 `#f43f5e` 区分开) |
| 4 | 节点数量与排序 | 5 篇,严格按年份升序(与其他 17 个家族 order 语义一致) |
| 5 | 本轮交付物 | 只写 markdown 正本 + 配图 SVG + 家族注册,不做金标本交互页 |

## 3. 节点列表(最终版,严格按年份)

| order | 年份 | 节点 | 论文 | 一句话定位 |
|---|---|---|---|---|
| 01 | 2020 | Wav2Vec 2.0 | *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*(Baevski et al., Meta AI, NeurIPS 2020) | 用对比学习目标从原始波形自监督学习语音表征,离散化后的语音单元 + 掩码预测,让下游 ASR 只需极少标注数据就能微调 |
| 02 | 2021 | HuBERT | *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units*(Hsu et al., Meta AI, 2021) | 用离线 k-means 聚类对 MFCC/隐藏层特征生成伪标签,再做 BERT 式掩码预测,解决 Wav2Vec 2.0 对比学习目标在训练早期不稳定的问题 |
| 03 | 2022 | Whisper | *Robust Speech Recognition via Large-Scale Weak Supervision*(Radford et al., OpenAI, 2022) | 68 万小时弱监督多语言/多任务数据 + 标准 Transformer encoder-decoder,证明"数据规模碾压架构精巧"这条 scaling 经验在语音识别上同样成立 |
| 04 | 2022 | AudioLM | *AudioLM: a Language Modeling Approach to Audio Generation*(Borsos et al., Google, 2022) | 把音频离散化成语义 token(粗粒度,来自 w2v-BERT)+ 声学 token(细粒度,来自 SoundStream)两级表示,用标准语言模型做 next-token 预测生成连贯音频,不需要文本条件也能续写 |
| 05 | 2023 | MusicGen | *Simple and Controllable Music Generation*(Copet et al., Meta AI, 2023) | 单阶段 Transformer decoder + EnCodec 码本交错(codebook interleaving)技巧,把多个 RVQ 码本的音频 token 摊平成一条序列自回归生成,支持文本/旋律双重条件控制 |

**排序说明**:AudioLM(2022.09 arXiv)与 Whisper(2022.12 arXiv)均为 2022 年内发布,按 arXiv 首次公开时间 Whisper 晚于 AudioLM,但 Whisper 影响力和引用路径更贴近 HuBERT→大规模监督这条主线的直接延续,故仍按"监督 ASR 先讲完整条主线,再转向生成方向"的教学顺序把 Whisper 排在 AudioLM 之前——这是本次唯一一处不严格按公开时间、而按主线连贯性排序的例外,需要在 README 的"排序说明"里显式注明。

## 4. 每篇节点的写作规范(复用现有 17 个家族的锁死结构,与 17-graph-neural-networks 完全一致)

严格复用仓库已收敛的节点写作模板(参照 `17-graph-neural-networks/01-gcn.md` 等最近节点的实际章节结构——注意是**扁平 `##` 二级标题**,不是"核心思想"下嵌套 `###` 三级标题的旧模板):

```
---
name: "..."
year: ...
family: "18-speech-audio"
order: ...
paper: "..."
authors: [...]
key_idea: "..."
---

# {{ name }} ({{ year }})

## 前作进展
## 核心思想 + 直觉
## 机制一:{{ ... }}
## 机制二:{{ ... }}
## 机制三:{{ ... }}
## 三件套协同
## 关键代码
## 性能数据
## 影响 / 后续
```

- 每篇至少配 1-2 张手绘风格 SVG 架构图,存到 `18-speech-audio/assets/`,复用现有家族统一的调色板与图注格式(`*图 N:...*`)
- **SVG 文件命名必须以对应节点 markdown 文件的完整 stem 开头**(如 `01-wav2vec2-architecture.svg` 对应 `01-wav2vec2.md`),这是 `scripts/generate_timeline.py` 的资产匹配规则(`{file_stem}-*.svg` glob)——16-world-models 家族曾因缩写命名踩过这个坑,17-graph-neural-networks 家族已经连续两轮无踩坑,本轮继续保持
- 每篇正文里的机制拆解、数字、benchmark 结果必须来自真实论文内容,不得编造——写作阶段先尝试用 WebSearch/WebFetch 查证;如果依然处于此前多轮反复确认过的基础设施故障状态,按已建立的惯例:用训练知识回忆,在文中加编辑备注注明"未经实时核实,建议核对原论文",不确定的数字宁可写方向性描述也不编造精确数字
- 跨家族引用走相对链接,前置依赖指向 `foundations/`:例如 Wav2Vec2/HuBERT 应讨论 `foundations/08-attention-mechanism` 与自监督对比学习的关系;Whisper 应链接 `../05-transformer/01-transformer.md`(标准 encoder-decoder 架构的直接应用);AudioLM/MusicGen 应讨论与 `../07-gpt-scaling/`(自回归语言模型范式搬到音频 token 上)的呼应,以及音频离散化(codec/RVQ)与 `../10-diffusion/`(另一条音频生成路线,VQ 系列)的对比边界
- 本次要主动规避此前三轮反复踩过的坑(已沉淀为自动化测试,但写作时仍需人工过一遍,不能完全依赖测试兜底):
  1. **CommonMark 加粗定界符边界情况**——`**` 紧贴标点(引号/问号/括号)时另一侧必须是空白或标点,不能直接接普通字符
  2. **`$` 货币符号与 remark-math 冲突**——本家族大概率不涉及美元数字,但如果提到云算力成本、数据集采购成本等,一律转义成 `\$`
  3. **跨节点链接必须用 markdown link 语法**——`→ [02-hubert.md](02-hubert.md) · ...`,不能写纯文本 `→ 02-hubert.md`
  4. **跨节点事实一致性**——写"前作进展"或"影响/后续"section 提到某个已写好的兄弟节点时,必须重新读一遍那个节点自己的正文,确认自己写的 claim 与对方自述的内容一致,不能凭训练知识里的一般印象凭空归因

## 5. 家族 README 写作规范

复用 `17-graph-neural-networks/README.md` 的章节结构:

```
# 语音/音频模型(Speech/Audio Models)

> {{ 一句话定位引言,blockquote }}

## 一句话定位
## 概念本身
### {{ 子概念拆解,如"自监督表征学习"这一统一视角:Wav2Vec2/HuBERT 如何从原始波形/隐藏特征学到可迁移表示,Whisper 如何转向大规模弱监督直接端到端,AudioLM/MusicGen 如何把音频离散化后套用语言模型范式做生成——三条子线索(自监督表征 → 监督端到端 → 生成式语言建模)如何共享"先把连续音频离散化/压缩成 token 序列"这个前提 }}
## 子时间线
{{ 必须手写一张真实的 5 行 4 列表格(年份|名字|关键贡献|之前卡在哪),不是留空!—— 这是 15 个既有家族全部手写的惯例,`scripts/generate_timeline.py` 的 `parse_family_readme()` 只读 H1 和 blockquote,不读这张表,本次不能再犯之前踩过的错 }}
## 依赖与延伸
```

家族 README 需要用 `**前置(foundations):**` 和 `**延伸方向:**` 两个 bold-subheader + 项目符号链接的格式写"依赖与延伸"(与 `16-world-models/README.md`、`17-graph-neural-networks/README.md` 一致的既有惯例),并明确解释这条主线与 `05-transformer`(Whisper/AudioLM/MusicGen 复用的架构)、`10-diffusion`(另一条音频/多模态生成路线)之间的关系与边界,避免读者误以为语音模型只是"把 Transformer 套用到另一种输入"。

## 6. 需要同步改动的注册文件

新家族要接入现有基础设施,需要改这几处(均为已有机制,非新增架构,与上两轮 Task 1 完全一致的三处):

1. `scripts/generate_timeline.py` 的 `FAMILY_IDS` 列表 —— 追加 `"18-speech-audio"`
2. `web/src/types/family.ts` 的 `FamilyId` 联合类型 —— 追加 `"18-speech-audio"`
3. `web/src/styles/tokens.css` —— 新增 `--family-18: #fb7185; /* Speech/Audio 浅玫瑰红 */`
4. 写完全部 markdown 正本后运行 `python3 scripts/generate_timeline.py` 重新生成 `TIMELINE.md` 和 `web/src/data/families.json`(后者是脚本自动生成的产物,不手工编辑)
5. **`web/src/components/home/familyHero.ts` 的 `FAMILY_HERO` 记录**——第一个节点(`01-wav2vec2.md`)写完后,在同一个任务里加上 `"18-speech-audio": "18-speech-audio/01-wav2vec2.md",`,避免 `FamilyId` 类型加宽后 `tsc --noEmit` 报 `Record<FamilyId, string>` 缺 key 的错(这是前两轮都验证过的固定坑,继续在第一个节点任务里就处理)

金标本相关的 `web/src/components/node/golden/index.ts`、`AllGoldenSamples.smoke.test.tsx`、`ProseCompleteness.test.tsx` 本轮不动——这些节点在没有金标本条目时会自动走 `NodePage.tsx` 的通用 markdown 渲染路径,不会导致任何测试失败。`web/src/test/svgAssets.test.ts` 会自动扫到本轮新增的 5+ 张 SVG 并校验合法性。

## 7. 验收标准

- `python3 scripts/generate_timeline.py` 跑完后,`TIMELINE.md` 出现 18 个家族、且新增 5 行按年份正确插入到原有行之间(Whisper/AudioLM 按第 3 节说明的教学顺序排列,不是严格公开时间顺序)
- `web/src/data/families.json` 出现 `"id": "18-speech-audio"` 家族块,5 个节点、`colorToken: "--family-18"`,且**每个节点的 `assets` 数组非空**(即 SVG 文件命名规则正确)
- 5 篇 markdown frontmatter 齐全(`name/year/family/order/paper/authors/key_idea`),`family` 字段全部为 `"18-speech-audio"`,`order` 为 1-5
- 每篇正文的 9 个必填章节全部非空
- 家族 README 的"子时间线"是真实手写的 5 行表格,不是留空占位
- `npx tsc --noEmit` 与 `npx vitest run`(web/ 下)全项目通过,不因新增 `FamilyId` 联合类型成员、tokens.css 改动、或 `familyHero.ts` 缺口引入任何回归
- 浏览器验证:`/families/18-speech-audio` 家族页面渲染 5 个节点卡片、`--family-18` 色值 `#fb7185` 正确;5 个节点详情页图片正确加载、返回链接/上下篇导航正确、console 无 error

## 8. Out of scope(本轮明确不做)

- 不写任何 `web/src/components/node/golden/18-speech-audio/*` 交互页面
- 不在首页/家族列表页做额外的视觉特殊处理(复用现有家族卡片组件即可)
- 不追加 `foundations/` 下的横切基础页面(如"音频离散化/RVQ 编解码器"独立成页),音频编码相关的技术点在各节点自己的"机制"章节里讲清楚即可,不拆共享基础页
