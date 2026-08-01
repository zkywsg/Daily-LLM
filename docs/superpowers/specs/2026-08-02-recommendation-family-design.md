# 第 19 个家族:推荐系统(Recommendation)· 设计

**日期**:2026-08-02
**作者**:通过 brainstorming 共同确定
**状态**:Design Approved,等待写实施计划

---

## 1. 背景与动机

仓库目前有 18 个架构家族(`01-cnn` … `18-speech-audio`),覆盖 CV / NLP / 生成模型 / PEFT / RLHF / MoE / RAG-Agent / 推理模型 / 世界模型-视频生成 / 图神经网络 / 语音-音频等主线,全部 84 个节点已配齐"金标本"交互页面。推荐系统是与 GNN 相关但独立成线的一条工业应用主线——"记忆(memorization)与泛化(generalization)的权衡""如何把用户行为序列/图结构信息融入排序模型"是这条线索的核心叙事,和纯学术研究驱动的其他家族形成对比(推荐系统这几篇论文全部来自工业界:Google/Huawei/Alibaba/Pinterest),值得单独开第 19 个家族收录。

本轮范围**只写 markdown 正本**(5 篇节点 + 1 篇家族 README + 配图 SVG),不做交互式金标本页面——延续前几轮确立的分阶段惯例:先把内容打好,金标本留到后续单独轮次按"开新一档金标本节点"的既有流程逐个补。

## 2. 核心决策

| # | 决策 | 选项 |
|---|------|-----|
| 1 | 家族目录 id | `19-recommendation` |
| 2 | 家族标题 | 推荐系统(Recommendation Systems) |
| 3 | 家族色 token | `--family-19: #f87171`(红色,延续现有 18 个 token 的色相环,色环绕回接近 01 号 `#db2777`附近但明显可区分) |
| 4 | 节点数量与排序 | 5 篇,严格按年份升序(与其他 18 个家族 order 语义一致) |
| 5 | 本轮交付物 | 只写 markdown 正本 + 配图 SVG + 家族注册,不做金标本交互页 |

## 3. 节点列表(最终版,严格按年份)

| order | 年份 | 节点 | 论文 | 一句话定位 |
|---|---|---|---|---|
| 01 | 2016 | Wide & Deep | *Wide & Deep Learning for Recommender Systems*(Cheng et al., Google, DLRS 2016) | 把线性模型(Wide,靠特征叉乘"记忆"共现规律)和深度神经网络(Deep,靠 embedding"泛化"到没见过的特征组合)联合训练成一个模型,首次系统性解决"记忆与泛化"的权衡问题 |
| 02 | 2016 | YouTube DNN | *Deep Neural Networks for YouTube Recommendations*(Covington, Adams & Sargin, Google, RecSys 2016) | 用"候选生成(candidate generation)+ 排序(ranking)"两阶段深度神经网络架构处理数亿视频规模的推荐,候选生成阶段把推荐建模成极端多分类问题,是工业界大规模深度推荐系统的奠基性架构 |
| 03 | 2017 | DeepFM | *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*(Guo et al., Huawei, IJCAI 2017) | 用因子分解机(FM)替代 Wide & Deep 里需要人工设计的特征叉乘部分,FM 和 DNN 共享同一套特征 embedding 端到端训练,不再需要特征工程就能同时建模低阶和高阶特征交互 |
| 04 | 2018 | DIN | *Deep Interest Network for Click-Through Rate Prediction*(Zhou et al., Alibaba, KDD 2018) | 用注意力机制让模型根据候选广告动态计算用户历史行为序列里每个行为的权重,解决了此前把用户兴趣压缩成单一定长向量、无法表达兴趣多样性的问题 |
| 05 | 2018 | PinSAGE | *Graph Convolutional Neural Networks for Web-Scale Recommender Systems*(Ying et al., Pinterest/Stanford, KDD 2018) | 把 GraphSAGE 的归纳式图卷积扩展到 30 亿节点、180 亿边的工业级二部图(用户-物品图),用随机游走采样 + 生产者-消费者流水线训练,是 GNN 在推荐系统里最早的大规模工业落地 |

**排序说明**:Wide & Deep(2016.06 arXiv)与 YouTube DNN(2016.09 RecSys)均为 2016 年,按公开时间 Wide & Deep 更早,故排 01、YouTube DNN 排 02。DIN(2018.06 arXiv)与 PinSAGE(2018.06 KDD,同月)按"排序模型→检索/召回模型"的教学顺序把 DIN 排在 PinSAGE 之前(与 GNN 家族此前"Whisper/AudioLM 同年按主线连贯性排序"是同一类惯例)。

## 4. 每篇节点的写作规范(复用现有 18 个家族的锁死结构,与 17/18 家族完全一致)

严格复用仓库已收敛的节点写作模板(参照 `18-speech-audio/01-wav2vec2.md` 等最近节点的实际章节结构——**扁平 `##` 二级标题**,不嵌套 `###`):

```
---
name: "..."
year: ...
family: "19-recommendation"
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

- 每篇至少配 1-2 张手绘风格 SVG 架构图,存到 `19-recommendation/assets/`,复用现有家族统一的调色板与图注格式(`*图 N:...*`)
- **SVG 文件命名必须以对应节点 markdown 文件的完整 stem 开头**(如 `01-wide-deep-architecture.svg` 对应 `01-wide-deep.md`),这是 `scripts/generate_timeline.py` 的资产匹配规则(`{file_stem}-*.svg` glob),前几轮都已验证过这条规则,本轮继续保持
- 每篇正文里的机制拆解、数字、benchmark 结果必须来自真实论文内容,不得编造——写作阶段先尝试用 WebSearch/WebFetch 查证;如果依然处于此前多轮反复确认过的基础设施故障状态,按已建立的惯例:用训练知识回忆,在文中加编辑备注注明"未经实时核实,建议核对原论文",不确定的数字宁可写方向性描述也不编造精确数字
- 跨家族引用走相对链接,前置依赖指向 `foundations/`:例如 Wide & Deep/DeepFM/DIN 应讨论与 `foundations/`(嵌入/归一化等)的关系;PinSAGE 应链接 `../17-graph-neural-networks/02-graphsage.md`(直接扩展的 GraphSAGE 架构)
- 本次要主动规避此前四轮反复踩过的坑(已沉淀为自动化测试,但写作时仍需人工过一遍,不能完全依赖测试兜底):
  1. **CommonMark 加粗定界符边界情况**——`**` 紧贴标点(引号/问号/括号)时另一侧必须是空白或标点,不能直接接普通字符
  2. **`$` 货币符号与 remark-math 冲突**——本家族可能提到"千亿级样本""数十亿参数"等规模数字,若涉及具体金额一律转义成 `\$`
  3. **跨节点链接必须用 markdown link 语法**——`→ [02-youtube-dnn.md](02-youtube-dnn.md) · ...`,不能写纯文本 `→ 02-youtube-dnn.md`
  4. **跨节点事实一致性**——写"前作进展"或"影响/后续"section 提到某个已写好的兄弟节点(包括 PinSAGE 提到 GraphSAGE 时)时,必须重新读一遍那个节点自己的正文,确认自己写的 claim 与对方自述的内容一致,不能凭训练知识里的一般印象凭空归因

## 5. 家族 README 写作规范

复用 `18-speech-audio/README.md` 的章节结构:

```
# 推荐系统(Recommendation Systems)

> {{ 一句话定位引言,blockquote }}

## 一句话定位
## 概念本身
### {{ 子概念拆解,如"记忆与泛化的权衡"这一统一视角:Wide&Deep/DeepFM 如何演化特征交互建模方式,YouTube DNN 如何确立候选生成+排序两阶段架构,DIN 如何引入注意力建模用户兴趣,PinSAGE 如何跳出"表格特征"框架改用图结构建模 }}
## 子时间线
{{ 必须手写一张真实的 5 行 4 列表格(年份|名字|关键贡献|之前卡在哪),不是留空!—— 这是 18 个既有家族全部手写的惯例,`scripts/generate_timeline.py` 的 `parse_family_readme()` 只读 H1 和 blockquote,不读这张表,本次不能再犯之前踩过的错 }}
## 依赖与延伸
```

家族 README 需要用 `**前置(foundations):**` 和 `**延伸方向:**` 两个 bold-subheader + 项目符号链接的格式写"依赖与延伸"(与 `17-graph-neural-networks/README.md`、`18-speech-audio/README.md` 一致的既有惯例),并明确解释这条主线与 `17-graph-neural-networks`(PinSAGE 直接复用的 GraphSAGE 架构)之间的关系与边界,避免读者误以为推荐系统只是"GNN 的一个应用案例"——前四篇(Wide&Deep/YouTube DNN/DeepFM/DIN)完全不涉及图结构,是表格特征 + 深度学习的独立技术路线。

## 6. 需要同步改动的注册文件

新家族要接入现有基础设施,需要改这几处(均为已有机制,非新增架构,与前几轮 Task 1 完全一致的三处):

1. `scripts/generate_timeline.py` 的 `FAMILY_IDS` 列表 —— 追加 `"19-recommendation"`
2. `web/src/types/family.ts` 的 `FamilyId` 联合类型 —— 追加 `"19-recommendation"`
3. `web/src/styles/tokens.css` —— 新增 `--family-19: #f87171; /* Recommendation 红色 */`
4. 写完全部 markdown 正本后运行 `python3 scripts/generate_timeline.py` 重新生成 `TIMELINE.md` 和 `web/src/data/families.json`(后者是脚本自动生成的产物,不手工编辑)
5. **`web/src/components/home/familyHero.ts` 的 `FAMILY_HERO` 记录**——第一个节点(`01-wide-deep.md`)写完后,在同一个任务里加上 `"19-recommendation": "19-recommendation/01-wide-deep.md",`,避免 `FamilyId` 类型加宽后 `tsc --noEmit` 报 `Record<FamilyId, string>` 缺 key 的错(这是前几轮都验证过的固定坑,继续在第一个节点任务里就处理)

金标本相关的 `web/src/components/node/golden/index.ts`、`AllGoldenSamples.smoke.test.tsx`、`ProseCompleteness.test.tsx` 本轮不动——这些节点在没有金标本条目时会自动走 `NodePage.tsx` 的通用 markdown 渲染路径,不会导致任何测试失败。`web/src/test/svgAssets.test.ts` 会自动扫到本轮新增的 5+ 张 SVG 并校验合法性。

## 7. 验收标准

- `python3 scripts/generate_timeline.py` 跑完后,`TIMELINE.md` 出现 19 个家族、且新增 5 行按年份正确插入到原有行之间(DIN/PinSAGE 按第 3 节说明的教学顺序排列,不是严格公开时间顺序)
- `web/src/data/families.json` 出现 `"id": "19-recommendation"` 家族块,5 个节点、`colorToken: "--family-19"`,且**每个节点的 `assets` 数组非空**(即 SVG 文件命名规则正确)
- 5 篇 markdown frontmatter 齐全(`name/year/family/order/paper/authors/key_idea`),`family` 字段全部为 `"19-recommendation"`,`order` 为 1-5
- 每篇正文的 9 个必填章节全部非空
- 家族 README 的"子时间线"是真实手写的 5 行表格,不是留空占位
- `npx tsc --noEmit` 与 `npx vitest run`(web/ 下)全项目通过,不因新增 `FamilyId` 联合类型成员、tokens.css 改动、或 `familyHero.ts` 缺口引入任何回归
- 浏览器验证:`/families/19-recommendation` 家族页面渲染 5 个节点卡片、`--family-19` 色值 `#f87171` 正确;5 个节点详情页图片正确加载、返回链接/上下篇导航正确、console 无 error

## 8. Out of scope(本轮明确不做)

- 不写任何 `web/src/components/node/golden/19-recommendation/*` 交互页面
- 不在首页/家族列表页做额外的视觉特殊处理(复用现有家族卡片组件即可)
- 不追加 `foundations/` 下的横切基础页面(如"因子分解机 FM"独立成页),FM 等技术点在 DeepFM 节点自己的"机制"章节里讲清楚即可,不拆共享基础页
