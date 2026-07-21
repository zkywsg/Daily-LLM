# 第 17 个家族:图神经网络(GNN)· 设计

**日期**:2026-07-22
**作者**:通过 brainstorming 共同确定
**状态**:Design Approved,等待写实施计划

---

## 1. 背景与动机

仓库目前有 16 个架构家族(`01-cnn` … `16-world-models`),覆盖 CV / NLP / 生成模型 / PEFT / RLHF / MoE / RAG-Agent / 推理模型 / 世界模型-视频生成等主线,全部 74 个节点已配齐"金标本"交互页面(除刚新增的 16-world-models 家族 6 个节点按计划暂缓)。图神经网络(Graph Neural Networks, GNN)是与 CNN/RNN/Transformer 平行的第四条经典深度学习主线——处理非欧几里得结构数据(图,而非网格/序列),分子性质预测、社交网络分析、推荐系统等场景广泛应用,值得单独开第 17 个家族收录。

本轮范围**只写 markdown 正本**(5 篇节点 + 1 篇家族 README + 配图 SVG),不做交互式金标本页面——延续上一轮(16-world-models)确立的分阶段惯例:先把内容打好,金标本留到后续单独轮次按"开新一档金标本节点"的既有流程逐个补。

## 2. 核心决策

| # | 决策 | 选项 |
|---|------|-----|
| 1 | 家族目录 id | `17-graph-neural-networks` |
| 2 | 家族标题 | 图神经网络(Graph Neural Networks) |
| 3 | 家族色 token | `--family-17: #f43f5e`(玫瑰红,延续现有 16 个 token 的色相环,与 01 号玫红 `#db2777` 区分开) |
| 4 | 节点数量与排序 | 5 篇,严格按年份升序(与其他 16 个家族 order 语义一致) |
| 5 | 本轮交付物 | 只写 markdown 正本 + 配图 SVG + 家族注册,不做金标本交互页 |

## 3. 节点列表(最终版,严格按年份)

| order | 年份 | 节点 | 论文 | 一句话定位 |
|---|---|---|---|---|
| 01 | 2017 | GCN | *Semi-Supervised Classification with Graph Convolutional Networks*(Kipf & Welling, ICLR 2017) | 把谱图卷积(Chebyshev 多项式近似图拉普拉斯)简化到一阶邻域聚合,一层 `D̃^(-1/2) Ã D̃^(-1/2) H W` 传播规则定义了"现代 GNN"这个范式起点 |
| 02 | 2017 | GraphSAGE | *Inductive Representation Learning on Large Graphs*(Hamilton, Ying & Leskovec, NeurIPS 2017) | SAmple + aggreGatE:固定大小邻域采样 + 可学习聚合函数(mean/LSTM/pooling),让 GNN 第一次能泛化到训练时没见过的节点/图(归纳式,而非 GCN 的直推式) |
| 03 | 2018 | GAT | *Graph Attention Networks*(Veličković et al., ICLR 2018) | 用可学习的 attention 权重替代 GCN 里固定的度数归一化系数,让模型隐式学会"哪个邻居更重要",不需要提前知道完整图结构做矩阵运算 |
| 04 | 2019 | GIN | *How Powerful are Graph Neural Networks?*(Xu et al., ICLR 2019) | 用 Weisfeiler-Lehman 图同构测试给 GNN 表达力定理上界:证明 mean/max 聚合(如 GraphSAGE)不如 WL test,提出 sum 聚合 + MLP 的 GIN,理论上证明达到 WL test 同等的最大可能表达力 |
| 05 | 2021 | Graphormer | *Do Transformers Really Perform Bad for Graph Representation?*(Ying et al., NeurIPS 2021) | 把标准 Transformer 搬到图上:用中心性编码 + 空间编码(最短路径距离)+ 边编码把图结构信息直接注入 attention,用全局注意力替代逐跳消息传递,OGB 大规模分子性质预测挑战赛冠军 |

**排序说明**:GCN(2016.09 arXiv / ICLR 2017)与 GraphSAGE(2017.06 arXiv / NeurIPS 2017)公开时间上 GCN 更早,故 GCN 排 01、GraphSAGE 排 02,与"同年多篇按实际发布顺序排"的既有惯例一致。

## 4. 每篇节点的写作规范(复用现有 16 个家族的锁死结构,与 16-world-models 完全一致)

严格复用仓库已收敛的节点写作模板(参照 `16-world-models/01-world-models.md` 等最近节点的实际章节结构):

```
---
name: "..."
year: ...
family: "17-graph-neural-networks"
order: ...
paper: "..."
authors: [...]
key_idea: "..."
---

# {{ name }} ({{ year }})

## 前作进展
## 核心思想:{{ 副标题 }}
### 直觉
## 机制一:{{ ... }}
## 机制二:{{ ... }}
## 机制三:{{ ... }}
## 三件套协同 — {{ ... }}
## 关键代码
## 性能数据
## 影响 / 后续
```

- 每篇至少配 1-2 张手绘风格 SVG 架构图,存到 `17-graph-neural-networks/assets/`,复用现有家族统一的调色板与图注格式(`*图 N:...*`)
- **SVG 文件命名必须以对应节点 markdown 文件的完整 stem 开头**(如 `01-gcn-architecture.svg` 对应 `01-gcn.md`),这是 `scripts/generate_timeline.py` 的资产匹配规则(`{file_stem}-*.svg` glob),16-world-models 家族 Task 4 曾因配图缩写命名(`02-vdm-architecture.svg` 而非 `02-video-diffusion-models-*.svg`)导致资产匹配失败,本轮写实施计划时要在每个 Task 里显式强调这条规则,不能再犯
- 每篇正文里的机制拆解、数字、benchmark 结果必须来自真实论文内容,不得编造——写作阶段需要先尝试用 WebSearch/WebFetch 查证(上一轮 WebSearch/WebFetch 全程处于基础设施故障状态,如果这次仍不可用,按上一轮建立的惯例:用训练知识回忆,在文中加编辑备注注明"未经实时核实,建议核对原论文",不确定的数字宁可写方向性描述也不编造精确数字)
- 跨家族引用走相对链接,前置依赖指向 `foundations/`:例如 GCN 应该讨论谱图理论/图拉普拉斯的背景;Graphormer 应链接 `../05-transformer/01-transformer.md`(直接复用 Transformer 架构)与 `../08-vit/01-vit.md`(同样是"把 Transformer 搬到新模态"的思路呼应)
- 本次要主动规避此前两轮反复踩过的三个坑(已沉淀为自动化测试,但写作时仍需人工过一遍,不能完全依赖测试兜底):
  1. **CommonMark 加粗定界符边界情况**——`**` 紧贴标点(引号/问号/括号)时另一侧必须是空白或标点,不能直接接普通字符
  2. **`$` 货币符号与 remark-math 冲突**——本家族大概率不涉及美元数字,但如果提到云算力成本等,一律转义成 `\$`
  3. **跨节点链接必须用 markdown link 语法**——`→ [02-graphsage.md](02-graphsage.md) · ...`,不能写纯文本 `→ 02-graphsage.md`
- **跨节点事实一致性**(16-world-models 家族的 Task 6/7/8 反复踩过、也反复被 review 抓到的坑):写"前作进展"或"影响/后续"section 提到某个已写好的兄弟节点时,必须重新读一遍那个节点自己的正文,确认自己写的claim 与对方自述的内容一致,不能凭训练知识里的一般印象凭空归因

## 5. 家族 README 写作规范

复用 `16-world-models/README.md` 的章节结构:

```
# 图神经网络(GNN)

> {{ 一句话定位引言,blockquote }}

## 一句话定位
## 概念本身
### {{ 子概念拆解,如"消息传递"这一统一视角:聚合 aggregate + 更新 update 两步范式如何贯穿 GCN→GraphSAGE→GAT→GIN,以及 Graphormer 如何跳出这一范式改用全局注意力 }}
## 子时间线
{{ 必须手写一张真实的 4 列表格(年份|名字|关键贡献|之前卡在哪),不是留空!—— 16-world-models 家族 Task 2 曾误以为这段会被脚本自动生成而留空,code review 发现 scripts/generate_timeline.py 的 parse_family_readme() 只读 H1 和 blockquote,根本不读这张表,是 15 个既有家族全部手写的惯例,本次不能再犯同样的错 }}
## 依赖与延伸
```

家族 README 需要明确解释这条主线与 `01-cnn`(欧式网格上的卷积 vs 图上的"卷积")、`05-transformer`(Graphormer 复用的架构)之间的关系与边界,避免读者误以为 GNN 只是 CNN 的一个变种。

## 6. 需要同步改动的注册文件

新家族要接入现有基础设施,需要改这几处(均为已有机制,非新增架构,与上一轮 Task 1 完全一致的三处):

1. `scripts/generate_timeline.py` 的 `FAMILY_IDS` 列表 —— 追加 `"17-graph-neural-networks"`
2. `web/src/types/family.ts` 的 `FamilyId` 联合类型 —— 追加 `"17-graph-neural-networks"`
3. `web/src/styles/tokens.css` —— 新增 `--family-17: #f43f5e; /* GNN 玫瑰红 */`
4. 写完全部 markdown 正本后运行 `python3 scripts/generate_timeline.py` 重新生成 `TIMELINE.md` 和 `web/src/data/families.json`(后者是脚本自动生成的产物,不手工编辑)
5. **`web/src/components/home/familyHero.ts` 的 `FAMILY_HERO` 记录**(上一轮 Task 1 遗漏、Task 3 补上的坑,这次直接在第一个节点任务里就处理,不留到 review 阶段才发现)—— 第一个节点(`01-gcn.md`)写完后,在同一个任务里加上 `"17-graph-neural-networks": "17-graph-neural-networks/01-gcn.md",`,避免 `FamilyId` 类型加宽后 `tsc --noEmit` 报 `Record<FamilyId, string>` 缺 key 的错

金标本相关的 `web/src/components/node/golden/index.ts`、`AllGoldenSamples.smoke.test.tsx`、`ProseCompleteness.test.tsx` 本轮不动——这些节点在没有金标本条目时会自动走 `NodePage.tsx` 的通用 markdown 渲染路径,不会导致任何测试失败。`web/src/test/svgAssets.test.ts` 会自动扫到本轮新增的 5+ 张 SVG 并校验合法性。

## 7. 验收标准

- `python3 scripts/generate_timeline.py` 跑完后,`TIMELINE.md` 出现 17 个家族、且新增 5 行按年份正确插入到原有行之间
- `web/src/data/families.json` 出现 `"id": "17-graph-neural-networks"` 家族块,5 个节点、`colorToken: "--family-17"`,且**每个节点的 `assets` 数组非空**(即 SVG 文件命名规则正确,不重蹈上一轮 Task 4 的覆辙)
- 5 篇 markdown frontmatter 齐全(`name/year/family/order/paper/authors/key_idea`),`family` 字段全部为 `"17-graph-neural-networks"`,`order` 为 1-5 且与年份升序一致
- 每篇正文的 9 个必填章节全部非空
- 家族 README 的"子时间线"是真实手写的 5 行表格,不是留空占位
- `npx tsc --noEmit` 与 `npx vitest run`(web/ 下)全项目通过,不因新增 `FamilyId` 联合类型成员、tokens.css 改动、或 `familyHero.ts` 缺口引入任何回归
- 浏览器验证:`/families/17-graph-neural-networks` 家族页面渲染 5 个节点卡片、`--family-17` 色值 `#f43f5e` 正确;5 个节点详情页图片正确加载、返回链接/上下篇导航正确、console 无 error

## 8. Out of scope(本轮明确不做)

- 不写任何 `web/src/components/node/golden/17-graph-neural-networks/*` 交互页面
- 不在首页/家族列表页做额外的视觉特殊处理(复用现有家族卡片组件即可)
- 不追加 `foundations/` 下的横切基础页面(如"图拉普拉斯"独立成页),GNN 特有的技术点(谱图卷积推导、WL test 等)在各节点自己的"机制"章节里讲清楚即可,不拆共享基础页
