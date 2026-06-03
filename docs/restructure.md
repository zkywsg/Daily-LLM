# Daily-LLM 知识库重构计划

> 目的：把当前「基础知识 / 编年史 / 主题模块」三类语义不同的内容，从「全挤在时间线」的现状，拆成清晰的三层架构。

---

## 1. 背景：为什么要重构

当前结构（简化）：

```
00-Prerequisites/   16 个平铺子目录（数学 + 深度学习共性 + Transformer 时代概念混在一起）
00-Timeline/        1948 + 2012–2025 共 15 个节点（1948 与其他不同质）
01-Visual-Intelligence/
02-Language-Transformers/
03-Scale-Multimodal/
04-Alignment-OpenSource/
05-Systems-Production/
06-Capstone-Projects/
```

四个根源症状：

1. **1948 Shannon 不是「被前一代逼出来的突破」**，被强放进时间线后，与 2012 AlexNet 形成语义错位。
2. **`00-Prerequisites/` 和 `00-Timeline/` 都用 `00-` 前缀平级出现**，但一个是"工具箱"、一个是"叙事主线"，命名暗示兄弟级、语义并非。
3. **主题模块编号 01→05 暗示线性顺序**，实际上视觉/语言/规模等是「在时间线某点之后分叉出的并列主题」，不是台阶。
4. **`00-Prerequisites/` 16 个子目录平铺**，缺一层"数学 / 深度学习共性 / 表示 / 结构"的分组。

---

## 2. 目标架构：三层

```
┌────────────────────────────────────────────────────────────┐
│  Layer 0 · foundations/   基础（无时间维度，工具箱）           │
│   ├─ math/                                                   │
│   ├─ deep-learning/                                          │
│   ├─ representations/                                        │
│   └─ structures/                                             │
└────────────────────────────────────────────────────────────┘
                              ↓ 被引用
┌────────────────────────────────────────────────────────────┐
│  Layer 1 · timeline/      编年主线（2012 起 + 前史小区）       │
│   ├─ prehistory/    1948 / 1958 / 1986 / 1997 折叠区          │
│   └─ 2012/ 2013/ ... 2025/                                   │
└────────────────────────────────────────────────────────────┘
                              ↓ 展开
┌────────────────────────────────────────────────────────────┐
│  Layer 2 · tracks/        主题深挖（跨年代纵向）               │
│   ├─ vision/                                                 │
│   ├─ language/                                               │
│   ├─ scale-multimodal/                                       │
│   ├─ alignment/                                              │
│   └─ systems/                                                │
└────────────────────────────────────────────────────────────┘

projects/   实战项目（原 06-Capstone-Projects）
```

**三层各自的导航逻辑**：

| 层 | 排序方式 | 入口/读法 |
|---|---|---|
| Layer 0 foundations | 按主题树，**不按时间** | 字典式工具箱，随取随用 |
| Layer 1 timeline | 按年份升序 | 每节点 = 旧瓶颈→突破→解决→新问题，节点下挂"前置基础"和"主题深挖"链接 |
| Layer 2 tracks | 按主题深度递进 | 每模块开头标"涉及的时间线节点"，向后追溯到 L1 |

**关键约束**：
- L1 时间线节点的正文**不重复 L0 / L2 已有内容**，只做"故事"和导航。
- L2 主题模块**不复制时间线叙事**，只做"概念→实操→工程化"纵向深挖。
- L0 基础**不带年份故事**，每篇是"概念是什么 / 为什么有 / 怎么算"。

---

## 3. 目录重命名映射表

### 3.1 顶层目录

| 旧路径 | 新路径 | 说明 |
|---|---|---|
| `00-Prerequisites/` | `foundations/` | 改名，并按 4 个子领域分组（见 3.2） |
| `00-Timeline/` | `timeline/` | 去前缀，正文按 3.3 调整 |
| `01-Visual-Intelligence/` | `tracks/vision/` | 进 tracks/ 子目录，去前缀 |
| `02-Language-Transformers/` | `tracks/language/` | 同上 |
| `03-Scale-Multimodal/` | `tracks/scale-multimodal/` | 同上 |
| `04-Alignment-OpenSource/` | `tracks/alignment/` | 同上 |
| `05-Systems-Production/` | `tracks/systems/` | 同上 |
| `06-Capstone-Projects/` | `projects/` | 去前缀 |

### 3.2 Layer 0 内部分组

把 `00-Prerequisites/` 的 16 个平铺子目录归入 4 组：

| 子组 | 包含原子目录 |
|---|---|
| `foundations/math/` | `linear-algebra/`, `probability-information-theory/` |
| `foundations/deep-learning/` | `deep-learning-basics/`, `backpropagation/`, `activation-functions/`, `loss-functions/`, `normalization/`, `optimization-scheduling/`, `regularization/`, `numerical-precision/` |
| `foundations/representations/` | `embeddings/`, `tokenization/`, `softmax/` |
| `foundations/structures/` | `encoder-decoder/`, `attention-primer/`, `residual-connections/`, `inductive-bias/` |

**新增（可选）**：`foundations/math/calculus/`（矩阵微分 / 链式法则的最小子集，BP 章节会复用）。

### 3.3 Layer 1 timeline 内部调整

```
timeline/
├── README.md             主时间线总览（去掉 1948 行）
├── prehistory/           深度学习前史（折叠区，正文明确标注「非主线」）
│   ├── README.md         前史导读 + 4 个里程碑摘要
│   ├── 1948-shannon.md
│   ├── 1958-perceptron.md
│   ├── 1986-backprop.md
│   └── 1997-lstm.md
└── 2012/ ... 2025/       每年一个子目录，原样保留正文
```

每个 `timeline/<年份>/README.md` 的字段约束（与网页 `timeline.ts` 对齐）：

- `previousLimit`（旧瓶颈）
- `whatHappened`（发生了什么）
- `solved`（解决了什么）
- `newProblems`（新问题）
- `prerequisites: []`（指向 L0 路径）
- `tracks: []`（指向 L2 路径）
- `keyWorks: []`

### 3.4 Layer 2 tracks 内部对齐

每个 `tracks/<主题>/README.md` 开头新增：

```markdown
## 涉及的时间线节点
- 2012 [AlexNet](../../timeline/2012/)
- 2015 [ResNet](../../timeline/2015/)
- ...

## 前置基础
- [foundations/deep-learning/backpropagation/](../../foundations/deep-learning/backpropagation/)
- [foundations/structures/residual-connections/](../../foundations/structures/residual-connections/)
```

内部子目录原样保留，不动现有 markdown 内容。

---

## 4. 网页改造点（web/）

### 4.1 数据层 `web/src/data/timeline.ts`

- 删除 `1948` 节点
- 新增 `TimelineNode` 字段：
  - `prerequisites: { label: string; path: string }[]` —— 指向 L0
  - 现有 `relatedModules` 改名 `tracks`，明确语义是 L2
- 新增 `prehistory` 单独数组（不进入主轴渲染）

### 4.2 组件

- `TimelineAxis`：起点改 2012，主轴最左端新增一个**「⏪ 前史」入口按钮**（不算主轴节点），点击打开抽屉，列 4 个前史里程碑
- 在 `TimelineContent` 右侧栏，把原"关联模块"拆成两块：
  - **前置基础**（L0 链接）
  - **主题深挖**（L2 链接）
- Hero 上方加**三入口地图组件**：基础工具箱 / 编年主线 / 主题深挖（首屏让结构一眼可见）

### 4.3 路由

当前是纯 hash 切年份，扩展为：

- `#year=2017`（当前节点，向后兼容旧 hash `#2017`）
- `#foundation=deep-learning/backpropagation`
- `#track=vision`
- `#prehistory`

---

## 5. 迁移步骤（建议执行顺序）

每一步都可独立提交，互不阻塞。

**Step 1 · 立计划（本文件）**
- ✅ 写完 `docs/restructure.md`

**Step 2 · 网页先试方向**（低风险、易回滚）
- 在 `web/` 里：删 1948 节点 → 加前史抽屉 → Hero 加三入口地图 → 把 `relatedModules` 拆 `prerequisites + tracks`
- 这一步**不动文件系统**，只改数据和组件，先验证"三层"在视觉上是否成立。

**Step 3 · 文件系统重组（一次性大改）**
- 用 `git mv` 重命名顶层目录（保留 history）
- 在 `foundations/` 下新建 4 组子目录、把 16 个子目录搬进去
- 把 1948 Shannon 节点拆为 `timeline/prehistory/1948-shannon.md`
- 新建 `timeline/prehistory/{1958-perceptron, 1986-backprop, 1997-lstm}.md` 占位

**Step 4 · 内部链接修复**
- 全仓 grep 旧路径（`00-Prerequisites/`, `01-Visual-Intelligence/` 等），批量替换
- README.md / README_EN.md 顶层目录介绍同步
- CLAUDE.md / STYLE.md / CONTRIBUTING.md 路径同步

**Step 5 · 时间线节点正文按新字段补齐**
- 每个 `timeline/<年份>/README.md` 加 `## 前置基础` 和 `## 主题深挖` 两栏
- 每个 `tracks/<主题>/README.md` 加 `## 涉及的时间线节点` 一栏

**Step 6 · 网页同步**
- 把 timeline.ts 数据从 markdown frontmatter 或脚本同步到组件

---

## 6. 风险与回滚

| 风险 | 应对 |
|---|---|
| 大量内部链接 broken | Step 4 之前先 grep 出全部旧路径清单，做 review checklist |
| GitHub stars / 外链失效 | README 顶部加一段「重构说明」并保留 30 天的旧路径软链/重定向（创建空 README 指回新路径） |
| 协作者本地分支 conflict | Step 3 选在主分支安静时间执行，PR 标题加 `[BREAKING]` |
| 网页路由旧 hash 失效 | 兼容老格式 `#2017` → 自动等价于 `#year=2017` |

---

## 7. 明确不动的东西（边界）

- **每篇 markdown 正文内容本身不重写**，本次只做目录重组 + 顶层导航
- **每个时间线节点的核心叙事字段不变**（previousLimit/whatHappened/solved/newProblems）
- **`tracks/` 内部已有的子目录结构不动**（例：tracks/vision/cnn-architectures/ 原样保留）
- **`web/dist` / `node_modules` 等构建产物不进 git**

---

## 8. 决策清单（已确认）

| 决策 | 选择 | 来源 |
|---|---|---|
| 总体方向 | 三层架构 | 用户确认 |
| 前史安置 | 独立「前史」小区 | 用户确认 |
| 主题模块命名 | 去掉数字前缀，语义名祖取代 | 用户确认 |
| 首要交付 | 本计划文档 | 用户确认 |

---

## 9. 细节决策（已确认）

| 决策 | 选择 |
|---|---|
| `foundations/math/calculus/` | **新增**，装最小子集：链式法则 / 梯度 / 矩阵微分 / 自动微分原理（4–6 篇） |
| `tracks/scale-multimodal/` | **不拆**，内部分 `scale/` + `multimodal/` 两子区；2024+ 多模态独立成熟后再升级为顶层 track |
| `projects/` | **单独保留为「综合层」**，与 tracks/ 同级；每个项目 README 顶部打「主要涉及 tracks」标签 |
| 路径语言 | **保持英文**，所有目录名英文；每个 README 第一行用中文 H1，工具层兼容、阅读层中文 |

## 10. 升级后的 foundations 完整目录

```
foundations/
├── README.md
├── math/
│   ├── linear-algebra/
│   ├── probability-information-theory/
│   └── calculus/                    ← 新增
│       ├── chain-rule.md
│       ├── gradients-and-jacobian.md
│       ├── matrix-calculus.md
│       └── autodiff-principles.md
├── deep-learning/
│   ├── deep-learning-basics/
│   ├── backpropagation/
│   ├── activation-functions/
│   ├── loss-functions/
│   ├── normalization/
│   ├── optimization-scheduling/
│   ├── regularization/
│   └── numerical-precision/
├── representations/
│   ├── embeddings/
│   ├── tokenization/
│   └── softmax/
└── structures/
    ├── encoder-decoder/
    ├── attention-primer/
    ├── residual-connections/
    └── inductive-bias/
```

## 11. 升级后的 tracks 完整目录

```
tracks/
├── README.md
├── vision/                          ← 原 01-Visual-Intelligence/
├── language/                        ← 原 02-Language-Transformers/
├── scale-multimodal/                ← 原 03-Scale-Multimodal/
│   ├── README.md                    总览 + 两条子线索引
│   ├── scale/                       预训练 / scaling laws / frameworks
│   └── multimodal/                  CLIP / Flamingo / GPT-4V / Sora 路线
├── alignment/                       ← 原 04-Alignment-OpenSource/
└── systems/                         ← 原 05-Systems-Production/
```

## 12. 升级后的 projects 完整目录

```
projects/
├── README.md                        项目总览，按「主题组合」打标签
├── enterprise-rag-system/
│   └── README.md                    顶部：tracks = [systems, scale, alignment]
└── finetune-deploy-pipeline/
    └── README.md                    顶部：tracks = [alignment, systems]
```
