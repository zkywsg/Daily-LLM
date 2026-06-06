# 统一写作风格指南 · 设计

**日期**：2026-06-07
**作者**：通过 brainstorming 共同确定
**状态**：Design Approved，等待写实施计划

---

## 1. 背景与动机

仓库刚完成结构重构（spec `2026-06-06-repo-restructure-by-architecture-families-design.md`），15 个家族目录、`foundations/` 九个子目录、节点 frontmatter 契约、TIMELINE 生成脚本均已就位。但 **写作风格层面有三处脱节**：

1. **新 `docs/templates/node.md` 与旧 `AGENTS.md` 的"模块 README 结构"打架**——旧指南给的是 5 段式（学习目标 / 直觉 / 机制 / 工程陷阱 / 演进笔记），新模板是 4 段式（之前卡在哪 / 核心思想 / 关键代码 / 影响 后续）。两份指引互相矛盾。
2. **旧 `AGENTS.md` 的"模块阅读顺序"指向已废弃的 track 路径**（machine-learning → deep-learning-basics → training → cnn-architectures …），全部失效。
3. **15 家族 × 平均 6–7 节点 ≈ 90+ 节点**将由 subagent 写出，缺少统一的写作锚会让风格漂得很厉害——光是"被逼出来的历史"这种叙事框架就可能被各种解读。

本 spec 的目标：**在大批量内容生产之前，先固定一套覆盖整个项目的写作风格指南**，以保证 15 个家族读起来像一个人写的。

## 2. 核心决策

| # | 决策 | 选项 |
|---|------|-----|
| 1 | 指南覆盖范围 | C：节点 + 家族 README + foundations + 技术规范，整合成项目权威指南 |
| 2 | 行文调性 | B：保留"被逼出来的历史"调性，强制 3 项签名元素，鼓励但不强制疑问句标题 |
| 3 | 节点章节结构 | D：4 必填 + 2 可选子块，按工作复杂度按需展开 |
| 4 | 示范节点 | B：附 1 个完整写好的 AlexNet 作为金标本 |
| 5 | 文件形态 | B：拆三份（`AGENTS.md` 元信息 + `docs/writing-style.md` 写作 + `docs/tech-conventions.md` 技术） |

## 3. 文件形态（产物）

```
AGENTS.md                       项目元信息（瘦身到 < 40 行）
docs/
  writing-style.md              主战场：三层模板 + 调性 + 跨引用规范
  tech-conventions.md           技术约定：Mermaid / 代码 / frontmatter / 命名
  templates/
    family-readme.md            （已存在，本次按规范微调）
    node.md                     （已存在，本次按规范微调）
01-cnn/
  02-alexnet.md                 重写为完整示范节点（金标本）
STYLE.md                        现状保留，跳板改指向 docs/writing-style.md
```

**职责切割**：
- 写节点 / 家族 README / foundations → 只需 `docs/writing-style.md`
- 排版 Mermaid / 写 PyTorch / 填 frontmatter → 只需 `docs/tech-conventions.md`
- 想了解"这是什么项目、有什么约定" → 只需 `AGENTS.md`

## 4. `docs/writing-style.md` 的内容规范

### 4.1 节点 .md 写作规范

#### 4.1.1 章节结构（锁死）

```
# {{ name }} ({{ year }})                            [H1 中性标题]
[frontmatter 在文件最顶 ---]

## 之前卡在哪                  [必填] · 60-200 字
## 核心思想                    [必填]
  ### 直觉                     [可选 ###]
  ### 机制                     [可选 ###]
## 工程陷阱                    [可选 ##]
## 关键代码                    [必填] · 一个 fenced PyTorch 块
## 影响 / 后续                  [必填] · 必须以 "→ 链接" 结尾
```

#### 4.1.2 何时展开可选块

| 工作复杂度 | 处理 | 例 |
|---|---|---|
| 引入 1 个新概念 | "核心思想"单段，不拆 | LeNet · GRU |
| 引入 ≥ 2 个新概念 / 有特殊公式 | 拆 `### 直觉` + `### 机制` | ResNet · Transformer |
| 有著名训练/部署坑 | 加 `## 工程陷阱` | Inception · EfficientNet · BatchNorm |

#### 4.1.3 调性约束（B 方案）

**强制 3 项：**

1. **「之前卡在哪」用因果叙述**：不是"X 提出了 Y"，而是"在 X 之前，业界因为 Z 卡住；Y 第一次让 Z 不再是障碍"。一句话现象，一句话数字（如有），一句话情绪/共识。
2. **「影响 / 后续」必须显式"传球给谁"**：以 `→ [家族 NN-xxx](../NN-xxx/) · [节点名]` 或 `→ [foundations/NN-xxx](../foundations/NN-xxx/)` 结尾。
3. **公式先于代码**：核心机制以数学先呈现，PyTorch 在后。禁止"直接上代码不解释"。

**鼓励但不强制：**

- **「你要记住」一句话钩子**：全节点 0–2 次，`> 你要记住：…` blockquote。
- 疑问句标题 `# 为什么 X 不够？—— Y`：**仅在替代关系成立时使用**。默认 `# 工作名 (年份)` 中性。

#### 4.1.4 跨家族 / foundations 引用语法

正本只在一处，其他位置全部相对链接：

```markdown
本节假设你熟悉[反向传播](../foundations/01-neural-network-basics/)
和[激活函数](../foundations/02-activations/)。

ResNet 终结了 [VGG](03-vgg.md) 路线的内卷。

这个套路后来被 [05 Transformer](../05-transformer/) 借走。
```

**禁止**：节点里复制 foundations 内容。foundations 讲得不够清楚，去改 foundations，不要在节点里补丁。

#### 4.1.5 长度目标

- 1 概念引入型节点：800–1500 字（LeNet · GRU）
- 标准节点：1500–2500 字（AlexNet · VGG · DenseNet）
- 大事件节点：2500–4000 字（ResNet · Transformer · GPT-3）
- 超过 4000 字 → 升级为 `NN-name/README.md` + 配套资源子目录

### 4.2 家族 README 写作规范

家族 README ≠ 节点放大版。**它是门面 + 导航 + 概念入门**。读者是"想搞清这个家族干啥"的人，不是"想深入某个具体工作"的人。

#### 4.2.1 章节细则

```
# {{ family_name }}                              [H1，纯名字]

> {{ one_line_positioning }}                     [blockquote · 12-25 字]

## 一句话定位                  [必填 · 100-250 字]
## 概念本身                    [必填 · 300-600 字]
## 子时间线                    [必填 · 表格]
## 依赖与延伸                  [必填]
```

- **一句话定位**：1 段，这个家族解决了什么、被什么逼出来。不上公式、不上代码。
- **概念本身**：直觉解释 + 1–2 个关键公式（如有）。可选 1 张家族级 Mermaid 总览图。让没读过节点的人也能"知道这家族大概是怎么回事"。
- **子时间线**：表格列固定 `年份 / 名字 / 关键贡献 / 之前卡在哪`，名字列是相对链接。表格下方可选 80 字内承上启下小结。
- **依赖与延伸**：前置 3–6 条 foundations 链接；延伸 1–3 条通向其他家族的链接 + 一句话理由。

#### 4.2.2 与节点的语气差异

- README **不写**「你要记住」钩子（节点专属）
- README **不写**「之前卡在哪」大段叙述（顶多在"一句话定位"一句话带过）
- README **不写**「工程陷阱」（节点专属）
- README 若放 Mermaid，必须是**家族级演进图**，不能是某个具体工作的内部结构

#### 4.2.3 长度

总长 **600–1200 字**。超过即说明把节点内容塞进 README 了，拆出去。

### 4.3 foundations 子模块写作规范

foundations 性质与家族根本不同：

- 家族 = 时间线（有先后、有因果、有传承）
- foundations = 概念簇（无先后，关注"是什么 / 各种变体 / 怎么选"）

#### 4.3.1 章节细则

```
# {{ topic_name }}                                [H1，主题名]
> 横切基础。被 [家族 NN-xxx](../NN-xxx/) 等家族引用。

## 是什么                      [必填 · 200-400 字]
## 直觉 + 公式                 [必填 · 300-500 字]
## 各种变体                    [必填 · 表格 + 可选段落]
## 怎么选                      [可选 · ≤ 300 字]
## 被谁用到                    [必填 · 列表]
## 极简代码                    [可选 · ≤ 30 行]
```

- **是什么**：定位 + 为什么必须存在（"如果没有它，整个模型会怎样？"）
- **直觉 + 公式**：最关键的几个公式，配 Mermaid/ASCII（如适合）。不展开特定家族应用。
- **各种变体**：表格列 `名字 / 公式（行内 LaTeX）/ 何时用 / 谁先用了`。下方按需展开 ≤ 500 字详细对比。
- **怎么选**：实用决策树或经验法则。没强建议就略掉。
- **被谁用到**：列出引用本基础的家族 + 一句话场景。
- **极简代码**：仅在 ≤ 30 行能精炼讲清时才放。

#### 4.3.2 长度

**800–1800 字**。

#### 4.3.3 frontmatter

foundations 模块**不写** frontmatter——不参与 `TIMELINE.md` 生成。生成脚本已按 `FAMILY_DIR_RE` 跳过，无需改。

### 4.4 三层规范对照表

| 项目 | 家族 README | 节点 .md | foundations 模块 |
|------|------------|---------|-----------------|
| 主轴 | 时间线 | 单个工作 | 概念簇 |
| "之前卡在哪" | 不写 | 必填 | 不写（替换为"为什么必须存在"） |
| "你要记住" | 不写 | 可选 0–2 次 | 不写 |
| "工程陷阱" | 不写 | 可选 | 不写（去家族节点里讲） |
| 公式 | 1–2 个 | 看复杂度 | 必填，主菜 |
| 代码 | 不写 | 必填 | 可选 |
| Mermaid | 家族级演进图 | 工作内部结构 | 概念示意（如有） |
| 长度 | 600–1200 字 | 800–4000 字 | 800–1800 字 |

## 5. `docs/tech-conventions.md` 的内容规范

### 5.1 Mermaid 配色（沿用旧 AGENTS.md，全仓库统一）

| 语义 | fill | stroke | color |
|------|------|--------|-------|
| 输入 / 数据 | `#fef3c7` | `#d97706` | `#92400e` |
| 计算 / 变换 | `#fce7f3` | `#db2777` | `#9d174d` |
| 输出 / 结果 | `#ecfdf5` | `#059669` | `#065f46` |
| 问题 / 局限 | `#fff7ed` | `#ea580c` | `#9a3412` |
| 演进 / 链接 | `#eff6ff` | `#2563eb` | `#1e40af` |

- 架构图默认 `graph TD`
- 连线灰 `#d6d3d1`
- 节点上标注 tensor shape `[B, C, H, W]`
- 跨家族对比图允许用"演进 / 链接"色

### 5.2 Python / PyTorch 代码规范（沿用 + refine）

- 文件头：`"""模块名 · 路径 · 核心 1-2 句 · 关键依赖"""`
- 函数三行注释头：
  ```python
  # 按相关性做加权聚合
  # softmax(QK^T / √d_k) @ V
  # 时间 O(n²d)，空间 O(n²)
  def scaled_dot_product_attention(q, k, v, mask=None):
  ```
- Docstring 必须标注 shape：`q: (batch, heads, seq, d_k)`
- 魔法数字命名为常量，随机种子统一 `torch.manual_seed(42)`
- 格式化：Black，行宽 88

**节点内代码块的额外要求：**

- 节点 `## 关键代码` 里的 fenced block 必须**单文件可跑**（不依赖未定义 import）
- 核心机制 30 行内能讲清就别堆样板
- 上一行用 1 行注释说"这段在演示什么"

### 5.3 frontmatter 契约（节点专属）

```yaml
---
name: "工作名（英文或惯用名）"
year: 2015
family: "01-cnn"
order: 5
paper: "完整论文标题"
authors: ["He Kaiming", "Zhang Xiangyu"]
key_idea: "≤ 80 字一句话核心"
---
```

- 7 字段全部出现（值可空但不缺）
- `key_idea` ≤ 80 字（`TIMELINE.md` 显示这句）
- `order` 与文件名前缀一致
- `family` 与所在目录一致
- `year` 用首次发表年（arXiv 首版优先）

家族 README / foundations 模块**不写** frontmatter。

### 5.4 文件命名

- 家族目录：`NN-kebab-case`（`01-cnn`, `06-bert-family`）
- 节点文件：`NN-kebab-case.md`，编号与 frontmatter `order` 一致
- 节点目录形态：`NN-kebab-case/README.md`（升级版）
- foundations 子目录：`NN-kebab-case`
- 全部 ASCII 小写 + 连字符；不要下划线、不要中文文件名

### 5.5 跨家族 / foundations 引用语法

正文引用语法见 `docs/writing-style.md §4.1.4`，本文件不重复。

## 6. `AGENTS.md` 瘦身后样貌

目标 < 40 行，结构如下：

```markdown
# AGENTS.md · Daily-LLM 项目元信息

> 这里只放"我是什么项目、有什么约定"。
> 怎么写内容 → docs/writing-style.md
> Mermaid/代码/frontmatter → docs/tech-conventions.md

## 项目背景
- 仓库：Daily-LLM · 深度学习与大模型双语学习知识库
- 主要语言：中文优先，英文为辅
- 主轴：以架构家族为单位

## 仓库结构
- 01-cnn/ … 15-reasoning-o1-r1/ — 15 个家族（仓库根，按时间）
- foundations/ — 9 个横切基础子模块
- TIMELINE.md — 由 scripts/generate_timeline.py 自动生成
- projects/ — 跨家族实战
- web/ — 可视化网页
- _archive/ — 只读历史素材

## 核心原则
- 单一正本：每个工作只在它所在家族目录有正本
- TIMELINE.md 不要手工编辑
- _archive 是只读素材区
- 网页开发固定端口 5173

## 开发服务器（web）
\`\`\`bash
cd web && npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
\`\`\`

## 写作规范入口
- 三层模板与调性：docs/writing-style.md
- 技术约定：docs/tech-conventions.md
```

`STYLE.md` 现状保留（一个跳板），把指向改为 `docs/writing-style.md`。

## 7. AlexNet 示范节点（金标本）

`01-cnn/02-alexnet.md` 目前是占位。本 spec 把它重写为：

1. CNN 家族 AlexNet 节点的**最终内容正本**
2. 整个项目所有节点的**写作金标本**

后续每个家族 content plan 在 brief 中都引用："参照 `01-cnn/02-alexnet.md` 的结构与语气"。

**必须落实的样板要素：**

| 风格条款 | AlexNet 体现 |
|---|---|
| 「之前卡在哪」因果叙述 | SIFT/HOG 时代的卡点 + 数字（Top-5 ~26%）+ 共识 |
| 4 必填章节 | 之前卡在哪 / 核心思想 / 关键代码 / 影响 后续 |
| 不展开可选块 | AlexNet 引入一组耦合概念（深 CNN + ReLU + Dropout + GPU 训练），不拆 ### 直觉/机制，不加 ## 工程陷阱。本身就是"小节点示范" |
| 「你要记住」钩子 | 用 1 次（不滥用）— 锁定核心论断 |
| 公式先于代码 | 卷积/Softmax/CE 公式在前，PyTorch 在后 |
| 引用 foundations | 至少 `02-activations`（ReLU）/`07-regularization`（Dropout）/`01-neural-network-basics`（反传） |
| 「影响 / 后续」传球 | 显式 `→ [03-vgg.md](03-vgg.md)`、`→ [05-resnet.md](05-resnet.md)` |
| 中性标题 | `# AlexNet (2012)` |
| 长度 | 1500–2500 字 |
| frontmatter | 7 字段补齐（现状 6 个，加 `authors`） |

**AlexNet 重写作为本 plan 的最终验收任务**：写完才算指南落地，写不好回去补指南。

## 8. 不在本次范围内的事

- 其他节点 / 家族 README / foundations 模块的实际内容（由后续每个家族 content plan 完成）
- 双语策略细节（沿用现状）
- Web 端样式（独立改造）

## 9. 验收标准

完成后应满足：

1. ✅ `AGENTS.md` 已瘦身到 < 40 行，三个职责切割清楚
2. ✅ `docs/writing-style.md` 存在，覆盖三层模板 + 调性 + 跨引用
3. ✅ `docs/tech-conventions.md` 存在，覆盖 Mermaid / 代码 / frontmatter / 命名
4. ✅ `docs/templates/family-readme.md` 和 `docs/templates/node.md` 与新规范一致（必要时微调）
5. ✅ `STYLE.md` 跳板指向 `docs/writing-style.md`
6. ✅ `01-cnn/02-alexnet.md` 重写完成，符合所有"金标本"样板要素，TIMELINE.md 重生成后正常显示
7. ✅ AlexNet 节点能被 subagent 在不读 archive 的前提下、仅凭风格指南 + 论文知识，重新写出风格相同的另一个节点（这是验证指南是否"够约束"的真正标准——但本 spec 不强求测试，留给 01-cnn content plan）
