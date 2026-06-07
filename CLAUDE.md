# CLAUDE.md — Daily-LLM 项目指令

## 工作流（按风险与体量分档）

**不是每件事都走 brainstorm → spec → plan → subagent**。按下面三档决定：

### 重型流程（brainstorm → spec → plan → subagent）

仅当**满足下面任一条件**时使用：

- 新增一整套结构或规范（仓库结构 / 写作风格 / 新模板 / 横切机制）
- 一次要派 ≥ 3 个独立 subagent 做相似内容（如批量写多个家族节点）
- 决策有多个站得住脚的方向且选错代价大（颗粒度、调性、节点列表）
- 改动会影响后续 ≥ 10 个文件的写作

特点：写 spec 落盘 + 写 plan 落盘 + 派 subagent 串行执行。

### 中型流程（轻 brainstorm → 直接动手）

适合：

- 单文件较大改动（如重写一个节点）
- 一次性补丁（如新增一张 SVG、加一个章节）
- 需要在 2–3 个具体方案里挑一个

特点：开头提 1–3 个 ABCD 选项，拍板后直接派 subagent 或自己写。**默认不写 spec、不写 plan**。
例外：若涉及未来可能跨家族复用的视觉/技术惯例（如项目第一张 SVG、第一份新模板），可酌情写 spec 落盘作为惯例参考，但仍不必写 plan。

### 轻型流程（直接动手）

适合：

- 撤回意外修改、修 typo、改配置
- 跑命令、读文件、查 git 状态
- 用户明确告诉下一步做什么且方向清楚
- 小到几分钟搞定的事

特点：不开 brainstorm，不写文档，错了再改。

### Subagent 派 / 不派

**派**：
- 任务会产出 ≥ 500 字内容（节点正文、家族 README）
- 任务独立、不依赖当前会话上下文
- 需要保留主上下文继续协调

**不派（inline 做）**：
- 跑 grep / 跑测试 / 跑 TIMELINE 脚本
- 改一两行文件
- 验收检查、状态查询

**不强制 spec-reviewer + code-quality-reviewer 双 reviewer**：
- 内容任务用 grep / 字数 / 章节校验代替（在 plan 的 Task brief 里预先写好）
- 代码任务我自己读 diff
- 仅在关键决策（如金标本验收）才单独叫人 review

### "对话开始" 不必先 invoke `using-superpowers`

之前的"每次对话必须先调 using-superpowers"那条作废。我会在真正需要 `brainstorming` / `writing-plans` / `subagent-driven-development` 的时刻才 invoke 对应 skill；其他时候按上面三档自己判断。

**边界场景的兜底**：如果我对"这事该走哪一档"拿不准，先一句话问你"这个用重型/中型/轻型"，再开始。如果我判断错了，你直接说"走重型流程"或"直接动手"，我立刻切换。

## 项目背景

- **仓库**：Daily-LLM，深度学习与大模型的双语学习知识库
- **主要语言**：中文优先，英文为辅
- **结构**：
  - 仓库根下 `01-cnn/` … `15-reasoning-o1-r1/` — 15 个架构/范式家族（按登场时间排序）
  - `foundations/` — 横切基础（激活、反传、优化器、归一化、注意力机制等）
  - `TIMELINE.md` — 由 `scripts/generate_timeline.py` 自动生成的按年份速查表
  - `projects/` — 跨家族实战项目
  - `web/` — 可视化网页
  - `_archive/` — 旧 `timeline/` 与 `tracks/` 内容（仅作为家族内容重写时的素材源，请勿在此新增内容）

## 核心原则

### 单一正本原则

每个具体工作（如 ResNet）只在它所属的家族目录下有正本（如 `01-cnn/05-resnet.md`）。**不要**把同一份内容复制到多处。
跨家族引用走相对链接，前置依赖指向 `foundations/`。

### TIMELINE.md 不要手工编辑

`TIMELINE.md` 由 `scripts/generate_timeline.py` 扫描所有家族节点的 frontmatter 生成。新增/修改节点后运行：

```bash
python3 scripts/generate_timeline.py
```

### `_archive/` 是只读素材区

旧 `timeline/` 与 `tracks/` 的内容归档于此。在写某个家族的内容时，可以从中取材，但不要在 `_archive/` 内新增或修改内容。
