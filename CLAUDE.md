# CLAUDE.md — Daily-LLM 项目指令

## 对话开始时必须做的事

每次对话开始，必须先调用 `using-superpowers` skill，再做任何其他事情（包括回答问题、澄清需求）。

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
