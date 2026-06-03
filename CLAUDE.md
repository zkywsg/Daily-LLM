# CLAUDE.md — Daily-LLM 项目指令

## 对话开始时必须做的事

每次对话开始，必须先调用 `using-superpowers` skill，再做任何其他事情（包括回答问题、澄清需求）。

## 项目背景

- **仓库**：Daily-LLM，深度学习与大模型的双语学习知识库
- **主要语言**：中文优先，英文为辅
- **结构**：
  - `timeline/` — 编年体时间线（2012–2025）
  - `foundations/` — 神经网络基础前置
  - `tracks/vision/` — 视觉线（2012–2017）
  - `tracks/language/` — 语言线（2013–2019）
  - `tracks/scale-multimodal/` — 汇流：规模与多模态（2020–2021）
  - `tracks/alignment/` — 对齐与开源（2022–2023）
  - `tracks/systems/` — 系统与生产（2023–2025）
  - `projects/` — 实战项目

## 核心原则

### 时间线与模块双向同步
当在 `timeline/README.md` 增加或修改某个条目时，必须同步更新对应模块的 README。
反之亦然：模块新增内容时，检查时间线是否需要补充对应节点。
