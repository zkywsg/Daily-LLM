# CLAUDE.md — Daily-LLM 项目指令

## 对话开始时必须做的事

每次对话开始，必须先调用 `using-superpowers` skill，再做任何其他事情（包括回答问题、澄清需求）。

## 项目背景

- **仓库**：Daily-LLM，深度学习与大模型的双语学习知识库
- **主要语言**：中文优先，英文为辅
- **结构**：
  - `00-Timeline/` — 编年体时间线（2012–2025）
  - `00-Prerequisites/` — 神经网络基础前置
  - `01-Visual-Intelligence/` — 视觉线（2012–2017）
  - `02-Language-Transformers/` — 语言线（2013–2019）
  - `03-Scale-Multimodal/` — 汇流：规模与多模态（2020–2021）
  - `04-Alignment-OpenSource/` — 对齐与开源（2022–2023）
  - `05-Systems-Production/` — 系统与生产（2023–2025）
  - `06-Capstone-Projects/` — 实战项目

## 核心原则

### 时间线与模块双向同步
当在 `00-Timeline/README.md` 增加或修改某个条目时，必须同步更新对应模块的 README。
反之亦然：模块新增内容时，检查时间线是否需要补充对应节点。
