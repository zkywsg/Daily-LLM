# Daily-LLM 风格指南

唯一权威参考。新建或修改任何内容，对照此文件。

---
## 项目背景

- **仓库**：Daily-LLM，深度学习与大模型的双语学习知识库
- **主要语言**：中文优先，英文为辅
- **结构**：以架构家族为主轴
  - `01-cnn/` … `15-reasoning-o1-r1/` — 15 个架构家族目录（仓库根目录）
  - `foundations/` — 神经网络基础前置（9 个编号子目录：`01-neural-network-basics/` … `09-tokenization-embedding/`）
  - `TIMELINE.md` — 由 `scripts/generate_timeline.py` 自动生成的编年体时间线
  - `projects/` — 实战项目
  - `_archive/` — 历史遗留内容（只读，原 `timeline/`、`tracks/` 等）

## 核心原则

### 时间线与家族目录双向同步
时间线由 `scripts/generate_timeline.py` 从各家族目录的 frontmatter 自动生成到 `TIMELINE.md`，不要手工编辑 `TIMELINE.md`。新增或修改某个家族下的条目时，更新该条目的 frontmatter，并重新运行脚本刷新时间线。

### 网页开发固定端口
优化或调试 `web/` 前端网页时，开发服务器端口固定使用 `5173`。

启动命令统一为：

```bash
cd web
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

- 不要随意更换端口，避免浏览器调试地址变化。
- 如果 `5173` 被占用，先说明占用情况并询问是否停止旧服务或临时换端口。
- 给用户预览地址时，默认使用 `http://127.0.0.1:5173/`。

## 模块 README 结构

```
# 为什么 [旧方法] 不够用了？—— [技术名]

## 这个问题从哪来
> [年份]，[背景]，[谁注意到了什么]。

## 学习目标
完成后你应能回答：1. [问题]  2. [问题]  3. [问题]

## 1. 直觉       生活类比，3-5 句，不用公式
## 2. 机制       公式 → 计算流图（Mermaid）→ 渐进式实现
## 3. 工程陷阱   [原因] → [现象]，按优先级排序

## 演进笔记
> 这一技术的遗产：[解决了什么，遗留了什么新问题]
→ 详见 [下一相关模块链接]

---
**上一章**: [名称](路径) | **下一章**: [名称](路径)
```

渐进式实现：每步注释写"解决什么问题"，不写"加了什么功能"。

```
Step 1  5-15 行，只含核心逻辑，可独立运行
Step 2  + 边界处理（mask / padding / shape 安全）
Step 3  + 工程完善（多头 / dropout / 归一化）
Step 4  + 生产级（性能优化 / 混合精度 / 完整注解）
```

---

## 签名元素（三个，每模块必须有）

| 元素 | 位置 | 规则 |
|------|------|------|
| `这个问题从哪来` | 模块开头 | 引用年份和论文，1-3 句 |
| `你要记住` | 关键知识点后 | `> 你要记住：[一句话结论]`，全模块 ≤ 3 次 |
| `演进笔记` | 模块结尾 | 技术遗产 + 链接后续模块 |

---

## Mermaid 色彩（暖色系，全仓库统一）

| 语义 | fill | stroke | color |
|------|------|--------|-------|
| 输入 / 数据 | `#fef3c7` | `#d97706` | `#92400e` |
| 计算 / 变换 | `#fce7f3` | `#db2777` | `#9d174d` |
| 输出 / 结果 | `#ecfdf5` | `#059669` | `#065f46` |
| 问题 / 局限 | `#fff7ed` | `#ea580c` | `#9a3412` |
| 演进 / 链接 | `#eff6ff` | `#2563eb` | `#1e40af` |

所有架构图用 `graph TD`，连线用灰色 `#d6d3d1`，节点标注 tensor shape。

---

## 代码规范

三行注释头，无前缀标签，直接写内容：

```python
# 按相关性做加权聚合
# softmax(QK^T / √d_k) @ V
# 时间 O(n²d)，空间 O(n²)
def scaled_dot_product_attention(q, k, v, mask=None):
```

- 文件头：`"""模块名 · 路径 · 核心内容（1-2 句）· 依赖"""`
- Docstring Args 必须标注 shape，如 `q: (batch, heads, seq, d_k)`
- 魔法数字命名为常量，随机种子统一用 `torch.manual_seed(42)`
- 格式化：Black，行宽 88

---

## Notebook 结构（10 cells，每 cell 可独立运行）

```
Cell 1  [MD]   标题 + 一句话说验证什么
Cell 2  [Code] 导入 + seed(42) + 环境检查
Cell 3  [MD]   ## 直觉（含公式）
Cell 4  [Code] Step 1 最小实现
Cell 5  [Code] Step 2 边界处理
Cell 6  [Code] Step 3 完整实现
Cell 7  [MD]   ## 验证
Cell 8  [Code] shape 验证 / 对比期望输出
Cell 9  [MD]   ## 可视化
Cell 10 [Code] matplotlib 图
```

Notebook 只做推理和可视化，训练逻辑放 `src/`。

---

## 模块阅读顺序

```
01  machine-learning → deep-learning-basics
02  training → cnn-architectures → sequence-models
03  attention-mechanisms → transformer-architecture → pretrained-models
04  pre-training → peft → alignment → prompt-engineering → frameworks → multimodal
05  rag-foundations → vector-databases → agents
06  training-infrastructure → model-serving → deployment → monitoring
07  enterprise-rag-system → finetune-deploy-pipeline
```
