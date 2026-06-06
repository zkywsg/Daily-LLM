# 仓库重构 · 第 1 层（骨架 + 工具链）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把仓库从 "timeline/ + tracks/ 双正本" 切换为 "15 个家族扁平挂在根 + foundations 横切 + TIMELINE.md 自动生成" 的骨架，旧目录归档而非删除，为后续逐家族内容重写做好框架。

**Architecture:** 纯结构性 + 工具链改动。先建模板与目录骨架（每个家族 README 是从模板生成的占位），再写 TIMELINE 生成脚本并用一个真实示例节点验证它能跑通，再改根 README 和 CLAUDE.md，最后把 `timeline/` 和 `tracks/` `git mv` 到 `_archive/` 留作内容素材。本计划**不重写任何家族节点的内容**——那是后续每家族独立 plan 的事。

**Tech Stack:** Markdown · YAML frontmatter · Python 3（已有 `scripts/sync_timeline_docs.py` 同栈）· `python-frontmatter` 或手写解析 · git

**Spec reference:** `docs/superpowers/specs/2026-06-06-repo-restructure-by-architecture-families-design.md`

---

## File Structure

新增：
- `docs/templates/family-readme.md` — 家族 README 模板
- `docs/templates/node.md` — 节点 markdown 模板（含 frontmatter）
- `scripts/generate_timeline.py` — 扫描家族节点 frontmatter 生成 `TIMELINE.md`
- 仓库根 `01-cnn/` … `15-reasoning-o1-r1/` 共 15 个家族目录，每个含一个由模板生成的占位 `README.md`
- `01-cnn/02-alexnet.md` — 单个真实示例节点，用于验证 TIMELINE 生成脚本
- `TIMELINE.md` — 脚本生成，根目录速查表
- `_archive/` — 容纳旧 `timeline/` 与 `tracks/`
- `foundations/` 内新增 9 个子目录（按 spec §4.3），每个含占位 README

修改：
- `README.md` — 顶部年表替换为 15 家族总时间线表
- `CLAUDE.md` — "项目背景"段的目录列表替换为新结构
- `AGENTS.md` — 若涉及旧结构描述，对齐更新

迁移（git mv）：
- `timeline/` → `_archive/timeline/`
- `tracks/` → `_archive/tracks/`

`foundations/` 已存在的 `deep-learning/`, `math/`, `representations/`, `structures/` 暂保留为 `_archive/foundations-old/`，新 `foundations/` 子目录用占位 README 起步。

---

### Task 1: 家族 README 模板

**Files:**
- Create: `docs/templates/family-readme.md`

- [ ] **Step 1: 创建模板文件**

写入 `docs/templates/family-readme.md`：

```markdown
# {{ family_name }}

> **{{ one_line_positioning }}**

## 一句话定位

<!-- 这个家族解决了什么、被什么逼出来的。一两句话。 -->

_待补充_

## 概念本身

<!-- "什么是 XXX"——直觉解释 + 关键公式 + 一张示意图（如有）。-->

_待补充_

## 子时间线

| 年份 | 名字 | 关键贡献 | 之前卡在哪 |
|------|------|---------|-----------|
| _待补充_ |  |  |  |

## 依赖与延伸

**前置（foundations）：**
- `../foundations/xx-xxx/`（待补充）

**通向哪些家族：**
- `../NN-xxx/`（待补充）
```

- [ ] **Step 2: 提交**

```bash
git add docs/templates/family-readme.md
git commit -m "chore(restructure): add family README template"
```

---

### Task 2: 节点 markdown 模板（含 frontmatter）

**Files:**
- Create: `docs/templates/node.md`

- [ ] **Step 1: 创建模板**

写入 `docs/templates/node.md`：

```markdown
---
name: ""
year: 0
family: ""
order: 0
paper: ""
authors: []
key_idea: ""
---

# {{ name }} ({{ year }})

## 之前卡在哪

<!-- 一句话：这个工作出现前，业界被什么问题卡住。 -->

_待补充_

## 核心思想

<!-- 直觉 + 关键公式。 -->

_待补充_

## 关键代码

```python
# PyTorch 极简实现，可跑
```

## 影响 / 后续

<!-- 被谁继承、被谁取代、对哪些下游工作有直接影响。 -->

_待补充_
```

- [ ] **Step 2: 提交**

```bash
git add docs/templates/node.md
git commit -m "chore(restructure): add node markdown template with frontmatter"
```

---

### Task 3: 15 个家族目录骨架

**Files:**
- Create: 15 个目录，每个含一个由模板生成的占位 `README.md`

家族清单（按 spec §3）：

```
01-cnn                      CNN 卷积神经网络
02-rnn-lstm                 RNN / LSTM / GRU 循环网络
03-word-embedding           Word Embedding 词嵌入
04-gan                      GAN 生成对抗
05-transformer              Transformer 架构本身
06-bert-family              预训练语言模型（BERT 系）
07-gpt-scaling              大语言模型（GPT 系 + Scaling）
08-vit                      视觉 Transformer
09-multimodal-clip          多模态对齐
10-diffusion                扩散模型
11-peft-lora                参数高效微调（PEFT）
12-rlhf-alignment           对齐与 RLHF
13-moe-efficient            MoE 与高效推理
14-rag-agent                RAG 与 Agent
15-reasoning-o1-r1          推理模型（Test-time compute）
```

- [ ] **Step 1: 写一个 shell 脚本一次性建好**

在仓库根执行：

```bash
mkdir -p 01-cnn 02-rnn-lstm 03-word-embedding 04-gan 05-transformer \
         06-bert-family 07-gpt-scaling 08-vit 09-multimodal-clip 10-diffusion \
         11-peft-lora 12-rlhf-alignment 13-moe-efficient 14-rag-agent 15-reasoning-o1-r1
```

- [ ] **Step 2: 给每个目录拷一份模板 README，顶部填入家族名**

逐个目录写 `README.md`。每个文件的第一行替换为对应家族名，其他保持模板原样。例如 `01-cnn/README.md` 顶部：

```markdown
# CNN 卷积神经网络

> **_一句话定位待补充_**
```

其余 14 个目录同理，分别用：
- `02-rnn-lstm/README.md`: `# RNN / LSTM / GRU 循环网络`
- `03-word-embedding/README.md`: `# Word Embedding 词嵌入`
- `04-gan/README.md`: `# GAN 生成对抗`
- `05-transformer/README.md`: `# Transformer 架构`
- `06-bert-family/README.md`: `# 预训练语言模型（BERT 系）`
- `07-gpt-scaling/README.md`: `# 大语言模型（GPT 系 + Scaling）`
- `08-vit/README.md`: `# 视觉 Transformer (ViT)`
- `09-multimodal-clip/README.md`: `# 多模态对齐`
- `10-diffusion/README.md`: `# 扩散模型`
- `11-peft-lora/README.md`: `# 参数高效微调 (PEFT)`
- `12-rlhf-alignment/README.md`: `# 对齐与 RLHF`
- `13-moe-efficient/README.md`: `# MoE 与高效推理`
- `14-rag-agent/README.md`: `# RAG 与 Agent`
- `15-reasoning-o1-r1/README.md`: `# 推理模型（Test-time compute）`

每个文件其余内容是 `docs/templates/family-readme.md` 中 `# {{ family_name }}` 之后的全部模板内容（一句话定位、概念本身、子时间线、依赖与延伸四块都保留为"_待补充_"）。

- [ ] **Step 3: 验证**

```bash
ls -d [0-9][0-9]-* | wc -l
```

期望输出：`15`

```bash
for d in [0-9][0-9]-*; do test -f "$d/README.md" || echo "MISSING: $d"; done
```

期望输出：无（全部存在）

- [ ] **Step 4: 提交**

```bash
git add 01-cnn 02-rnn-lstm 03-word-embedding 04-gan 05-transformer \
        06-bert-family 07-gpt-scaling 08-vit 09-multimodal-clip 10-diffusion \
        11-peft-lora 12-rlhf-alignment 13-moe-efficient 14-rag-agent 15-reasoning-o1-r1
git commit -m "feat(restructure): scaffold 15 architecture family directories"
```

---

### Task 4: foundations 重切骨架

**Files:**
- 创建 9 个新子目录于 `foundations/`，每个含占位 README
- 旧 `foundations/{deep-learning,math,representations,structures}` 暂不动（Task 9 统一归档）

- [ ] **Step 1: 创建新子目录**

```bash
cd foundations
mkdir -p 01-neural-network-basics 02-activations 03-optimizers 04-normalization \
         05-initialization 06-losses 07-regularization 08-attention-mechanism \
         09-tokenization-embedding
```

- [ ] **Step 2: 给每个新子目录写占位 README**

每个 `foundations/NN-xxx/README.md` 的内容：

```markdown
# {{ topic_name }}

> 横切基础。被多个家族引用。

## 是什么

_待补充_

## 关键公式 / 直觉

_待补充_

## 被哪些家族用到

- _待补充_
```

其中 `{{ topic_name }}` 替换为对应中文名：
- `01-neural-network-basics`: `神经网络基础（神经元 · 前向 · 反向传播 · 链式法则）`
- `02-activations`: `激活函数（ReLU · GELU · Swish · Softmax）`
- `03-optimizers`: `优化器（SGD · Momentum · Adam · AdamW）`
- `04-normalization`: `归一化（BatchNorm · LayerNorm · RMSNorm）`
- `05-initialization`: `初始化（Xavier · He · 正交）`
- `06-losses`: `损失函数（CE · MSE · 对比损失 · KL）`
- `07-regularization`: `正则化（Dropout · 权重衰减 · 早停）`
- `08-attention-mechanism`: `注意力机制（通用，不绑定 Transformer）`
- `09-tokenization-embedding`: `Tokenization 与 Embedding（BPE · WordPiece · SentencePiece）`

- [ ] **Step 3: 改写 `foundations/README.md`**

把现在的 `foundations/README.md` 整体替换为：

```markdown
# foundations · 横切基础

被多个家族引用的通用模块。按"概念类别"分组，不按时间排序。

## 子模块

| 编号 | 主题 | 被哪些家族用到（典型） |
|------|------|---------------------|
| 01 | [神经网络基础](01-neural-network-basics/) | 全部 |
| 02 | [激活函数](02-activations/) | 01 CNN · 05 Transformer · 多数 |
| 03 | [优化器](03-optimizers/) | 全部 |
| 04 | [归一化](04-normalization/) | 01 CNN · 05 Transformer · 13 MoE |
| 05 | [初始化](05-initialization/) | 01 CNN · 05 Transformer |
| 06 | [损失函数](06-losses/) | 全部 |
| 07 | [正则化](07-regularization/) | 01 CNN · 多数 |
| 08 | [注意力机制](08-attention-mechanism/) | 02 RNN · 05 Transformer · 09 多模态 |
| 09 | [Tokenization & Embedding](09-tokenization-embedding/) | 03 Word Embedding · 06 BERT · 07 GPT |

> 旧版按 `deep-learning/math/representations/structures` 分组的内容已归档至 `_archive/foundations-old/`，待逐项重写后并入新子目录。
```

- [ ] **Step 4: 验证**

```bash
ls -d foundations/[0-9][0-9]-* | wc -l
```

期望输出：`9`

- [ ] **Step 5: 提交**

```bash
git add foundations/
git commit -m "feat(restructure): scaffold new foundations subdirectories"
```

---

### Task 5: TIMELINE.md 生成脚本

**Files:**
- Create: `scripts/generate_timeline.py`
- Test: `scripts/test_generate_timeline.py`

- [ ] **Step 1: 写测试（先红）**

创建 `scripts/test_generate_timeline.py`：

```python
"""测试 generate_timeline.py 能正确扫描家族节点 frontmatter 并生成 TIMELINE.md。"""
import os
import subprocess
import tempfile
import textwrap
from pathlib import Path


def write_node(dir_path: Path, filename: str, frontmatter: dict) -> None:
    lines = ["---"]
    for k, v in frontmatter.items():
        if isinstance(v, list):
            lines.append(f"{k}: {v}")
        else:
            lines.append(f"{k}: {v!r}" if isinstance(v, str) else f"{k}: {v}")
    lines.append("---")
    lines.append("")
    lines.append("# body")
    (dir_path / filename).write_text("\n".join(lines), encoding="utf-8")


def test_generate_timeline_basic(tmp_path: Path) -> None:
    # 准备一个最小仓库样子
    fam = tmp_path / "01-cnn"
    fam.mkdir()
    (fam / "README.md").write_text("# CNN\n", encoding="utf-8")
    write_node(fam, "02-alexnet.md", {
        "name": "AlexNet",
        "year": 2012,
        "family": "01-cnn",
        "order": 2,
        "paper": "ImageNet Classification with Deep CNNs",
        "key_idea": "卷积破冰",
    })
    write_node(fam, "05-resnet.md", {
        "name": "ResNet",
        "year": 2015,
        "family": "01-cnn",
        "order": 5,
        "paper": "Deep Residual Learning",
        "key_idea": "残差连接",
    })

    # 调用脚本
    script = Path(__file__).resolve().parent / "generate_timeline.py"
    result = subprocess.run(
        ["python3", str(script), "--root", str(tmp_path), "--out", str(tmp_path / "TIMELINE.md")],
        capture_output=True, text=True, check=True,
    )

    out = (tmp_path / "TIMELINE.md").read_text(encoding="utf-8")
    # 必须按年份升序列出
    assert out.index("2012") < out.index("2015")
    # 必须包含名字、家族、关键思想
    assert "AlexNet" in out
    assert "ResNet" in out
    assert "01-cnn" in out
    assert "卷积破冰" in out
    assert "残差连接" in out


def test_generate_timeline_ignores_readmes(tmp_path: Path) -> None:
    fam = tmp_path / "02-rnn-lstm"
    fam.mkdir()
    # 家族 README 不应被算作节点
    (fam / "README.md").write_text("# RNN\n", encoding="utf-8")
    write_node(fam, "01-lstm.md", {
        "name": "LSTM", "year": 1997, "family": "02-rnn-lstm",
        "order": 1, "paper": "...", "key_idea": "门控",
    })

    script = Path(__file__).resolve().parent / "generate_timeline.py"
    subprocess.run(
        ["python3", str(script), "--root", str(tmp_path), "--out", str(tmp_path / "TIMELINE.md")],
        capture_output=True, text=True, check=True,
    )
    out = (tmp_path / "TIMELINE.md").read_text(encoding="utf-8")
    assert out.count("LSTM") == 1  # 只出现一次，README 没被计入
```

- [ ] **Step 2: 运行测试确认红**

```bash
python3 -m pytest scripts/test_generate_timeline.py -v
```

期望：FAIL（脚本不存在或导入失败）

- [ ] **Step 3: 写脚本（最小实现）**

创建 `scripts/generate_timeline.py`：

```python
#!/usr/bin/env python3
"""扫描仓库根下的家族目录（NN-xxx/），收集每个节点 markdown 的 frontmatter，
生成按年份升序的 TIMELINE.md 速查表。

用法：
    python3 scripts/generate_timeline.py [--root .] [--out TIMELINE.md]
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


FAMILY_DIR_RE = re.compile(r"^\d{2}-[a-z0-9\-]+$")
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Node:
    name: str
    year: int
    family: str
    order: int
    key_idea: str
    paper: str
    path: Path


def parse_frontmatter(text: str) -> dict | None:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None
    block = m.group(1)
    data: dict = {}
    for line in block.splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, _, raw = line.partition(":")
        key = key.strip()
        raw = raw.strip()
        # 去掉外层引号
        if raw.startswith(("'", '"')) and raw.endswith(("'", '"')) and len(raw) >= 2:
            raw = raw[1:-1]
        # 尝试解析整数
        if raw.lstrip("-").isdigit():
            data[key] = int(raw)
        else:
            data[key] = raw
    return data


def collect_nodes(root: Path) -> list[Node]:
    nodes: list[Node] = []
    for fam_dir in sorted(root.iterdir()):
        if not fam_dir.is_dir() or not FAMILY_DIR_RE.match(fam_dir.name):
            continue
        for md in sorted(fam_dir.glob("*.md")):
            if md.name.lower() == "readme.md":
                continue
            fm = parse_frontmatter(md.read_text(encoding="utf-8"))
            if not fm or "year" not in fm or "name" not in fm:
                continue
            nodes.append(Node(
                name=str(fm.get("name", "")),
                year=int(fm.get("year", 0)),
                family=str(fm.get("family", fam_dir.name)),
                order=int(fm.get("order", 0)),
                key_idea=str(fm.get("key_idea", "")),
                paper=str(fm.get("paper", "")),
                path=md.relative_to(root),
            ))
        # 同时支持节点目录形式 NN-name/README.md
        for sub in sorted(fam_dir.iterdir()):
            if not sub.is_dir():
                continue
            readme = sub / "README.md"
            if not readme.exists():
                continue
            fm = parse_frontmatter(readme.read_text(encoding="utf-8"))
            if not fm or "year" not in fm or "name" not in fm:
                continue
            nodes.append(Node(
                name=str(fm.get("name", "")),
                year=int(fm.get("year", 0)),
                family=str(fm.get("family", fam_dir.name)),
                order=int(fm.get("order", 0)),
                key_idea=str(fm.get("key_idea", "")),
                paper=str(fm.get("paper", "")),
                path=readme.relative_to(root),
            ))
    return nodes


def render(nodes: list[Node]) -> str:
    nodes_sorted = sorted(nodes, key=lambda n: (n.year, n.family, n.order))
    lines = [
        "# TIMELINE",
        "",
        "> 自动生成自各家族节点的 frontmatter。**请勿手工编辑。**",
        "> 重新生成：`python3 scripts/generate_timeline.py`",
        "",
        "| 年份 | 名字 | 家族 | 关键思想 | 路径 |",
        "|------|------|------|---------|------|",
    ]
    for n in nodes_sorted:
        lines.append(
            f"| {n.year} | **{n.name}** | `{n.family}` | {n.key_idea} | [{n.path}]({n.path}) |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="仓库根（默认当前目录）")
    parser.add_argument("--out", default="TIMELINE.md", help="输出文件路径")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    nodes = collect_nodes(root)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.write_text(render(nodes), encoding="utf-8")
    print(f"wrote {out_path} ({len(nodes)} nodes)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认绿**

```bash
python3 -m pytest scripts/test_generate_timeline.py -v
```

期望：2 个测试 PASS

- [ ] **Step 5: 提交**

```bash
git add scripts/generate_timeline.py scripts/test_generate_timeline.py
git commit -m "feat(scripts): add TIMELINE.md generator from family node frontmatter"
```

---

### Task 6: 写一个真实示例节点验证脚本

**Files:**
- Create: `01-cnn/02-alexnet.md`

这一步用 AlexNet 作为唯一一个真实节点（仅 frontmatter 必填部分 + 占位正文），用来证明：(a) 模板和脚本端到端通；(b) 后续家族 plan 可以照抄这一份。**本计划不写其他节点内容。**

- [ ] **Step 1: 写节点文件**

创建 `01-cnn/02-alexnet.md`：

```markdown
---
name: "AlexNet"
year: 2012
family: "01-cnn"
order: 2
paper: "ImageNet Classification with Deep Convolutional Neural Networks"
key_idea: "卷积网络在 ImageNet 上一举把 Top-5 错误率从 26% 拉到 15.3%，宣告深度学习时代开始"
---

# AlexNet (2012)

## 之前卡在哪

视觉识别多年停在 25–26%，特征靠 SIFT/HOG 等手工设计；GPU 训练神经网络的实践罕见。

## 核心思想

_待 01-cnn 家族 plan 重写。占位。_

## 关键代码

```python
# 待 01-cnn 家族 plan 重写。占位。
```

## 影响 / 后续

启动了 CNN 黄金期：VGG → Inception → ResNet。详见同家族其他节点。
```

- [ ] **Step 2: 跑脚本生成 TIMELINE.md**

```bash
python3 scripts/generate_timeline.py
```

期望输出：`wrote /Users/.../Daily-LLM/TIMELINE.md (1 nodes)`

- [ ] **Step 3: 检查生成结果**

```bash
head -20 TIMELINE.md
```

期望：见到一行包含 `2012`、`AlexNet`、`01-cnn` 的表格行。

- [ ] **Step 4: 提交**

```bash
git add 01-cnn/02-alexnet.md TIMELINE.md
git commit -m "feat(restructure): seed AlexNet node to validate timeline generator"
```

---

### Task 7: 根 README 改头换面

**Files:**
- Modify: `README.md`

把现在 README 顶部的"按年表"替换为"15 家族总时间线表"。其他段落（License、Quick Start 等）保留，但删除指向 `timeline/` 和 `tracks/` 的链接。

- [ ] **Step 1: 读现状**

读取 `README.md` 全文，定位以下两个区段：
- `## 时间线：被逼出来的历史（2012–2025）` 下面的那张年表
- `## 模块索引` 表

- [ ] **Step 2: 替换"时间线"段**

把 `## 时间线：被逼出来的历史（2012–2025）` 段下面的整张年表替换为：

```markdown
## 时间线：15 个家族（2012–2025）

| # | 家族 | 关键年份 | 一句话定位 |
|---|------|---------|-----------|
| 01 | [CNN 卷积神经网络](01-cnn/) | 2012– | 把视觉特征从手工设计交给反向传播 |
| 02 | [RNN / LSTM / GRU](02-rnn-lstm/) | 1997, 2014– | 给神经网络装上"记忆" |
| 03 | [Word Embedding](03-word-embedding/) | 2013– | 让"词"有了分布式的语义坐标 |
| 04 | [GAN](04-gan/) | 2014– | 用对抗博弈学会"生成" |
| 05 | [Transformer](05-transformer/) | 2017 | 用纯注意力替代循环，彻底并行 |
| 06 | [BERT 系预训练](06-bert-family/) | 2018– | 双向预训练 + 微调成为 NLP 新范式 |
| 07 | [GPT 系 + Scaling](07-gpt-scaling/) | 2018–2020 | 把规模做到底，涌现 Few-shot |
| 08 | [视觉 Transformer (ViT)](08-vit/) | 2020– | Transformer 反攻视觉 |
| 09 | [多模态对齐](09-multimodal-clip/) | 2021– | 让"图"和"文"住进同一个空间 |
| 10 | [扩散模型](10-diffusion/) | 2020– | 生成的新王，从噪声到图像 |
| 11 | [PEFT / LoRA](11-peft-lora/) | 2021– | 把大模型微调成本压到普通人能玩 |
| 12 | [对齐与 RLHF](12-rlhf-alignment/) | 2022– | 把"会答"变成"答得好" |
| 13 | [MoE 与高效推理](13-moe-efficient/) | 2023– | 大模型在不变贵的前提下变更大 |
| 14 | [RAG 与 Agent](14-rag-agent/) | 2023– | 把模型接上外部世界 |
| 15 | [推理模型 (o1/R1)](15-reasoning-o1-r1/) | 2024– | 推理时多想几步，能力再上一个台阶 |

> 按年份速查：[TIMELINE.md](TIMELINE.md)（自动生成）
> 横切基础：[foundations/](foundations/)
```

- [ ] **Step 3: 删除/改写"模块索引"段**

把原 `## 模块索引` 整段替换为：

```markdown
## 仓库结构

- `01-cnn/` … `15-reasoning-o1-r1/` — 15 个架构/范式家族，按登场时间排序
- `foundations/` — 横切基础（激活、反传、优化器、归一化、注意力机制…）
- `projects/` — 跨家族实战项目
- `web/` — 可视化网页
- `TIMELINE.md` — 自动生成的按年份索引
- `_archive/` — 旧 `timeline/` 与 `tracks/` 内容，作为家族内容重写时的素材源
```

- [ ] **Step 4: 删除指向旧目录的链接**

在 README 全文搜索 `timeline/`、`tracks/`，把对它们的引用全部移除或改为指向新位置。锚点 `#timeline`、`#modules` 也一并清理。

- [ ] **Step 5: 提交**

```bash
git add README.md
git commit -m "docs(restructure): replace yearly table with 15-family timeline in root README"
```

---

### Task 8: 更新 CLAUDE.md 项目结构段

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: 替换"结构"段**

在 `CLAUDE.md` 中找到现有的 `**结构**：` 那段（列了 `timeline/`、`foundations/`、`tracks/vision/` 等），整段替换为：

```markdown
- **结构**：
  - 仓库根下 `01-cnn/` … `15-reasoning-o1-r1/` — 15 个架构/范式家族（按登场时间排序）
  - `foundations/` — 横切基础（激活、反传、优化器、归一化、注意力机制等）
  - `TIMELINE.md` — 由 `scripts/generate_timeline.py` 自动生成的按年份速查表
  - `projects/` — 跨家族实战项目
  - `web/` — 可视化网页
  - `_archive/` — 旧 `timeline/` 与 `tracks/` 内容（仅作为家族内容重写时的素材源，请勿在此新增内容）
```

- [ ] **Step 2: 替换"核心原则"段**

把原 `### 时间线与模块双向同步` 这段替换为：

```markdown
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
```

- [ ] **Step 3: 提交**

```bash
git add CLAUDE.md
git commit -m "docs(restructure): update CLAUDE.md to reflect family-based structure"
```

---

### Task 9: 归档旧 timeline/ 与 tracks/ 与 foundations 旧子目录

**Files:**
- `git mv timeline/ _archive/timeline/`
- `git mv tracks/ _archive/tracks/`
- `git mv foundations/{deep-learning,math,representations,structures} _archive/foundations-old/`

- [ ] **Step 1: 建 _archive**

```bash
mkdir -p _archive
```

- [ ] **Step 2: 归档 timeline 和 tracks**

```bash
git mv timeline _archive/timeline
git mv tracks _archive/tracks
```

- [ ] **Step 3: 归档旧 foundations 子目录**

```bash
mkdir -p _archive/foundations-old
git mv foundations/deep-learning _archive/foundations-old/deep-learning
git mv foundations/math _archive/foundations-old/math
git mv foundations/representations _archive/foundations-old/representations
git mv foundations/structures _archive/foundations-old/structures
# 旧的 foundations/README_EN.md 也归档（新 README 已在 Task 4 写好）
git mv foundations/README_EN.md _archive/foundations-old/README_EN.md
```

- [ ] **Step 4: 写 _archive/README.md**

```markdown
# _archive · 历史素材区

> **只读。** 不要在这里新增或修改内容。

包含：

- `timeline/` — 旧的"按年"目录（已被仓库根下 15 个家族目录 + `TIMELINE.md` 替代）
- `tracks/` — 旧的"按主题"目录（vision/language/scale-multimodal/alignment/systems，已被家族结构替代）
- `foundations-old/` — 旧 `foundations/` 子目录（deep-learning/math/representations/structures），等待按新 `foundations/NN-xxx/` 分类重写

后续每个家族 plan 在重写其负责的节点时，从这里取材，**完成后从本目录删除对应已被吸收的部分**。
```

- [ ] **Step 5: 验证**

```bash
ls -d _archive/* && ls -d [0-9][0-9]-* | wc -l && ls foundations/
```

期望：
- `_archive/timeline _archive/tracks _archive/foundations-old _archive/README.md`
- 仓库根的 15 家族目录数量仍为 15
- `foundations/` 下只剩新的 9 个子目录 + `README.md`

- [ ] **Step 6: 重新跑 TIMELINE 生成脚本**（确保归档不影响）

```bash
python3 scripts/generate_timeline.py
```

期望：`wrote .../TIMELINE.md (1 nodes)`（仍是 AlexNet 那 1 个节点）

- [ ] **Step 7: 提交**

```bash
git add _archive foundations TIMELINE.md
git commit -m "chore(restructure): archive legacy timeline/ tracks/ and old foundations subdirs"
```

---

### Task 10: AGENTS.md 校对

**Files:**
- Modify: `AGENTS.md`（如包含对旧结构的描述）

- [ ] **Step 1: 检查**

```bash
grep -nE "timeline/|tracks/|foundations/(deep-learning|math|representations|structures)" AGENTS.md
```

- [ ] **Step 2: 若有命中**

把命中的行改为新结构（参考 Task 8 的 CLAUDE.md 改法）；若无命中，跳过本任务的提交步骤。

- [ ] **Step 3: 提交（仅在有改动时）**

```bash
git add AGENTS.md
git commit -m "docs(restructure): align AGENTS.md with new family-based structure"
```

---

### Task 11: 端到端冒烟测试

**Files:** 无新增，仅验证

- [ ] **Step 1: 目录骨架检查**

```bash
ls -d [0-9][0-9]-* | wc -l   # 期望 15
ls -d foundations/[0-9][0-9]-* | wc -l   # 期望 9
ls _archive/ | sort   # 期望含 README.md timeline tracks foundations-old
```

- [ ] **Step 2: 脚本运行**

```bash
python3 -m pytest scripts/test_generate_timeline.py -v   # 期望 2 PASS
python3 scripts/generate_timeline.py   # 期望 wrote ... (1 nodes)
```

- [ ] **Step 3: 模板存在**

```bash
test -f docs/templates/family-readme.md && test -f docs/templates/node.md && echo OK
```

- [ ] **Step 4: 旧路径不再在根**

```bash
test ! -e timeline && test ! -e tracks && echo OK
```

- [ ] **Step 5: 文档无悬空链接**

```bash
grep -nE "\]\(tracks/|\]\(timeline/" README.md CLAUDE.md AGENTS.md || echo "no stale links"
```

期望：`no stale links`

- [ ] **Step 6: 工作树干净**

```bash
git status
```

期望：`nothing to commit, working tree clean`

---

## 完成判定（对齐 spec §8 验收标准）

执行完上述 11 个任务后应满足：

1. ✅ 仓库根 `ls` 输出 15 个 `NN-xxx/` 家族目录按编号排序
2. ✅ `timeline/` 和 `tracks/` 不再在仓库根（已归档到 `_archive/`）
3. ⏳ 每个家族目录骨架已就位（README 模板四块齐全，等待内容 plan 填充）
4. ✅ AlexNet 一个示例节点已落位，验证只有一个正本的形态
5. ✅ `foundations/` 9 子目录就位
6. ✅ 根 `README.md` 已用 15 家族表替换旧年表，`TIMELINE.md` 由脚本生成
7. ✅ `CLAUDE.md`（及 `AGENTS.md`，如适用）已更新

`⏳` 项为本 plan 显式不覆盖的部分——内容重写由后续每家族独立 plan 完成。

---

## 后续 Plan 概览（不在本计划范围内）

- **Plan 2:** `01-cnn` 内容重写（LeNet · AlexNet · VGG · Inception · ResNet · DenseNet · EfficientNet）
- **Plan 3:** `02-rnn-lstm` 内容重写
- … 每个家族一个 plan，每个 plan 完成后从 `_archive/` 删除对应已吸收的素材
- 最后一个 plan：当 `_archive/` 被清空时，整体删除 `_archive/` 目录

Web 端的对应改造作为独立 brainstorm + plan，不在本系列内。
