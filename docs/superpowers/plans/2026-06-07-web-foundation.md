# Web 顶级视觉化基础设施（W1+W2+W3）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 `2026-06-07-web-foundation-design.md`：扩展 `generate_timeline.py` 输出 `families.json`，搭建 web 设计系统 + 动效底座，重写页面架构（主页 D 方案 + 家族页 + 节点页占位），保留 mini-arches 资产。

**Architecture:** 14 个串行任务，分三段。**W1**（Task 1–3）建数据基础；**W2**（Task 4–6）建设计系统 + 安装依赖 + 抽 mini-arches；**W3**（Task 7–14）拆旧 + 建路由 + 建主页/家族页/节点页占位 + 测试 + 冒烟。每个任务自包含、可独立 dispatch subagent。

**Tech Stack:** TypeScript · React 19 · Vite 7 · React Router v7 · Framer Motion 11 · D3 utilities · CSS Modules + tokens · Fontsource · Mermaid · Python 3（脚本）

**Spec reference:** `docs/superpowers/specs/2026-06-07-web-foundation-design.md`

---

## File Structure

**新建（按 Task 顺序）：**

```
scripts/generate_timeline.py            ← 扩展（Task 1）
scripts/test_generate_timeline.py        ← 扩展（Task 1）

web/src/types/family.ts                  ← Task 2
web/src/data/families.json               ← Task 1 脚本生成（也提交）

web/package.json                         ← Task 3 修改

web/src/styles/tokens.css                ← Task 4
web/src/styles/fonts.css                 ← Task 4
web/src/styles/global.css                ← Task 4

web/src/lib/motion.ts                    ← Task 5
web/src/lib/motion.test.ts               ← Task 5
web/src/lib/colors.ts                    ← Task 5
web/src/lib/colors.test.ts               ← Task 5

web/src/components/mini-arches/MiniLeNet.tsx           ← Task 6
web/src/components/mini-arches/MiniAlexNet.tsx         ← Task 6
web/src/components/mini-arches/MiniVGG.tsx             ← Task 6
web/src/components/mini-arches/MiniGoogLeNet.tsx       ← Task 6
web/src/components/mini-arches/MiniResNet.tsx          ← Task 6
web/src/components/mini-arches/MiniDenseNet.tsx        ← Task 6
web/src/components/mini-arches/MiniSENet.tsx           ← Task 6
web/src/components/mini-arches/MiniEfficientNet.tsx    ← Task 6（新画或留 placeholder）
web/src/components/mini-arches/MiniConvNeXt.tsx        ← Task 6（新画或留 placeholder）
web/src/components/mini-arches/index.ts                ← Task 6 barrel export

web/src/App.tsx                          ← Task 8 完全重写
web/src/components/ui/Layout.tsx         ← Task 8
web/src/components/ui/NotFoundPage.tsx   ← Task 8

web/src/components/node/MarkdownRenderer.tsx  ← Task 9
web/src/components/node/MermaidBlock.tsx      ← Task 9

web/src/components/home/HomePage.tsx          ← Task 10
web/src/components/home/TimeAxisView.tsx      ← Task 10
web/src/components/home/FamilyGridView.tsx    ← Task 10
web/src/components/home/NodeHoverCard.tsx     ← Task 10

web/src/components/family/FamilyPage.tsx      ← Task 11

web/src/components/node/NodePage.tsx          ← Task 12

web/src/App.test.tsx                     ← Task 13 完全重写
web/src/components/home/HomePage.test.tsx ← Task 13
web/src/components/family/FamilyPage.test.tsx ← Task 13
web/src/components/node/NodePage.test.tsx ← Task 13
```

**删除（Task 7）：**

```
web/src/components/TimelineAxis.tsx
web/src/components/TimelineAxis.test.tsx
web/src/components/TimelineContent.tsx
web/src/components/TimelineWorkList.tsx
web/src/components/RelatedModules.tsx
web/src/components/ArchitectureMap.tsx
web/src/components/TimelineIllustration.tsx
web/src/components/PrehistoryDrawer.tsx
web/src/components/TrackView.tsx
web/src/components/tracks/CnnTrack.tsx    ← Task 6 之后才能删
web/src/components/tracks/                ← 整个目录
web/src/data/timeline.ts
web/src/data/timeline.test.ts
web/src/data/phaseFamily.ts
web/src/App.test.tsx                      ← Task 13 重写
```

---

### Task 1: 扩展 `generate_timeline.py` 输出 `families.json`

**Files:**
- Modify: `scripts/generate_timeline.py`
- Modify: `scripts/test_generate_timeline.py`
- Create: `web/src/data/families.json`（脚本生成）

#### Brief

现有 `scripts/generate_timeline.py` 扫描家族节点 frontmatter 生成 `TIMELINE.md`。本任务**保留**该行为，**新增**输出 `web/src/data/families.json`。

JSON schema（见 spec §4.1）：

```typescript
{
  generatedAt: string;
  families: Array<{
    id: string;             // "01-cnn"
    label: string;          // "CNN 卷积神经网络"
    blurb: string;          // 一句话定位
    yearRange: [number, number] | null;
    colorToken: string;     // "--family-01"
    nodes: Array<{
      name, year, family, order, paper, authors, key_idea, path, assets
    }>;
  }>;
}
```

15 个家族 ID（按 spec §4.1 顺序）：
`01-cnn`, `02-rnn-lstm`, `03-word-embedding`, `04-gan`, `05-transformer`, `06-bert-family`, `07-gpt-scaling`, `08-vit`, `09-multimodal-clip`, `10-diffusion`, `11-peft-lora`, `12-rlhf-alignment`, `13-moe-efficient`, `14-rag-agent`, `15-reasoning-o1-r1`.

未填家族（除 01-cnn 外）：`label` 用 README H1 行（已有占位）、`blurb` 用 `_待补充_` 替代为 `"（待补充）"`、`yearRange` 为 `null`、`nodes` 为 `[]`、`colorToken` 仍按 id 分配（`--family-NN`）。

每节点的 `assets` 字段：扫描 `<family>/assets/<NN>-<node-slug>-*.svg`（按 frontmatter `order` 和文件名规则）。

#### Steps

- [ ] **Step 1**: 写测试（先红）

修改 `scripts/test_generate_timeline.py`，添加：

```python
import json

def test_generate_families_json(tmp_path: Path) -> None:
    # 准备一个最小仓库：1 个真实家族 + 1 个空家族
    fam_cnn = tmp_path / "01-cnn"
    fam_cnn.mkdir()
    (fam_cnn / "README.md").write_text(
        "# CNN 卷积神经网络\n\n> **测试一句话定位**\n", encoding="utf-8"
    )
    write_node(fam_cnn, "02-alexnet.md", {
        "name": "AlexNet",
        "year": 2012,
        "family": "01-cnn",
        "order": 2,
        "paper": "ImageNet Classification with Deep CNNs",
        "key_idea": "深 CNN 第一次跑赢手工特征",
    })
    assets = fam_cnn / "assets"
    assets.mkdir()
    (assets / "02-alexnet-arch.svg").write_text("<svg></svg>", encoding="utf-8")

    fam_empty = tmp_path / "02-rnn-lstm"
    fam_empty.mkdir()
    (fam_empty / "README.md").write_text("# RNN / LSTM / GRU 循环网络\n", encoding="utf-8")

    out_md = tmp_path / "TIMELINE.md"
    out_json = tmp_path / "web/src/data/families.json"

    script = Path(__file__).resolve().parent / "generate_timeline.py"
    subprocess.run(
        ["python3", str(script), "--root", str(tmp_path),
         "--out", str(out_md), "--families-out", str(out_json)],
        capture_output=True, text=True, check=True,
    )

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "generatedAt" in data
    assert isinstance(data["families"], list)

    by_id = {f["id"]: f for f in data["families"]}

    # 01-cnn: 真实节点
    cnn = by_id["01-cnn"]
    assert cnn["label"] == "CNN 卷积神经网络"
    assert cnn["blurb"] == "测试一句话定位"
    assert cnn["yearRange"] == [2012, 2012]
    assert cnn["colorToken"] == "--family-01"
    assert len(cnn["nodes"]) == 1
    node = cnn["nodes"][0]
    assert node["name"] == "AlexNet"
    assert node["year"] == 2012
    assert node["path"] == "01-cnn/02-alexnet.md"
    assert "01-cnn/assets/02-alexnet-arch.svg" in node["assets"]

    # 02-rnn-lstm: 空家族
    rnn = by_id["02-rnn-lstm"]
    assert rnn["label"] == "RNN / LSTM / GRU 循环网络"
    assert rnn["yearRange"] is None
    assert rnn["nodes"] == []
    assert rnn["colorToken"] == "--family-02"
```

注意：fixture 里只有 2 个家族，但生成的 `families.json` 应该有 15 个（其余 13 个家族目录在 fixture tmp_path 里不存在，但脚本可能要按"所有 15 个 ID 都列出"还是"只列存在的"——本测试只要求**存在的家族被正确处理**；脚本可以为不存在的目录跳过或填空，我们用 `by_id` 取值即可）。

- [ ] **Step 2**: 跑测试确认红

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
python3 -m pytest scripts/test_generate_timeline.py::test_generate_families_json -v
```

期望：FAIL（脚本不支持 `--families-out` 参数）

- [ ] **Step 3**: 扩展脚本

修改 `scripts/generate_timeline.py`：

a. 在 `import` 区加 `import json` 和 `from datetime import datetime, timezone`

b. 定义家族 ID 列表常量：

```python
FAMILY_IDS = [
    "01-cnn", "02-rnn-lstm", "03-word-embedding", "04-gan",
    "05-transformer", "06-bert-family", "07-gpt-scaling",
    "08-vit", "09-multimodal-clip", "10-diffusion",
    "11-peft-lora", "12-rlhf-alignment", "13-moe-efficient",
    "14-rag-agent", "15-reasoning-o1-r1",
]
```

c. 添加 `collect_families` 函数：

```python
def parse_family_readme(readme_path: Path) -> tuple[str, str]:
    """从家族 README.md 提取 H1 label 和 blockquote blurb。"""
    if not readme_path.exists():
        return "", ""
    text = readme_path.read_text(encoding="utf-8")
    label = ""
    blurb = ""
    for line in text.splitlines():
        line_stripped = line.strip()
        if not label and line_stripped.startswith("# "):
            label = line_stripped[2:].strip()
        if not blurb and line_stripped.startswith("> "):
            raw = line_stripped[2:].strip()
            # 去掉 **...**, _..._, {{ }} 模板字符
            raw = re.sub(r"\*\*([^*]+)\*\*", r"\1", raw)
            raw = re.sub(r"\{\{[^}]+\}\}", "", raw).strip()
            if raw and raw != "_one_line_positioning_":
                blurb = raw
        if label and blurb:
            break
    return label, blurb


def collect_families(root: Path) -> list[dict]:
    families = []
    nodes_by_family: dict[str, list[Node]] = {}
    for node in collect_nodes(root):
        nodes_by_family.setdefault(node.family, []).append(node)

    for fid in FAMILY_IDS:
        fam_dir = root / fid
        readme = fam_dir / "README.md"
        label, blurb = parse_family_readme(readme)
        if not label:
            label = fid  # 保底
        if not blurb:
            blurb = "（待补充）"

        nodes = sorted(nodes_by_family.get(fid, []), key=lambda n: n.order)
        # 节点附带 assets：扫 fam_dir/assets/<NN>-<slug>-*.svg
        assets_dir = fam_dir / "assets"
        node_dicts = []
        for n in nodes:
            file_stem = n.path.stem  # "02-alexnet"
            assets = []
            if assets_dir.exists():
                for svg in sorted(assets_dir.glob(f"{file_stem}-*.svg")):
                    assets.append(str(svg.relative_to(root)))
            node_dicts.append({
                "name": n.name,
                "year": n.year,
                "family": n.family,
                "order": n.order,
                "paper": n.paper,
                "authors": [],  # 注意：当前 parser 没存 authors，需要扩
                "key_idea": n.key_idea,
                "path": str(n.path),
                "assets": assets,
            })

        years = [n.year for n in nodes]
        year_range = [min(years), max(years)] if years else None

        family_num = fid.split("-")[0]  # "01"
        families.append({
            "id": fid,
            "label": label,
            "blurb": blurb,
            "yearRange": year_range,
            "colorToken": f"--family-{family_num}",
            "nodes": node_dicts,
        })
    return families
```

d. 修改 `Node` dataclass 加 `authors` 字段并在 `collect_nodes` 里解析：

在 `@dataclass\nclass Node:` 加 `authors: list[str]`（初始 `[]`）；
在 `collect_nodes` 内构造 Node 时：

```python
authors_raw = fm.get("authors", [])
if isinstance(authors_raw, str):
    # frontmatter parser 可能把 list 当字符串读 — 简单解析
    authors_raw = authors_raw.strip()
    if authors_raw.startswith("[") and authors_raw.endswith("]"):
        authors_raw = [
            a.strip().strip('"').strip("'")
            for a in authors_raw[1:-1].split(",")
            if a.strip()
        ]
    else:
        authors_raw = []
nodes.append(Node(
    ...,
    authors=authors_raw if isinstance(authors_raw, list) else [],
    ...,
))
```

注意 `parse_frontmatter` 当前对 list 类型字段（如 `authors: [...]`）按字符串原样存到 `fm[key]`。需要在 collect_nodes 里识别并拆分。也可以改进 parser，但本任务保守只在 collect_nodes 里处理。

e. 扩展 `main()` 接受 `--families-out` 参数：

```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="TIMELINE.md")
    parser.add_argument("--families-out", default="web/src/data/families.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()

    # TIMELINE.md
    nodes = collect_nodes(root)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.write_text(render(nodes), encoding="utf-8")
    print(f"wrote {out_path} ({len(nodes)} nodes)")

    # families.json
    families = collect_families(root)
    families_path = Path(args.families_out)
    if not families_path.is_absolute():
        families_path = root / families_path
    families_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "families": families,
    }
    # 头注释
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    families_path.write_text(json_text, encoding="utf-8")
    n_nodes = sum(len(f["nodes"]) for f in families)
    print(f"wrote {families_path} ({len(families)} families, {n_nodes} nodes)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4**: 跑测试确认绿

```bash
python3 -m pytest scripts/test_generate_timeline.py -v
```

期望：3 个测试全 PASS（含新加的 `test_generate_families_json`）

- [ ] **Step 5**: 真实运行，确认 families.json 产出正常

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
python3 scripts/generate_timeline.py
```

期望输出含：

```
wrote .../TIMELINE.md (8 nodes)
wrote .../web/src/data/families.json (15 families, 8 nodes)
```

确认 `web/src/data/families.json` 存在，可读：

```bash
python3 -c "import json; d=json.load(open('web/src/data/families.json')); print('families:', len(d['families'])); print('01-cnn nodes:', len(d['families'][0]['nodes']))"
```

期望：`families: 15`, `01-cnn nodes: 8`

- [ ] **Step 6**: 提交

```bash
git add scripts/generate_timeline.py scripts/test_generate_timeline.py web/src/data/families.json
git commit -m "feat(scripts): generate web/families.json alongside TIMELINE.md"
```

---

### Task 2: 创建 `web/src/types/family.ts`

**Files:**
- Create: `web/src/types/family.ts`

#### Brief

定义 TS 类型，与 Task 1 生成的 `families.json` schema 一一对应。

#### Steps

- [ ] **Step 1**: 创建文件

写入 `web/src/types/family.ts`：

```typescript
// AUTO-CONSUMED. Schema must match scripts/generate_timeline.py output.

export type FamilyId =
  | "01-cnn"
  | "02-rnn-lstm"
  | "03-word-embedding"
  | "04-gan"
  | "05-transformer"
  | "06-bert-family"
  | "07-gpt-scaling"
  | "08-vit"
  | "09-multimodal-clip"
  | "10-diffusion"
  | "11-peft-lora"
  | "12-rlhf-alignment"
  | "13-moe-efficient"
  | "14-rag-agent"
  | "15-reasoning-o1-r1";

export interface NodeData {
  name: string;
  year: number;
  family: FamilyId;
  order: number;
  paper: string;
  authors: string[];
  key_idea: string;
  path: string; // repo-root-relative, e.g. "01-cnn/05-resnet.md"
  assets: string[]; // repo-root-relative SVG paths
}

export interface FamilyData {
  id: FamilyId;
  label: string;
  blurb: string;
  yearRange: [number, number] | null;
  colorToken: string; // "--family-NN"
  nodes: NodeData[]; // sorted by order asc
}

export interface FamiliesData {
  generatedAt: string;
  families: FamilyData[];
}
```

- [ ] **Step 2**: 验证 TS 编译

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npx tsc --noEmit
```

期望：无报错（也可能旧的 timeline.ts 引用未删除导致报错——本任务先确保 family.ts 自身合法。如果旧文件干扰，跳过到 Task 7 后再回头跑全量 tsc）

- [ ] **Step 3**: 提交

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add web/src/types/family.ts
git commit -m "feat(web): add Family / Node TypeScript types matching families.json schema"
```

---

### Task 3: 安装新依赖

**Files:**
- Modify: `web/package.json`
- Modify: `web/package-lock.json`

#### Brief

按 spec §7.1 加入 12 个 production deps + 3 个 dev deps。

#### Steps

- [ ] **Step 1**: 安装

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npm install --save \
  react-router@^7 \
  framer-motion@^11 \
  lucide-react@latest \
  d3-scale@^4 \
  d3-shape@^3 \
  d3-axis@^3 \
  @fontsource/inter@^5 \
  @fontsource/noto-sans-sc@^5 \
  @fontsource/jetbrains-mono@^5 \
  @fontsource/source-serif-pro@^5 \
  mermaid@^11
```

```bash
npm install --save-dev \
  @types/d3-scale@^4 \
  @types/d3-shape@^3 \
  @types/d3-axis@^3
```

- [ ] **Step 2**: 验证

```bash
npx tsc --noEmit 2>&1 | grep -E "Cannot find module|TS2307" | head -5
```

期望：无关于 `react-router`、`framer-motion` 等 module 的报错（**注意**：旧的 timeline.ts 引用 `phaseFamily` 之类的报错可以忽略，那是 Task 7 删的）

```bash
npm list react-router framer-motion mermaid 2>&1 | head -10
```

期望：每个包都列出版本号

- [ ] **Step 3**: 提交

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add web/package.json web/package-lock.json
git commit -m "build(web): add deps for router / motion / d3 / fonts / mermaid"
```

---

### Task 4: 设计系统 — tokens / fonts / global

**Files:**
- Create: `web/src/styles/tokens.css`
- Create: `web/src/styles/fonts.css`
- Create: `web/src/styles/global.css`

#### Brief

落实 spec §6.1（tokens）+ §6.2（字体）+ 全局 reset。

#### Steps

- [ ] **Step 1**: 创建 `web/src/styles/tokens.css`

把 spec §6.1 的完整 `:root` 和 `[data-theme="dark"]` 块**逐字符**写入。完整内容见 spec §6.1。

- [ ] **Step 2**: 创建 `web/src/styles/fonts.css`

```css
/* Self-hosted via Fontsource */
@import "@fontsource/inter/400.css";
@import "@fontsource/inter/500.css";
@import "@fontsource/inter/600.css";
@import "@fontsource/inter/700.css";
@import "@fontsource/noto-sans-sc/400.css";
@import "@fontsource/noto-sans-sc/500.css";
@import "@fontsource/noto-sans-sc/700.css";
@import "@fontsource/jetbrains-mono/400.css";
@import "@fontsource/jetbrains-mono/500.css";
@import "@fontsource/source-serif-pro/400.css";
@import "@fontsource/source-serif-pro/600.css";
```

- [ ] **Step 3**: 创建 `web/src/styles/global.css`

```css
@import "./fonts.css";
@import "./tokens.css";

*,
*::before,
*::after {
  box-sizing: border-box;
}

html,
body {
  margin: 0;
  padding: 0;
}

body {
  font-family: var(--font-sans);
  font-size: var(--fs-base);
  line-height: 1.5;
  color: var(--ink-primary);
  background: var(--bg-canvas);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

h1, h2, h3, h4, h5, h6 {
  margin: 0;
  font-weight: 600;
  line-height: 1.2;
  color: var(--ink-primary);
}

a {
  color: var(--accent-link);
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

button {
  font-family: inherit;
  border: none;
  background: none;
  cursor: pointer;
  padding: 0;
  color: inherit;
}

code,
pre {
  font-family: var(--font-mono);
  font-size: 0.9em;
}

#root {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}
```

- [ ] **Step 4**: 在 `web/src/main.tsx` 引入 global.css

读取现有 `web/src/main.tsx`，在顶部 import 处加：

```typescript
import "./styles/global.css";
```

如果文件已有 `import './index.css'` 之类，**保留**——global.css 是补充而非替代（旧 index.css 在 Task 7 删除）。

- [ ] **Step 5**: dev server 烟测

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npx vite build 2>&1 | tail -20
```

期望：build 成功（旧组件引用的 phaseFamily 等可能报错——记下错误，但 build 出现在 Task 7 删除旧组件后才能干净通过）

- [ ] **Step 6**: 提交

```bash
git add web/src/styles/ web/src/main.tsx
git commit -m "feat(web): add design tokens, fonts, and global styles"
```

---

### Task 5: lib utilities — motion + colors

**Files:**
- Create: `web/src/lib/motion.ts`
- Create: `web/src/lib/motion.test.ts`
- Create: `web/src/lib/colors.ts`
- Create: `web/src/lib/colors.test.ts`

#### Brief

落实 spec §6.3 + §6.4。

#### Steps

- [ ] **Step 1**: 创建 `web/src/lib/motion.ts`

```typescript
import type { Variants } from "framer-motion";

export const duration = {
  fast: 0.15,
  base: 0.25,
  slow: 0.4,
} as const;

export const ease = {
  out: [0.16, 1, 0.3, 1] as const,
  inOut: [0.65, 0, 0.35, 1] as const,
  spring: [0.34, 1.56, 0.64, 1] as const,
} as const;

export const fadeUp: Variants = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 },
};

export const fadeIn: Variants = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit: { opacity: 0 },
};

export const staggerContainer: Variants = {
  animate: { transition: { staggerChildren: 0.05 } },
};
```

- [ ] **Step 2**: 创建 `web/src/lib/motion.test.ts`

```typescript
import { describe, it, expect } from "vitest";
import { duration, ease, fadeUp } from "./motion";

describe("motion presets", () => {
  it("duration values are seconds", () => {
    expect(duration.fast).toBeGreaterThan(0);
    expect(duration.fast).toBeLessThan(duration.base);
    expect(duration.base).toBeLessThan(duration.slow);
  });

  it("ease tuples are 4-tuples of numbers", () => {
    expect(ease.out).toHaveLength(4);
    expect(ease.inOut).toHaveLength(4);
    expect(ease.spring).toHaveLength(4);
  });

  it("fadeUp variant has initial/animate/exit", () => {
    expect(fadeUp.initial).toBeDefined();
    expect(fadeUp.animate).toBeDefined();
    expect(fadeUp.exit).toBeDefined();
  });
});
```

- [ ] **Step 3**: 创建 `web/src/lib/colors.ts`

```typescript
import type { FamilyId } from "../types/family";

/** 把家族 id 映射到 CSS 变量引用，用于 inline style */
export function familyColorVar(id: FamilyId): string {
  const num = id.split("-")[0];
  return `var(--family-${num})`;
}

/** 取得家族 colorToken 的 CSS 变量名（不含 var() 包装） */
export function familyColorToken(id: FamilyId): string {
  const num = id.split("-")[0];
  return `--family-${num}`;
}
```

- [ ] **Step 4**: 创建 `web/src/lib/colors.test.ts`

```typescript
import { describe, it, expect } from "vitest";
import { familyColorVar, familyColorToken } from "./colors";

describe("colors", () => {
  it("familyColorVar wraps in var()", () => {
    expect(familyColorVar("01-cnn")).toBe("var(--family-01)");
    expect(familyColorVar("15-reasoning-o1-r1")).toBe("var(--family-15)");
  });

  it("familyColorToken returns bare variable name", () => {
    expect(familyColorToken("05-transformer")).toBe("--family-05");
  });
});
```

- [ ] **Step 5**: 跑测试

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npm test -- --run web/src/lib/
```

期望：5 个测试全 PASS

- [ ] **Step 6**: 提交

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add web/src/lib/
git commit -m "feat(web): add motion presets and family color utilities"
```

---

### Task 6: 从 `CnnTrack.tsx` 抽取 mini-arch 组件

**Files:**
- Create: `web/src/components/mini-arches/MiniLeNet.tsx`
- Create: `web/src/components/mini-arches/MiniAlexNet.tsx`
- Create: `web/src/components/mini-arches/MiniVGG.tsx`
- Create: `web/src/components/mini-arches/MiniGoogLeNet.tsx`
- Create: `web/src/components/mini-arches/MiniResNet.tsx`
- Create: `web/src/components/mini-arches/MiniDenseNet.tsx`
- Create: `web/src/components/mini-arches/MiniSENet.tsx`
- Create: `web/src/components/mini-arches/MiniEfficientNet.tsx` （新画或留 placeholder）
- Create: `web/src/components/mini-arches/MiniConvNeXt.tsx` （新画或留 placeholder）
- Create: `web/src/components/mini-arches/index.ts`

#### Brief

`web/src/components/tracks/CnnTrack.tsx`（1100 行）含 6 个 `Mini*` 内嵌函数组件：`MiniAlexNet`、`MiniVGG`、`MiniGoogLeNet`、`MiniResNet`、`MiniDenseNet`、`MiniSENet`。把这 6 个提到独立文件，**保持渲染逻辑 100% 不变**，只改：

- 导出方式：从内嵌 function 改为 `export function MiniXxx({ width = 160, height = 70 }: MiniArchProps): JSX.Element`
- 共享 className 来自原 CnnTrack 的 SVG 子样式（`illustration__layer`、`illustration__featuremap` 等）—— 这些 className 在原 CnnTrack 关联的 CSS 文件里。**本任务也复制对应 CSS**到一个共享文件 `web/src/components/mini-arches/mini-arch.css`。
- 没有 LeNet/EfficientNet/ConvNeXt 的现成 mini-arch——本任务用极简 placeholder（一个标了 "LeNet"/"EfficientNet"/"ConvNeXt" 的小灰框 SVG），W4 阶段再画精品。

#### Steps

- [ ] **Step 1**: 读取 `web/src/components/tracks/CnnTrack.tsx`，找到 `function MiniAlexNet`、`MiniVGG`、`MiniGoogLeNet`、`MiniResNet`、`MiniDenseNet`、`MiniSENet` 6 个函数（每个 ~30–50 行）。复制每个的 SVG body。

- [ ] **Step 2**: 创建 `web/src/components/mini-arches/types.ts`

```typescript
export interface MiniArchProps {
  width?: number;
  height?: number;
  ariaLabel?: string;
}
```

- [ ] **Step 3**: 为每个真实组件（6 个）创建独立文件

模板（以 MiniAlexNet 为例）：

```typescript
import type { MiniArchProps } from "./types";

export function MiniAlexNet({
  width = 160,
  height = 70,
  ariaLabel = "AlexNet 架构缩图",
}: MiniArchProps): JSX.Element {
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <g>
        {/* —— 复制原 CnnTrack.tsx 里 MiniAlexNet 函数体内的 SVG 元素 —— */}
      </g>
    </svg>
  );
}
```

其余 5 个（MiniVGG/MiniGoogLeNet/MiniResNet/MiniDenseNet/MiniSENet）按同样模式。

- [ ] **Step 4**: 为 3 个没现成的（MiniLeNet、MiniEfficientNet、MiniConvNeXt）创建 placeholder

```typescript
import type { MiniArchProps } from "./types";

export function MiniLeNet({
  width = 160,
  height = 70,
  ariaLabel = "LeNet 架构缩图（占位）",
}: MiniArchProps): JSX.Element {
  return (
    <svg
      viewBox="0 0 160 70"
      width={width}
      height={height}
      role="img"
      aria-label={ariaLabel}
    >
      <rect x="10" y="20" width="140" height="30" rx="4"
            fill="#f4f4f5" stroke="#a1a1aa" strokeWidth="1" />
      <text x="80" y="40" textAnchor="middle" fontSize="11" fill="#71717a">
        LeNet · 占位
      </text>
    </svg>
  );
}
```

`MiniEfficientNet`、`MiniConvNeXt` 同模式，只改 ariaLabel 和文本。

- [ ] **Step 5**: 共享 CSS：从原 CnnTrack 关联 CSS 复制 `illustration__layer*` 等 classNames

```bash
grep -rn "illustration__layer" web/src/styles/ 2>/dev/null
```

如果找到（如 `web/src/styles/CnnTrack.css` 或类似），把这些 illustration class 的 CSS 规则复制到新建文件：

`web/src/components/mini-arches/mini-arch.css`

```css
/* 从 CnnTrack 关联 CSS 中提取的 illustration class 规则 */
/* 占位：把原文件里所有 .illustration__layer*、.illustration__featuremap*、.illustration__proj*、
   .illustration__branch*、.illustration__addnorm、.illustration__residual、.illustration__block-label*
   等规则**逐字符复制**到这里 */
```

实际操作：

```bash
# 找到所有 illustration class 用过的 CSS 文件
grep -rln "illustration__" web/src/styles/
# 复制相关规则到 mini-arch.css
```

如果找不到（CnnTrack 用的 className 在某个全局 CSS 里），把整段 illustration block 复制过来。如果完全找不到（说明 className 是引用未定义的 class，渲染靠浏览器默认），则空文件即可——后续视觉调整再处理。

每个 mini-arch 组件文件**不需要**单独 import CSS，统一在 `index.ts` 中 import 一次。

- [ ] **Step 6**: 创建 `index.ts` barrel

```typescript
import "./mini-arch.css";

export { MiniLeNet } from "./MiniLeNet";
export { MiniAlexNet } from "./MiniAlexNet";
export { MiniVGG } from "./MiniVGG";
export { MiniGoogLeNet } from "./MiniGoogLeNet";
export { MiniResNet } from "./MiniResNet";
export { MiniDenseNet } from "./MiniDenseNet";
export { MiniSENet } from "./MiniSENet";
export { MiniEfficientNet } from "./MiniEfficientNet";
export { MiniConvNeXt } from "./MiniConvNeXt";
export type { MiniArchProps } from "./types";
```

- [ ] **Step 7**: 验证 mini-arches 编译

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npx tsc --noEmit 2>&1 | grep "mini-arches" | head
```

期望：无 mini-arches 相关 TS 错误。

- [ ] **Step 8**: 提交

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add web/src/components/mini-arches/
git commit -m "feat(web): extract mini-arches from CnnTrack into 9 reusable components"
```

---

### Task 7: 删除旧组件 / 数据 / 测试

**Files:**
- Delete: 见 File Structure §删除列表

#### Steps

- [ ] **Step 1**: 删除旧组件

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git rm web/src/components/TimelineAxis.tsx \
       web/src/components/TimelineAxis.test.tsx \
       web/src/components/TimelineContent.tsx \
       web/src/components/TimelineWorkList.tsx \
       web/src/components/RelatedModules.tsx \
       web/src/components/ArchitectureMap.tsx \
       web/src/components/TimelineIllustration.tsx \
       web/src/components/PrehistoryDrawer.tsx \
       web/src/components/TrackView.tsx \
       web/src/components/tracks/CnnTrack.tsx
rmdir web/src/components/tracks/ 2>/dev/null || true
```

如果 `web/src/components/tracks/` 还有其他文件（如 CSS），先 grep 确认：

```bash
ls web/src/components/tracks/ 2>/dev/null
```

若有 CSS 文件，也 git rm（mini-arch.css 已在 Task 6 创建独立副本）。

- [ ] **Step 2**: 删除旧数据

```bash
git rm web/src/data/timeline.ts \
       web/src/data/timeline.test.ts \
       web/src/data/phaseFamily.ts
```

- [ ] **Step 3**: 删除旧 App + 测试

```bash
git rm web/src/App.tsx web/src/App.test.tsx
```

（**新 App.tsx 在 Task 8 创建**；这里先删掉旧的）

- [ ] **Step 4**: 删除旧 index.css 或 App.css（如果存在）

```bash
ls web/src/*.css 2>/dev/null
```

如果有 `index.css` / `App.css`，确认它们的内容是不是已经被 `global.css` 取代——是则 `git rm` 删掉。

- [ ] **Step 5**: 验证 `main.tsx` 仍能工作（暂时会报错"没有 App.tsx"）

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npx tsc --noEmit 2>&1 | head -10
```

期望：报错"找不到 App"——这是预期的，下个 Task 创建新 App.tsx 后修复。

- [ ] **Step 6**: 提交

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git commit -m "refactor(web): remove old timeline / phaseFamily components for blank-slate rewrite"
```

---

### Task 8: 路由根 + Layout + 404

**Files:**
- Create: `web/src/App.tsx`
- Create: `web/src/components/ui/Layout.tsx`
- Create: `web/src/components/ui/NotFoundPage.tsx`

#### Brief

新 `App.tsx` 是 React Router v7 的入口，挂 4 个路由：`/`、`/families/:familyId`、`/families/:familyId/:nodeSlug`、`*`。Layout 给所有页面提供顶部头部（logo + 切换 dark mode 占位）+ 全宽容器。

#### Steps

- [ ] **Step 1**: 创建 `web/src/components/ui/NotFoundPage.tsx`

```typescript
import { Link } from "react-router";

export function NotFoundPage() {
  return (
    <div style={{ padding: "var(--space-16)", textAlign: "center" }}>
      <h1 style={{ fontSize: "var(--fs-4xl)", marginBottom: "var(--space-4)" }}>
        404
      </h1>
      <p style={{ color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        页面不存在
      </p>
      <Link to="/">返回主页</Link>
    </div>
  );
}
```

- [ ] **Step 2**: 创建 `web/src/components/ui/Layout.tsx`

```typescript
import type { ReactNode } from "react";
import { Link } from "react-router";

export function Layout({ children }: { children: ReactNode }) {
  return (
    <>
      <header
        style={{
          padding: "var(--space-4) var(--space-8)",
          borderBottom: "1px solid var(--border)",
          background: "var(--bg-surface)",
        }}
      >
        <Link
          to="/"
          style={{
            fontSize: "var(--fs-lg)",
            fontWeight: 600,
            color: "var(--ink-primary)",
          }}
        >
          Daily-LLM · 被逼出来的历史
        </Link>
      </header>
      <main style={{ flex: 1 }}>{children}</main>
      <footer
        style={{
          padding: "var(--space-4) var(--space-8)",
          borderTop: "1px solid var(--border)",
          fontSize: "var(--fs-sm)",
          color: "var(--ink-muted)",
        }}
      >
        Daily-LLM · 2026
      </footer>
    </>
  );
}
```

- [ ] **Step 3**: 创建 `web/src/App.tsx`

```typescript
import { BrowserRouter, Route, Routes } from "react-router";
import { Layout } from "./components/ui/Layout";
import { NotFoundPage } from "./components/ui/NotFoundPage";
import { HomePage } from "./components/home/HomePage";
import { FamilyPage } from "./components/family/FamilyPage";
import { NodePage } from "./components/node/NodePage";

export function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/families/:familyId" element={<FamilyPage />} />
          <Route
            path="/families/:familyId/:nodeSlug"
            element={<NodePage />}
          />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
```

- [ ] **Step 4**: 创建临时空组件骨架让编译过（Task 10/11/12 才填实）

`web/src/components/home/HomePage.tsx`:

```typescript
export function HomePage() {
  return <div>HomePage (Task 10 will fill)</div>;
}
```

`web/src/components/family/FamilyPage.tsx`:

```typescript
export function FamilyPage() {
  return <div>FamilyPage (Task 11 will fill)</div>;
}
```

`web/src/components/node/NodePage.tsx`:

```typescript
export function NodePage() {
  return <div>NodePage (Task 12 will fill)</div>;
}
```

- [ ] **Step 5**: 更新 `web/src/main.tsx` 引用新 App

读取 main.tsx 现状，确认 import 是 `import { App } from "./App"`（命名导出）或 `import App from "./App"`（默认）。新 App.tsx 是命名导出，调整 import 为：

```typescript
import { App } from "./App";
```

- [ ] **Step 6**: 编译 + dev server 烟测

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npx tsc --noEmit
```

期望：无 TS 报错。

```bash
npx vite build
```

期望：build 成功。

- [ ] **Step 7**: 提交

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add web/src/App.tsx web/src/components/ui/ web/src/components/home/ web/src/components/family/ web/src/components/node/ web/src/main.tsx
git commit -m "feat(web): add router root with Layout, 4 routes, and placeholder pages"
```

---

### Task 9: Markdown 渲染基础设施

**Files:**
- Create: `web/src/components/node/MarkdownRenderer.tsx`
- Create: `web/src/components/node/MermaidBlock.tsx`

#### Brief

NodePage（Task 12）会渲染节点 markdown。需要：

- `MarkdownRenderer`：包装 react-markdown + remark-gfm + remark-math + rehype-highlight + rehype-katex；自定义 `code` 渲染器把 ```mermaid 块转交给 `MermaidBlock`
- `MermaidBlock`：动态 import mermaid.js，初始化并渲染单个 mermaid 块；显示 loading 状态

#### Steps

- [ ] **Step 1**: 创建 `web/src/components/node/MermaidBlock.tsx`

```typescript
import { useEffect, useRef, useState } from "react";

interface MermaidBlockProps {
  code: string;
}

let mermaidPromise: Promise<typeof import("mermaid").default> | null = null;

function loadMermaid() {
  if (!mermaidPromise) {
    mermaidPromise = import("mermaid").then((m) => {
      m.default.initialize({
        startOnLoad: false,
        theme: "neutral",
        fontFamily: "var(--font-sans)",
      });
      return m.default;
    });
  }
  return mermaidPromise;
}

export function MermaidBlock({ code }: MermaidBlockProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    loadMermaid().then(async (mermaid) => {
      if (cancelled || !ref.current) return;
      try {
        const id = `mermaid-${Math.random().toString(36).slice(2)}`;
        const { svg } = await mermaid.render(id, code);
        if (!cancelled && ref.current) {
          ref.current.innerHTML = svg;
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
        }
      }
    });
    return () => {
      cancelled = true;
    };
  }, [code]);

  if (error) {
    return (
      <pre style={{ color: "var(--accent-warn)", padding: "var(--space-4)" }}>
        Mermaid render error: {error}
        {"\n\n"}
        {code}
      </pre>
    );
  }
  return <div ref={ref} style={{ margin: "var(--space-6) 0" }} />;
}
```

- [ ] **Step 2**: 创建 `web/src/components/node/MarkdownRenderer.tsx`

```typescript
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import "highlight.js/styles/github.css";
import { MermaidBlock } from "./MermaidBlock";

interface MarkdownRendererProps {
  markdown: string;
}

export function MarkdownRenderer({ markdown }: MarkdownRendererProps) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeHighlight, rehypeKatex]}
        components={{
          code({ className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || "");
            const lang = match?.[1];
            const codeStr = String(children).replace(/\n$/, "");
            if (lang === "mermaid") {
              return <MermaidBlock code={codeStr} />;
            }
            return (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
        }}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  );
}
```

- [ ] **Step 3**: 编译验证

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npx tsc --noEmit 2>&1 | grep -E "MarkdownRenderer|MermaidBlock" | head -5
```

期望：无错。

- [ ] **Step 4**: 提交

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add web/src/components/node/MarkdownRenderer.tsx web/src/components/node/MermaidBlock.tsx
git commit -m "feat(web): add MarkdownRenderer with Mermaid + KaTeX + highlight.js"
```

---

### Task 10: HomePage 主页（D 方案）

**Files:**
- Create / overwrite: `web/src/components/home/HomePage.tsx`
- Create: `web/src/components/home/TimeAxisView.tsx`
- Create: `web/src/components/home/FamilyGridView.tsx`
- Create: `web/src/components/home/NodeHoverCard.tsx`
- Create: `web/src/components/home/HomePage.module.css`

#### Brief

主页是 W3 重头戏。两种视图（时间轴 / 家族网格）通过顶部 toggle 切换；用 Framer Motion `AnimatePresence` 做平滑切换。

数据：`import familiesData from "../../data/families.json"`。

视觉目标（W3 阶段）：达到"明显比 GitHub 漂亮"，但不要求"distill.pub 顶级"——那是 W4 节点页的目标。这里先**功能可用、配色统一、过渡流畅**。

#### Steps

- [ ] **Step 1**: 创建 `web/src/components/home/HomePage.module.css`

```css
.container {
  padding: var(--space-12) var(--space-8);
  max-width: 1400px;
  margin: 0 auto;
}

.header {
  text-align: center;
  margin-bottom: var(--space-12);
}

.title {
  font-size: var(--fs-4xl);
  margin-bottom: var(--space-3);
}

.subtitle {
  font-size: var(--fs-lg);
  color: var(--ink-secondary);
}

.toggle {
  display: inline-flex;
  background: var(--bg-subtle);
  border-radius: var(--radius-full);
  padding: var(--space-1);
  margin: var(--space-6) auto;
}

.toggleBtn {
  padding: var(--space-2) var(--space-6);
  border-radius: var(--radius-full);
  font-size: var(--fs-sm);
  font-weight: 500;
  color: var(--ink-secondary);
  transition: color var(--dur-fast) var(--ease-out);
}

.toggleBtnActive {
  background: var(--bg-surface);
  color: var(--ink-primary);
  box-shadow: var(--shadow-sm);
}
```

- [ ] **Step 2**: 创建 `web/src/components/home/NodeHoverCard.tsx`

```typescript
import { motion } from "framer-motion";
import { Link } from "react-router";
import type { NodeData } from "../../types/family";
import { familyColorVar } from "../../lib/colors";
import { fadeUp, duration, ease } from "../../lib/motion";

interface NodeHoverCardProps {
  node: NodeData;
  x: number;
  y: number;
}

export function NodeHoverCard({ node, x, y }: NodeHoverCardProps) {
  const nodeSlug = node.path.split("/").pop()!.replace(/\.md$/, "");
  return (
    <motion.div
      variants={fadeUp}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={{ duration: duration.fast, ease: ease.out }}
      style={{
        position: "absolute",
        left: x,
        top: y,
        zIndex: 10,
        padding: "var(--space-4)",
        background: "var(--bg-surface)",
        border: `2px solid ${familyColorVar(node.family)}`,
        borderRadius: "var(--radius-md)",
        boxShadow: "var(--shadow-lg)",
        minWidth: 240,
        maxWidth: 320,
      }}
    >
      <div style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        {node.year} · {node.family}
      </div>
      <div
        style={{
          fontSize: "var(--fs-lg)",
          fontWeight: 600,
          marginTop: "var(--space-1)",
        }}
      >
        {node.name}
      </div>
      <p
        style={{
          fontSize: "var(--fs-sm)",
          color: "var(--ink-secondary)",
          margin: "var(--space-3) 0",
          lineHeight: 1.5,
        }}
      >
        {node.key_idea}
      </p>
      <div style={{ display: "flex", gap: "var(--space-3)" }}>
        <Link
          to={`/families/${node.family}`}
          style={{ fontSize: "var(--fs-sm)" }}
        >
          → 进入家族
        </Link>
        <Link
          to={`/families/${node.family}/${nodeSlug}`}
          style={{ fontSize: "var(--fs-sm)" }}
        >
          → 节点详情
        </Link>
      </div>
    </motion.div>
  );
}
```

- [ ] **Step 3**: 创建 `web/src/components/home/TimeAxisView.tsx`

```typescript
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { scaleLinear } from "d3-scale";
import type { FamiliesData, NodeData } from "../../types/family";
import { familyColorVar } from "../../lib/colors";
import { NodeHoverCard } from "./NodeHoverCard";

interface TimeAxisViewProps {
  data: FamiliesData;
}

const PADDING = 80;
const AXIS_Y = 200;
const NODE_RADIUS = 8;

export function TimeAxisView({ data }: TimeAxisViewProps) {
  const allNodes = data.families.flatMap((f) => f.nodes);
  if (allNodes.length === 0) {
    return <div>暂无节点数据</div>;
  }
  const minYear = Math.min(...allNodes.map((n) => n.year)) - 1;
  const maxYear = Math.max(...allNodes.map((n) => n.year)) + 1;
  const width = 1200;
  const height = 400;
  const xScale = scaleLinear()
    .domain([minYear, maxYear])
    .range([PADDING, width - PADDING]);

  // y 偏移：同一年多个节点错开
  const yByNode = new Map<string, number>();
  const groupedByYear = new Map<number, NodeData[]>();
  for (const n of allNodes) {
    if (!groupedByYear.has(n.year)) groupedByYear.set(n.year, []);
    groupedByYear.get(n.year)!.push(n);
  }
  for (const [, nodes] of groupedByYear) {
    nodes.sort((a, b) => a.family.localeCompare(b.family));
    nodes.forEach((n, i) => {
      yByNode.set(n.path, AXIS_Y - (i - (nodes.length - 1) / 2) * 24);
    });
  }

  const [hovered, setHovered] = useState<NodeData | null>(null);
  const [pos, setPos] = useState({ x: 0, y: 0 });

  // years to show as labels (every other year)
  const years: number[] = [];
  for (let y = Math.ceil(minYear); y <= Math.floor(maxYear); y += 2) {
    years.push(y);
  }

  return (
    <div style={{ position: "relative" }}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{ width: "100%", height: "auto", maxHeight: 500 }}
      >
        {/* axis line */}
        <line
          x1={PADDING}
          x2={width - PADDING}
          y1={AXIS_Y}
          y2={AXIS_Y}
          stroke="var(--border)"
          strokeWidth={1.5}
        />
        {/* year ticks */}
        {years.map((y) => (
          <g key={y} transform={`translate(${xScale(y)}, ${AXIS_Y})`}>
            <line y2={6} stroke="var(--ink-muted)" />
            <text
              y={22}
              textAnchor="middle"
              fontSize={12}
              fill="var(--ink-muted)"
            >
              {y}
            </text>
          </g>
        ))}
        {/* nodes */}
        {allNodes.map((n) => (
          <motion.circle
            key={n.path}
            cx={xScale(n.year)}
            cy={yByNode.get(n.path)!}
            r={NODE_RADIUS}
            fill={familyColorVar(n.family)}
            stroke="var(--bg-canvas)"
            strokeWidth={2}
            whileHover={{ scale: 1.4 }}
            onMouseEnter={(e) => {
              setHovered(n);
              const rect = (e.currentTarget as SVGCircleElement)
                .ownerSVGElement!.getBoundingClientRect();
              setPos({
                x: rect.left + window.scrollX + xScale(n.year) - 120,
                y: rect.top + window.scrollY + yByNode.get(n.path)! - 200,
              });
            }}
            onMouseLeave={() => setHovered(null)}
            style={{ cursor: "pointer" }}
            layoutId={`node-${n.path}`}
          />
        ))}
      </svg>
      <AnimatePresence>
        {hovered && <NodeHoverCard node={hovered} x={pos.x} y={pos.y} />}
      </AnimatePresence>
    </div>
  );
}
```

- [ ] **Step 4**: 创建 `web/src/components/home/FamilyGridView.tsx`

```typescript
import { motion } from "framer-motion";
import { Link } from "react-router";
import type { FamiliesData } from "../../types/family";
import { familyColorVar } from "../../lib/colors";
import { staggerContainer, fadeUp, duration, ease } from "../../lib/motion";

interface FamilyGridViewProps {
  data: FamiliesData;
}

export function FamilyGridView({ data }: FamilyGridViewProps) {
  return (
    <motion.div
      variants={staggerContainer}
      initial="initial"
      animate="animate"
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
        gap: "var(--space-6)",
      }}
    >
      {data.families.map((f) => (
        <motion.div
          key={f.id}
          variants={fadeUp}
          transition={{ duration: duration.base, ease: ease.out }}
        >
          <Link
            to={`/families/${f.id}`}
            style={{
              display: "block",
              padding: "var(--space-6)",
              background: "var(--bg-surface)",
              border: "1px solid var(--border)",
              borderTop: `4px solid ${familyColorVar(f.id)}`,
              borderRadius: "var(--radius-md)",
              transition: `transform var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out)`,
              color: "var(--ink-primary)",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = "translateY(-4px)";
              e.currentTarget.style.boxShadow = "var(--shadow-md)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = "";
              e.currentTarget.style.boxShadow = "";
            }}
          >
            <div
              style={{
                fontSize: "var(--fs-sm)",
                color: "var(--ink-muted)",
                marginBottom: "var(--space-2)",
              }}
            >
              {f.id}
            </div>
            <h3
              style={{
                fontSize: "var(--fs-lg)",
                marginBottom: "var(--space-3)",
              }}
            >
              {f.label}
            </h3>
            <p
              style={{
                fontSize: "var(--fs-sm)",
                color: "var(--ink-secondary)",
                lineHeight: 1.5,
                marginBottom: "var(--space-4)",
              }}
            >
              {f.blurb}
            </p>
            <div
              style={{
                fontSize: "var(--fs-xs)",
                color: "var(--ink-muted)",
              }}
            >
              {f.nodes.length > 0
                ? `${f.nodes.length} 节点 · ${f.yearRange?.[0]}–${f.yearRange?.[1]}`
                : "待补充"}
            </div>
          </Link>
        </motion.div>
      ))}
    </motion.div>
  );
}
```

- [ ] **Step 5**: 创建 `web/src/components/home/HomePage.tsx`（覆盖 Task 8 的占位）

```typescript
import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import type { FamiliesData } from "../../types/family";
import familiesJson from "../../data/families.json";
import { TimeAxisView } from "./TimeAxisView";
import { FamilyGridView } from "./FamilyGridView";
import { fadeIn, duration, ease } from "../../lib/motion";
import styles from "./HomePage.module.css";

const data = familiesJson as FamiliesData;

type Mode = "time" | "family";

export function HomePage() {
  const [mode, setMode] = useState<Mode>("time");

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1 className={styles.title}>被逼出来的历史</h1>
        <p className={styles.subtitle}>
          深度学习与大模型 · 1998–2025 · {data.families.length} 家族
        </p>
        <div className={styles.toggle}>
          <button
            className={`${styles.toggleBtn} ${
              mode === "time" ? styles.toggleBtnActive : ""
            }`}
            onClick={() => setMode("time")}
          >
            按时间
          </button>
          <button
            className={`${styles.toggleBtn} ${
              mode === "family" ? styles.toggleBtnActive : ""
            }`}
            onClick={() => setMode("family")}
          >
            按家族
          </button>
        </div>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={mode}
          variants={fadeIn}
          initial="initial"
          animate="animate"
          exit="exit"
          transition={{ duration: duration.base, ease: ease.out }}
        >
          {mode === "time" ? (
            <TimeAxisView data={data} />
          ) : (
            <FamilyGridView data={data} />
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
```

- [ ] **Step 6**: 编译验证

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npx tsc --noEmit
```

期望：无错。

- [ ] **Step 7**: dev server 手动验证

```bash
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

浏览器访问 `http://127.0.0.1:5173/`，确认：

- 看到 "被逼出来的历史" 标题 + 切换按钮
- 默认时间轴模式：横向轴 + 节点点 + hover 弹卡
- 点 "按家族"：15 张卡片网格 + hover 上移效果
- 点回 "按时间"：平滑切换

按 Ctrl+C 停 dev server。

- [ ] **Step 8**: 提交

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add web/src/components/home/
git commit -m "feat(web): HomePage with time-axis / family-grid views and toggle"
```

---

### Task 11: FamilyPage 家族详情页

**Files:**
- Create / overwrite: `web/src/components/family/FamilyPage.tsx`
- Create: `web/src/components/family/FamilyPage.module.css`

#### Brief

`/families/01-cnn` 等。布局：返回链接 + 家族标题 + 一句话定位 + 子时间线（含 mini-arch 缩略图）+ "概念本身/依赖与延伸"暂用 README markdown 渲染（W4 阶段优化）。

#### Steps

- [ ] **Step 1**: 创建 `web/src/components/family/FamilyPage.module.css`

```css
.container {
  padding: var(--space-12) var(--space-8);
  max-width: 1100px;
  margin: 0 auto;
}

.back {
  display: inline-block;
  margin-bottom: var(--space-6);
  font-size: var(--fs-sm);
  color: var(--ink-secondary);
}

.familyId {
  font-size: var(--fs-sm);
  color: var(--ink-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.title {
  font-size: var(--fs-3xl);
  margin: var(--space-2) 0 var(--space-3);
}

.blurb {
  font-size: var(--fs-lg);
  color: var(--ink-secondary);
  font-style: italic;
  margin-bottom: var(--space-12);
  padding-left: var(--space-4);
  border-left: 4px solid var(--accent);
}

.subtimeline {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: var(--space-4);
  margin: var(--space-8) 0;
}

.nodeCard {
  padding: var(--space-4);
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  text-decoration: none;
  color: var(--ink-primary);
  transition: transform var(--dur-fast) var(--ease-out);
}

.nodeCard:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
  text-decoration: none;
}

.nodeYear {
  font-size: var(--fs-sm);
  color: var(--ink-muted);
}

.nodeName {
  font-size: var(--fs-md);
  font-weight: 600;
  margin: var(--space-2) 0;
}

.nodeIdea {
  font-size: var(--fs-sm);
  color: var(--ink-secondary);
  line-height: 1.4;
  margin: var(--space-2) 0 0;
}
```

- [ ] **Step 2**: 创建 mini-arch 选择器

新增 `web/src/components/mini-arches/getMiniArch.tsx`：

```typescript
import type { ComponentType } from "react";
import type { MiniArchProps } from "./types";
import {
  MiniLeNet,
  MiniAlexNet,
  MiniVGG,
  MiniGoogLeNet,
  MiniResNet,
  MiniDenseNet,
  MiniEfficientNet,
  MiniConvNeXt,
} from ".";

const MAP: Record<string, ComponentType<MiniArchProps>> = {
  "01-lenet": MiniLeNet,
  "02-alexnet": MiniAlexNet,
  "03-vgg": MiniVGG,
  "04-inception": MiniGoogLeNet,
  "05-resnet": MiniResNet,
  "06-densenet": MiniDenseNet,
  "07-efficientnet": MiniEfficientNet,
  "08-convnext": MiniConvNeXt,
};

export function getMiniArch(
  nodePath: string
): ComponentType<MiniArchProps> | null {
  const slug = nodePath.split("/").pop()?.replace(/\.md$/, "") ?? "";
  return MAP[slug] ?? null;
}
```

把它加进 `web/src/components/mini-arches/index.ts` 的 export：

```typescript
export { getMiniArch } from "./getMiniArch";
```

- [ ] **Step 3**: 创建 `web/src/components/family/FamilyPage.tsx`（覆盖 Task 8 占位）

```typescript
import { useParams, Link, Navigate } from "react-router";
import type { FamiliesData, FamilyId } from "../../types/family";
import familiesJson from "../../data/families.json";
import { familyColorVar } from "../../lib/colors";
import { getMiniArch } from "../mini-arches/getMiniArch";
import styles from "./FamilyPage.module.css";

const data = familiesJson as FamiliesData;

export function FamilyPage() {
  const { familyId } = useParams<{ familyId: FamilyId }>();
  const family = data.families.find((f) => f.id === familyId);
  if (!family) {
    return <Navigate to="/404" replace />;
  }
  const accent = familyColorVar(family.id);

  return (
    <div className={styles.container}>
      <Link to="/" className={styles.back}>
        ← 返回主页
      </Link>
      <div className={styles.familyId}>{family.id}</div>
      <h1 className={styles.title} style={{ color: accent }}>
        {family.label}
      </h1>
      <p
        className={styles.blurb}
        style={{ ["--accent" as string]: accent } as React.CSSProperties}
      >
        {family.blurb}
      </p>

      {family.nodes.length === 0 ? (
        <p style={{ color: "var(--ink-muted)" }}>本家族内容待补充。</p>
      ) : (
        <>
          <h2 style={{ fontSize: "var(--fs-xl)" }}>子时间线</h2>
          <div className={styles.subtimeline}>
            {family.nodes.map((n) => {
              const slug = n.path.split("/").pop()!.replace(/\.md$/, "");
              const MiniArch = getMiniArch(n.path);
              return (
                <Link
                  key={n.path}
                  to={`/families/${family.id}/${slug}`}
                  className={styles.nodeCard}
                  style={{ borderTopColor: accent, borderTopWidth: 3 }}
                >
                  <div className={styles.nodeYear}>{n.year}</div>
                  <div className={styles.nodeName}>{n.name}</div>
                  {MiniArch && (
                    <div style={{ margin: "var(--space-2) 0" }}>
                      <MiniArch width={160} height={50} />
                    </div>
                  )}
                  <p className={styles.nodeIdea}>{n.key_idea}</p>
                </Link>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 4**: 编译 + dev 验证

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npx tsc --noEmit
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

浏览器访问 `http://127.0.0.1:5173/families/01-cnn`，确认：

- 看到 CNN 家族标题 + 一句话定位
- 8 个节点卡片成网格排列，含 mini-arch 缩略图（部分是 placeholder）
- 点某个节点卡片，跳到 `/families/01-cnn/05-resnet` 这样的 URL

访问 `/families/02-rnn-lstm`（空家族），确认显示"待补充"。

访问 `/families/nonexistent`，确认重定向到 404。

按 Ctrl+C 停 dev server。

- [ ] **Step 5**: 提交

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add web/src/components/family/ web/src/components/mini-arches/getMiniArch.tsx web/src/components/mini-arches/index.ts
git commit -m "feat(web): FamilyPage with sub-timeline and mini-arch cards"
```

---

### Task 12: NodePage（W3 占位）

**Files:**
- Create / overwrite: `web/src/components/node/NodePage.tsx`
- Create: `web/src/components/node/NodePage.module.css`

#### Brief

W3 阶段节点页是"markdown 高质量渲染"。Markdown 内容通过 Vite 的 `import.meta.glob` + `?raw` 在构建时静态导入。W4 阶段会重写整个 NodePage 加 scrollytelling。

#### Steps

- [ ] **Step 1**: 创建 `web/src/components/node/NodePage.module.css`

```css
.container {
  padding: var(--space-12) var(--space-8);
  max-width: 800px;
  margin: 0 auto;
}

.back {
  display: inline-block;
  margin-bottom: var(--space-6);
  font-size: var(--fs-sm);
  color: var(--ink-secondary);
}

.meta {
  margin-bottom: var(--space-8);
  padding-bottom: var(--space-6);
  border-bottom: 1px solid var(--border);
}

.metaLine {
  font-size: var(--fs-sm);
  color: var(--ink-muted);
  margin: var(--space-1) 0;
}

.body {
  font-family: var(--font-serif);
  font-size: var(--fs-md);
  line-height: 1.7;
}

.body :global(h1) {
  font-family: var(--font-sans);
  font-size: var(--fs-3xl);
  margin: var(--space-8) 0 var(--space-4);
}

.body :global(h2) {
  font-family: var(--font-sans);
  font-size: var(--fs-2xl);
  margin: var(--space-8) 0 var(--space-3);
}

.body :global(h3) {
  font-family: var(--font-sans);
  font-size: var(--fs-xl);
  margin: var(--space-6) 0 var(--space-2);
}

.body :global(p) {
  margin: var(--space-4) 0;
}

.body :global(blockquote) {
  border-left: 3px solid var(--accent-link);
  padding-left: var(--space-4);
  margin: var(--space-4) 0;
  color: var(--ink-secondary);
}

.body :global(table) {
  border-collapse: collapse;
  margin: var(--space-4) 0;
  font-size: var(--fs-sm);
  font-family: var(--font-sans);
}

.body :global(th),
.body :global(td) {
  border: 1px solid var(--border);
  padding: var(--space-2) var(--space-3);
  text-align: left;
}

.body :global(th) {
  background: var(--bg-subtle);
  font-weight: 600;
}

.body :global(img) {
  max-width: 100%;
  height: auto;
  margin: var(--space-4) 0;
}

.body :global(em) {
  display: block;
  text-align: center;
  color: var(--ink-muted);
  font-size: var(--fs-sm);
  margin-top: var(--space-2);
}
```

- [ ] **Step 2**: 创建 `web/src/components/node/NodePage.tsx`（覆盖 Task 8 占位）

```typescript
import { useParams, Link, Navigate } from "react-router";
import { useEffect, useState } from "react";
import type { FamiliesData, FamilyId, NodeData } from "../../types/family";
import familiesJson from "../../data/families.json";
import { familyColorVar } from "../../lib/colors";
import { MarkdownRenderer } from "./MarkdownRenderer";
import styles from "./NodePage.module.css";

const data = familiesJson as FamiliesData;

// Glob all markdown files under repo root NN-xxx/ directories at build time
const markdownModules = import.meta.glob("../../../../[0-9][0-9]-*/*.md", {
  as: "raw",
  eager: false,
}) as Record<string, () => Promise<string>>;

function nodePathToModuleKey(nodePath: string): string {
  // nodePath: "01-cnn/05-resnet.md" → "../../../../01-cnn/05-resnet.md"
  return `../../../../${nodePath}`;
}

export function NodePage() {
  const { familyId, nodeSlug } = useParams<{
    familyId: FamilyId;
    nodeSlug: string;
  }>();
  const family = data.families.find((f) => f.id === familyId);
  const node = family?.nodes.find(
    (n) => n.path.split("/").pop()?.replace(/\.md$/, "") === nodeSlug
  );

  const [markdown, setMarkdown] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    if (!node) return;
    const key = nodePathToModuleKey(node.path);
    const loader = markdownModules[key];
    if (!loader) {
      setLoadError(`Markdown not found: ${node.path}`);
      return;
    }
    loader().then(setMarkdown).catch((e) => setLoadError(String(e)));
  }, [node?.path]);

  if (!family || !node) {
    return <Navigate to="/404" replace />;
  }

  const accent = familyColorVar(family.id);

  // Strip frontmatter from markdown before render
  const body = markdown?.replace(/^---[\s\S]*?---\n?/, "") ?? "";

  return (
    <div className={styles.container}>
      <Link to={`/families/${family.id}`} className={styles.back}>
        ← 返回 {family.label}
      </Link>
      <div className={styles.meta}>
        <h1 style={{ fontSize: "var(--fs-3xl)", color: accent }}>
          {node.name} ({node.year})
        </h1>
        <div className={styles.metaLine}>
          作者: {node.authors.join(", ") || "—"}
        </div>
        <div className={styles.metaLine}>论文: {node.paper}</div>
      </div>
      <div className={styles.body}>
        {loadError && (
          <p style={{ color: "var(--accent-warn)" }}>
            加载失败: {loadError}
          </p>
        )}
        {!markdown && !loadError && <p>加载中…</p>}
        {markdown && <MarkdownRenderer markdown={body} />}
      </div>
    </div>
  );
}
```

- [ ] **Step 3**: 验证 Vite glob 路径

`import.meta.glob` 的相对路径以**当前文件**为起点。NodePage.tsx 在 `web/src/components/node/`，仓库根的家族目录在 `../../../../01-cnn/`（四层 `..`：node/ → components/ → src/ → web/ → repo-root）。如果实际路径不对，调整 `../../../../` 的层数。

可以临时在 NodePage 里 `console.log(Object.keys(markdownModules))` 看 keys 长什么样。

- [ ] **Step 4**: 编译 + dev 验证

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npx tsc --noEmit
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

浏览器访问 `http://127.0.0.1:5173/families/01-cnn/05-resnet`，确认：

- 看到 ResNet 标题 + 作者 + 论文
- Markdown 正文渲染：H2 / H3 / 段落 / 表格 / blockquote 都正常
- Mermaid 块渲染成 SVG（可能需要点时间）
- SVG `![残差块](assets/...)` —— 注意这个图片**可能 404**，因为 Vite dev server 不知道仓库根的 `assets/` 在哪。这是已知限制，W4 处理。**本任务接受 SVG 显示为破图标**。
- KaTeX 公式 `$$ $$` 块渲染正常
- 代码块语法高亮正常

按 Ctrl+C 停 dev server。

- [ ] **Step 5**: 提交

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add web/src/components/node/NodePage.tsx web/src/components/node/NodePage.module.css
git commit -m "feat(web): NodePage with markdown rendering (placeholder for W4 scrollytelling)"
```

---

### Task 13: 测试

**Files:**
- Create: `web/src/App.test.tsx`
- Create: `web/src/components/home/HomePage.test.tsx`
- Create: `web/src/components/family/FamilyPage.test.tsx`
- Create: `web/src/components/node/NodePage.test.tsx`

#### Brief

12 个关键测试，确保 happy path 不破。

#### Steps

- [ ] **Step 1**: 创建 `web/src/App.test.tsx`

```typescript
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router";
import { Layout } from "./components/ui/Layout";
import { NotFoundPage } from "./components/ui/NotFoundPage";
import { HomePage } from "./components/home/HomePage";

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Layout>
    </MemoryRouter>
  );
}

describe("App router", () => {
  it("renders HomePage at /", () => {
    renderAt("/");
    expect(screen.getByText(/被逼出来的历史/)).toBeInTheDocument();
  });

  it("renders 404 at unknown route", () => {
    renderAt("/nonexistent");
    expect(screen.getByText("404")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2**: 创建 `web/src/components/home/HomePage.test.tsx`

```typescript
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { HomePage } from "./HomePage";

describe("HomePage", () => {
  it("renders title and mode toggle", () => {
    render(
      <MemoryRouter>
        <HomePage />
      </MemoryRouter>
    );
    expect(screen.getByText(/被逼出来的历史/)).toBeInTheDocument();
    expect(screen.getByText("按时间")).toBeInTheDocument();
    expect(screen.getByText("按家族")).toBeInTheDocument();
  });

  it("toggles between time and family views", () => {
    render(
      <MemoryRouter>
        <HomePage />
      </MemoryRouter>
    );
    // default is time mode — find some axis element
    fireEvent.click(screen.getByText("按家族"));
    // family grid should show all 15 family labels
    expect(screen.getByText(/01-cnn/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 3**: 创建 `web/src/components/family/FamilyPage.test.tsx`

```typescript
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router";
import { FamilyPage } from "./FamilyPage";

function renderFamily(familyId: string) {
  return render(
    <MemoryRouter initialEntries={[`/families/${familyId}`]}>
      <Routes>
        <Route path="/families/:familyId" element={<FamilyPage />} />
        <Route path="/404" element={<div>404</div>} />
      </Routes>
    </MemoryRouter>
  );
}

describe("FamilyPage", () => {
  it("renders CNN family with 8 nodes", () => {
    renderFamily("01-cnn");
    expect(screen.getByText(/CNN/)).toBeInTheDocument();
    expect(screen.getByText("子时间线")).toBeInTheDocument();
    expect(screen.getByText(/ResNet/)).toBeInTheDocument();
    expect(screen.getByText(/AlexNet/)).toBeInTheDocument();
  });

  it("shows '待补充' for empty family", () => {
    renderFamily("02-rnn-lstm");
    expect(screen.getByText(/待补充/)).toBeInTheDocument();
  });

  it("redirects to 404 for unknown family", () => {
    renderFamily("invalid-family");
    // because invalid-family ≠ any FamilyId in data
    expect(screen.getByText("404")).toBeInTheDocument();
  });
});
```

- [ ] **Step 4**: 创建 `web/src/components/node/NodePage.test.tsx`

```typescript
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router";
import { NodePage } from "./NodePage";

describe("NodePage", () => {
  it("renders ResNet node header", async () => {
    render(
      <MemoryRouter initialEntries={["/families/01-cnn/05-resnet"]}>
        <Routes>
          <Route
            path="/families/:familyId/:nodeSlug"
            element={<NodePage />}
          />
        </Routes>
      </MemoryRouter>
    );
    expect(await screen.findByText(/ResNet/)).toBeInTheDocument();
    expect(screen.getByText(/Kaiming He/)).toBeInTheDocument();
  });

  it("redirects to 404 for unknown node", () => {
    render(
      <MemoryRouter initialEntries={["/families/01-cnn/nonexistent"]}>
        <Routes>
          <Route
            path="/families/:familyId/:nodeSlug"
            element={<NodePage />}
          />
          <Route path="/404" element={<div>404 here</div>} />
        </Routes>
      </MemoryRouter>
    );
    expect(screen.getByText("404 here")).toBeInTheDocument();
  });
});
```

- [ ] **Step 5**: 跑测试

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npm test
```

期望：约 12 个测试（4 套 .test.tsx + lib/motion + lib/colors）全 PASS。

如果有些组件测试因为 `import.meta.glob` 在测试环境下行为不同而失败（NodePage 加载 markdown），可以接受 NodePage test 跳过 markdown 加载、只测 header 渲染。

- [ ] **Step 6**: 提交

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add web/src/App.test.tsx web/src/components/home/HomePage.test.tsx web/src/components/family/FamilyPage.test.tsx web/src/components/node/NodePage.test.tsx
git commit -m "test(web): add tests for router, HomePage toggle, FamilyPage, NodePage"
```

---

### Task 14: 端到端冒烟（spec §11 验收）

**Files:** 无新增

#### Steps

- [ ] **Step 1**: 数据基础

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
python3 scripts/generate_timeline.py
test -s web/src/data/families.json && echo OK || echo FAIL
python3 -c "import json; d=json.load(open('web/src/data/families.json')); assert len(d['families'])==15 and sum(len(f['nodes']) for f in d['families'])==8; print('OK')"
```

期望：`OK / OK / OK`

- [ ] **Step 2**: TS 编译

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npx tsc --noEmit
```

期望：0 错。

- [ ] **Step 3**: Build

```bash
npm run build 2>&1 | tail -10
```

期望：build 成功，输出 `dist/`。

```bash
du -sh dist/assets/*.js | head
```

记录主包大小，应该 < 500 KB（gzip 后）。可以用：

```bash
ls -la dist/assets/*.js | awk '{print $5, $9}'
```

- [ ] **Step 4**: 测试

```bash
npm test 2>&1 | tail -3
```

期望：所有测试 PASS。

- [ ] **Step 5**: dev server 手动验证 3 个路由

```bash
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort &
```

打开 `http://127.0.0.1:5173/`，验证：

- ✅ 主页：标题 + 切换 toggle + 时间轴/家族网格切换有动画
- ✅ Hover 节点点 → 弹卡片（按家族着色）

打开 `http://127.0.0.1:5173/families/01-cnn`：

- ✅ 标题 "CNN 卷积神经网络"
- ✅ 8 节点卡片网格
- ✅ 至少一个 mini-arch 可见（AlexNet/VGG/ResNet 等是真画的）

打开 `http://127.0.0.1:5173/families/01-cnn/05-resnet`：

- ✅ 节点标题、作者、论文
- ✅ markdown 正文渲染
- ✅ Mermaid 块渲染（可能延迟几百毫秒）
- ✅ KaTeX 公式正常

打开 `http://127.0.0.1:5173/nonexistent`：

- ✅ 404 页

按 Ctrl+C 停 dev server。

- [ ] **Step 6**: 工作树清洁

```bash
git status -s
```

期望：除 `.claude/` 等本地未追踪外，无未提交改动。

---

## 完成判定（对齐 spec §11）

| # | 验收 | 落实任务 |
|---|------|--------|
| 1 | `families.json` 含 15 家族 + 8 CNN 节点真实数据 | Task 1 |
| 2 | `family.ts` 与 JSON schema 一致，TS 编译通过 | Task 2 + Task 14 |
| 3 | 旧 timeline.ts / phaseFamily.ts / 旧组件全部删除 | Task 7 |
| 4 | tokens.css 含所有 tokens（15 家族色板）| Task 4 |
| 5 | 自托管字体 import 通 | Task 4 |
| 6 | 8 个 mini-arch 组件可独立 import | Task 6 |
| 7 | 路由 `/` / `/families/:id` / `/families/:id/:slug` / `*` 就位 | Task 8 |
| 8 | 主页 D 方案：时间/家族切换 + hover 卡片 | Task 10 |
| 9 | 家族页 `/families/01-cnn` 渲染 + 8 节点子时间线 | Task 11 |
| 10 | 节点页渲染 markdown（含 Mermaid + SVG + 公式 + 代码）| Task 12 |
| 11 | `npm run build` 成功 + 产物 < 500 KB gzip | Task 14 |
| 12 | `npm test` 测试全过 | Task 14 |
| 13 | dev server 3 路由手动验证通过 | Task 14 |

---

## 后续 plan（不在本计划范围内）

- **W4: CNN 金标本节点**——挑 1 个节点（如 ResNet）做 distill.pub 标准 scrollytelling 可视化
- **W5: CNN 其余 7 节点**——按 W4 模式铺开
- **W6+: 其他 14 家族 web 内容**——每家族独立 plan
- **资产路径修复**——NodePage 引用 SVG（`assets/05-resnet-residual.svg`）当前在 dev server 下 404，需要 Vite 配置把仓库根目录暴露为 static asset
- **Dark mode** 切换 UI
- **移动端响应式深度优化**
- **a11y 完整审计**
