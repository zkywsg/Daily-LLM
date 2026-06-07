# ResNet 残差块 SVG 精品图 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 spec `2026-06-07-resnet-residual-svg-design.md` 落地：写出 `01-cnn/assets/05-resnet-residual.svg`（上下两栏：单 Bottleneck + 梯度高速公路），并把 `01-cnn/05-resnet.md` 里的 `TODO(SVG)` 注释替换为图 3 引用。

**Architecture:** 2 个任务串行——Task 1 写 SVG（brief 提供完整骨架代码，subagent 复制粘贴 + 验证 XML 与渲染）；Task 2 改 markdown + 跑端到端验证。SVG 文件单独可看（GitHub 直接渲染），不需要构建步骤。

**Tech Stack:** SVG 1.1 · XML · Markdown · git

**Spec reference:** `docs/superpowers/specs/2026-06-07-resnet-residual-svg-design.md`

---

## File Structure

新建：

- `01-cnn/assets/` 目录（家族第一份资产）
- `01-cnn/assets/05-resnet-residual.svg` 

修改：

- `01-cnn/05-resnet.md`（删 `<!-- TODO(SVG) -->` 注释 + 插入图 3 引用）

---

### Task 1: 写 SVG 文件

**Files:**
- Create: `01-cnn/assets/` （目录）
- Create: `01-cnn/assets/05-resnet-residual.svg`

#### Brief

写入下面这份**完整 SVG**到 `01-cnn/assets/05-resnet-residual.svg`，逐字符不增不减。所有坐标、颜色、字体大小都按 spec §3 锁死。

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600"
     preserveAspectRatio="xMidYMid meet"
     font-family="system-ui, -apple-system, sans-serif">

  <defs>
    <marker id="arr-main" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 Z" fill="#9d174d"/>
    </marker>
    <marker id="arr-shortcut" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 Z" fill="#2563eb"/>
    </marker>
    <marker id="arr-grad-decay" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="5" markerHeight="5" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 Z" fill="#dc2626"/>
    </marker>
    <marker id="arr-grad-blue" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="5" markerHeight="5" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 Z" fill="#2563eb"/>
    </marker>
  </defs>

  <!-- ============== UPPER PANEL: single Bottleneck ============== -->
  <text x="400" y="28" text-anchor="middle" font-size="16"
        fill="#1f2937" font-weight="600">
    Bottleneck Residual Block
  </text>

  <!-- Input x -->
  <rect x="30" y="190" width="60" height="40" rx="6"
        fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>
  <text x="60" y="215" text-anchor="middle" font-size="14" fill="#92400e">x</text>
  <text x="60" y="254" text-anchor="middle" font-size="11" fill="#92400e">[B,256,H,W]</text>

  <!-- 1×1 Conv ↓ 64 -->
  <rect x="120" y="190" width="80" height="40" rx="6"
        fill="#fce7f3" stroke="#db2777" stroke-width="1.5"/>
  <text x="160" y="208" text-anchor="middle" font-size="13" fill="#9d174d">1×1 Conv ↓</text>
  <text x="160" y="223" text-anchor="middle" font-size="11" fill="#9d174d">64</text>

  <!-- ReLU 1 -->
  <rect x="210" y="190" width="50" height="40" rx="6"
        fill="#ecfdf5" stroke="#059669" stroke-width="1.5"/>
  <text x="235" y="215" text-anchor="middle" font-size="13" fill="#065f46">ReLU</text>

  <!-- 3×3 Conv 64 -->
  <rect x="270" y="190" width="80" height="40" rx="6"
        fill="#fce7f3" stroke="#db2777" stroke-width="1.5"/>
  <text x="310" y="208" text-anchor="middle" font-size="13" fill="#9d174d">3×3 Conv</text>
  <text x="310" y="223" text-anchor="middle" font-size="11" fill="#9d174d">64</text>

  <!-- ReLU 2 -->
  <rect x="360" y="190" width="50" height="40" rx="6"
        fill="#ecfdf5" stroke="#059669" stroke-width="1.5"/>
  <text x="385" y="215" text-anchor="middle" font-size="13" fill="#065f46">ReLU</text>

  <!-- 1×1 Conv ↑ 256 -->
  <rect x="420" y="190" width="80" height="40" rx="6"
        fill="#fce7f3" stroke="#db2777" stroke-width="1.5"/>
  <text x="460" y="208" text-anchor="middle" font-size="13" fill="#9d174d">1×1 Conv ↑</text>
  <text x="460" y="223" text-anchor="middle" font-size="11" fill="#9d174d">256</text>

  <!-- ⊕ Add node -->
  <circle cx="540" cy="210" r="18" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
  <text x="540" y="216" text-anchor="middle" font-size="18" fill="#92400e" font-weight="700">⊕</text>

  <!-- ReLU final -->
  <rect x="578" y="190" width="50" height="40" rx="6"
        fill="#ecfdf5" stroke="#059669" stroke-width="1.5"/>
  <text x="603" y="215" text-anchor="middle" font-size="13" fill="#065f46">ReLU</text>

  <!-- Output y -->
  <rect x="660" y="190" width="60" height="40" rx="6"
        fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>
  <text x="690" y="215" text-anchor="middle" font-size="14" fill="#92400e">y</text>

  <!-- Main path arrows -->
  <g stroke="#9d174d" stroke-width="1.5" fill="none">
    <line x1="92"  y1="210" x2="118" y2="210" marker-end="url(#arr-main)"/>
    <line x1="202" y1="210" x2="208" y2="210" marker-end="url(#arr-main)"/>
    <line x1="262" y1="210" x2="268" y2="210" marker-end="url(#arr-main)"/>
    <line x1="352" y1="210" x2="358" y2="210" marker-end="url(#arr-main)"/>
    <line x1="412" y1="210" x2="418" y2="210" marker-end="url(#arr-main)"/>
    <line x1="502" y1="210" x2="520" y2="210" marker-end="url(#arr-main)"/>
    <line x1="560" y1="210" x2="576" y2="210" marker-end="url(#arr-main)"/>
    <line x1="630" y1="210" x2="658" y2="210" marker-end="url(#arr-main)"/>
  </g>

  <!-- Shortcut arc: from Input top to ⊕ top -->
  <path d="M 60 190 C 60 80, 540 80, 540 192"
        stroke="#2563eb" stroke-width="3" fill="none"
        marker-end="url(#arr-shortcut)"/>

  <!-- identity label on arc -->
  <text x="300" y="78" text-anchor="middle" font-size="13"
        fill="#1e40af" font-style="italic" font-weight="600">
    identity
  </text>

  <!-- F(x)+x label near ⊕ -->
  <text x="555" y="178" text-anchor="start" font-size="14"
        fill="#1f2937" font-style="italic">
    F(x) + x
  </text>

  <!-- F(x) description below main path -->
  <text x="285" y="285" text-anchor="middle" font-size="11" fill="#6b7280">
    F(x) = 1×1↓ → 3×3 → 1×1↑（"瓶颈"结构，节省 ~75% 参数）
  </text>

  <!-- ============== DIVIDER ============== -->
  <line x1="40" y1="380" x2="760" y2="380"
        stroke="#d6d3d1" stroke-width="1" stroke-dasharray="4,4"/>

  <!-- ============== LOWER PANEL: gradient highway ============== -->
  <text x="400" y="416" text-anchor="middle" font-size="16"
        fill="#1f2937" font-weight="600">
    Gradient Highway · 多块串联时梯度回传路径
  </text>

  <!-- 6 simplified blocks: x = 80, 195, 310, 425, 540, 655 -->
  <!-- Block 1 -->
  <rect x="80" y="460" width="70" height="30" rx="4"
        fill="#fce7f3" stroke="#db2777" stroke-width="1.5"/>
  <text x="115" y="480" text-anchor="middle" font-size="11" fill="#9d174d">Block 1</text>
  <path d="M 85 460 C 85 440, 145 440, 145 460"
        stroke="#2563eb" stroke-width="1.5" fill="none"/>

  <!-- Block 2 -->
  <rect x="195" y="460" width="70" height="30" rx="4"
        fill="#fce7f3" stroke="#db2777" stroke-width="1.5"/>
  <text x="230" y="480" text-anchor="middle" font-size="11" fill="#9d174d">Block 2</text>
  <path d="M 200 460 C 200 440, 260 440, 260 460"
        stroke="#2563eb" stroke-width="1.5" fill="none"/>

  <!-- Block 3 -->
  <rect x="310" y="460" width="70" height="30" rx="4"
        fill="#fce7f3" stroke="#db2777" stroke-width="1.5"/>
  <text x="345" y="480" text-anchor="middle" font-size="11" fill="#9d174d">Block 3</text>
  <path d="M 315 460 C 315 440, 375 440, 375 460"
        stroke="#2563eb" stroke-width="1.5" fill="none"/>

  <!-- Block 4 -->
  <rect x="425" y="460" width="70" height="30" rx="4"
        fill="#fce7f3" stroke="#db2777" stroke-width="1.5"/>
  <text x="460" y="480" text-anchor="middle" font-size="11" fill="#9d174d">Block 4</text>
  <path d="M 430 460 C 430 440, 490 440, 490 460"
        stroke="#2563eb" stroke-width="1.5" fill="none"/>

  <!-- Block 5 -->
  <rect x="540" y="460" width="70" height="30" rx="4"
        fill="#fce7f3" stroke="#db2777" stroke-width="1.5"/>
  <text x="575" y="480" text-anchor="middle" font-size="11" fill="#9d174d">Block 5</text>
  <path d="M 545 460 C 545 440, 605 440, 605 460"
        stroke="#2563eb" stroke-width="1.5" fill="none"/>

  <!-- Block 6 -->
  <rect x="655" y="460" width="70" height="30" rx="4"
        fill="#fce7f3" stroke="#db2777" stroke-width="1.5"/>
  <text x="690" y="480" text-anchor="middle" font-size="11" fill="#9d174d">Block 6</text>
  <path d="M 660 460 C 660 440, 720 440, 720 460"
        stroke="#2563eb" stroke-width="1.5" fill="none"/>

  <!-- Forward path arrows between blocks -->
  <g stroke="#9d174d" stroke-width="1" fill="none">
    <line x1="150" y1="475" x2="193" y2="475" marker-end="url(#arr-main)"/>
    <line x1="265" y1="475" x2="308" y2="475" marker-end="url(#arr-main)"/>
    <line x1="380" y1="475" x2="423" y2="475" marker-end="url(#arr-main)"/>
    <line x1="495" y1="475" x2="538" y2="475" marker-end="url(#arr-main)"/>
    <line x1="610" y1="475" x2="653" y2="475" marker-end="url(#arr-main)"/>
  </g>

  <!-- ∂L/∂y starting label (right end) -->
  <text x="740" y="448" text-anchor="middle" font-size="12"
        fill="#1f2937" font-weight="600">∂L/∂y</text>

  <!-- Main path gradient: red dashed, decaying (right→left) -->
  <g fill="none" stroke="#dc2626" stroke-dasharray="4,3">
    <path d="M 720 520 L 660 520" stroke-width="2"/>
    <path d="M 605 520 L 545 520" stroke-width="1.7"/>
    <path d="M 490 520 L 430 520" stroke-width="1.4"/>
    <path d="M 375 520 L 315 520" stroke-width="1.1"/>
    <path d="M 260 520 L 200 520" stroke-width="0.8"/>
    <path d="M 145 520 L 95 520" stroke-width="0.5"
          marker-end="url(#arr-grad-decay)"/>
  </g>

  <!-- Shortcut gradient: blue solid, constant -->
  <line x1="720" y1="555" x2="95" y2="555"
        stroke="#2563eb" stroke-width="2.5"
        marker-end="url(#arr-grad-blue)"/>

  <!-- Gradient path labels -->
  <text x="180" y="514" font-size="11" fill="#dc2626">∂L/∂x（主路：衰减）</text>
  <text x="180" y="572" font-size="11" fill="#2563eb">∂L/∂x（shortcut：不衰减）</text>
</svg>
```

#### Steps

- [ ] **Step 1: 建目录**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
mkdir -p 01-cnn/assets
```

- [ ] **Step 2: 写入 SVG 文件**

把上面整段 `<svg>...</svg>` 内容**逐字符**写入 `01-cnn/assets/05-resnet-residual.svg`。注意：

- 首行 `<svg ...>` 之前不要有任何 XML 声明（GitHub 也接受无声明形式）
- 不要修改任何坐标、颜色、字体值
- 中文字符直接 UTF-8 写入（无需 entity 转义）

- [ ] **Step 3: 验证 SVG 是合法 XML**

```bash
python3 -c "import xml.etree.ElementTree as ET; ET.parse('01-cnn/assets/05-resnet-residual.svg'); print('OK valid XML')"
```

期望：`OK valid XML`。如果失败说明粘贴出错，需要重写。

- [ ] **Step 4: 验证关键元素都在**

```bash
F=01-cnn/assets/05-resnet-residual.svg
echo "viewBox:" && grep -c 'viewBox="0 0 800 600"' $F
echo "shortcut 蓝色:" && grep -c 'stroke="#2563eb" stroke-width="3"' $F
echo "⊕ 节点:" && grep -c '⊕' $F
echo "identity 标签:" && grep -c 'identity' $F
echo "F(x) + x:" && grep -c 'F(x) + x' $F
echo "6 个 Block:" && grep -cE 'Block [1-6]' $F
echo "梯度衰减 (6 段):" && grep -cE 'stroke-width="(2|1\.7|1\.4|1\.1|0\.8|0\.5)"' $F
echo "字体:" && grep -c 'system-ui' $F
echo "文件大小:" && wc -c < $F
```

期望（最低要求）：
- viewBox：1
- shortcut 蓝色：1
- ⊕ 节点：≥ 1
- identity 标签：≥ 1
- F(x) + x：≥ 1
- 6 个 Block：6
- 梯度衰减 6 段：≥ 6（六个不同 stroke-width 各一段）
- 字体：1（font-family 引用）
- 文件大小：< 10 KB（spec §7 风险 3 的约束）

- [ ] **Step 5: 提交**

```bash
git add 01-cnn/assets/05-resnet-residual.svg
git commit -m "feat(01-cnn): add ResNet residual block SVG (Bottleneck + gradient highway)"
```

---

### Task 2: 在 ResNet 节点正文里插入 SVG 引用

**Files:**
- Modify: `01-cnn/05-resnet.md`

#### Brief

把 `01-cnn/05-resnet.md` 中的 SVG TODO 注释整段替换为图 3 的 markdown 引用 + caption。

#### Steps

- [ ] **Step 1: 定位现有 TODO 注释**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
grep -n "TODO(SVG)" 01-cnn/05-resnet.md
```

期望：命中 1 行（HTML 注释 `<!-- TODO(SVG): ... -->`）。

- [ ] **Step 2: 替换为图 3 引用**

把那一行 HTML 注释（整行，包括 `<!-- ... -->` 标签）替换为下面两行：

```markdown
![残差块与梯度高速公路](assets/05-resnet-residual.svg)
*图 3：残差块的弧形 shortcut（上）与多块串联的梯度高速公路（下）。*
```

注意：

- 上下不要加额外空行（让它紧贴上下文，与正文同一段落）—— 但如果原 TODO 注释前后本来就有空行，保持空行
- caption 必须用斜体 `*...*`（不是 `_..._`），与项目惯例一致

- [ ] **Step 3: 验证**

```bash
F=01-cnn/05-resnet.md
echo "TODO 已删:" && (grep -n "TODO(SVG)" $F && echo FAIL) || echo OK
echo "SVG 引用:" && grep -nE "!\[.*\]\(assets/05-resnet-residual\.svg\)" $F
echo "图 3 caption:" && grep -nE "^\*图 3：" $F
echo "Mermaid 仍 = 2:" && grep -cE "^\`\`\`mermaid" $F
echo "frontmatter 7 字段:" && head -10 $F | grep -cE "^(name|year|family|order|paper|authors|key_idea):"
echo "字数（应保持，仅替换注释）:" && python3 -c "import re; t=open('$F').read(); print('total=', len(re.findall(r'[一-鿿]',t)) + len(re.findall(r'[A-Za-z]+',t)))"
```

期望：
- TODO 已删：OK
- SVG 引用：命中 1 行
- 图 3 caption：命中 1 行
- Mermaid 仍 = 2（原 Mermaid 不动）
- frontmatter 7 字段：= 7
- 字数：相对原 3101 字符变化不大（±50），因为只是替换一行注释为两行引用

- [ ] **Step 4: TIMELINE 重生成（验证 frontmatter 仍正常）**

```bash
python3 scripts/generate_timeline.py
```

期望：`wrote /Users/lauzanhing/Desktop/Daily-LLM/TIMELINE.md (8 nodes)`（不变）

- [ ] **Step 5: 测试仍正常**

```bash
python3 -m pytest scripts/test_generate_timeline.py 2>&1 | tail -2
```

期望：`2 passed`

- [ ] **Step 6: 工作树检查**

```bash
git status -s
```

期望：仅 `01-cnn/05-resnet.md` 有改动，加 `.claude/` 等本地未追踪。

- [ ] **Step 7: 提交**

```bash
git add 01-cnn/05-resnet.md
git commit -m "feat(01-cnn): wire ResNet node to new residual SVG as 图 3"
```

---

## 完成判定（对齐 spec §5）

| # | 验收 | 落实任务 |
|---|------|--------|
| 1 | SVG 文件存在且是合法 XML | Task 1 Step 3 |
| 2 | viewBox + system-ui 字体 | Task 1 Step 4 |
| 3 | 配色严格按 §3.3 | Task 1（complete content） |
| 4 | 上栏 7 主路节点 + ⊕ + 弧形 shortcut + identity/F(x)+x 标签 | Task 1（complete content） |
| 5 | 下栏 6 block + 双梯度（红虚线渐细 + 蓝实线恒粗）| Task 1 Step 4 |
| 6 | 上下栏间灰虚线分隔 | Task 1（complete content） |
| 7 | TODO(SVG) 注释已删除 | Task 2 Step 3 |
| 8 | 节点用 ![...]() + 图 3 caption 引用 SVG | Task 2 Step 3 |
| 9 | 节点 mermaid 仍 = 2 | Task 2 Step 3 |
| 10 | TIMELINE 重生成不变（8 nodes）| Task 2 Step 4 |
| 11 | 工作树干净 | Task 2 Step 6 |

---

## 后续 plan（不在本计划范围内）

- 其他节点的 SVG 精品图（按需，比如 Transformer Add & Norm、Diffusion UNet skip）
- 把 SVG 抽取成 Web 端可复用的资产
- 添加 SVG 暗色模式（一份 light + 一份 dark）
