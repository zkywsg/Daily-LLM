# W4 · ResNet 金标本节点页 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 `2026-06-08-w4-resnet-golden-node-design.md`：为 ResNet (`01-cnn/05-resnet`) 路由打造 distill.pub 标准 scrollytelling 节点页，MVP 6 个交互单元（3 阶段 × 2 单元），其他节点继续走 W3 markdown 渲染。

**Architecture:** 9 个串行任务。Task 1-2 建基础工具（prose 提取、合成曲线数据）；Task 3-5 建 widget 层（3 SVG 主图 + 3 控件 + 3 stage 组装）；Task 6 主组件；Task 7 NodePage 路由分流 + 金标本注册表；Task 8 测试；Task 9 端到端冒烟。

**Tech Stack:** TypeScript · React 19 · Framer Motion 11 (`useScroll`, `AnimatePresence`) · d3-scale 4 · d3-shape 3 · CSS Modules · react-markdown · Vite `import.meta.glob`

**Spec reference:** `docs/superpowers/specs/2026-06-08-w4-resnet-golden-node-design.md`

---

## File Structure

新建（13 个文件）：

```
web/src/components/node/golden/
├── index.ts                                  ← Task 7
└── resnet/
    ├── NodePageResNet.tsx                    ← Task 6
    ├── NodePageResNet.module.css             ← Task 6
    ├── NodePageResNet.test.tsx               ← Task 8
    ├── lib/
    │   ├── prose.ts                          ← Task 1
    │   ├── prose.test.ts                     ← Task 1
    │   └── curves.ts                         ← Task 2
    ├── widgets/
    │   ├── DegradationCurves.tsx             ← Task 3
    │   ├── BottleneckSVG.tsx                 ← Task 3
    │   ├── GradientHighwaySVG.tsx            ← Task 3
    │   ├── DepthSlider.tsx                   ← Task 4
    │   ├── BlockTypeToggle.tsx               ← Task 4
    │   └── StackDepthSlider.tsx              ← Task 4
    └── stages/
        ├── DegradationStage.tsx              ← Task 5
        ├── ResidualBlockStage.tsx            ← Task 5
        └── GradientHighwayStage.tsx          ← Task 5
```

修改（1 个文件）：

- `web/src/components/node/NodePage.tsx` —— Task 7 加金标本路由分流

---

### Task 1: prose 提取工具

**Files:**
- Create: `web/src/components/node/golden/resnet/lib/prose.ts`
- Create: `web/src/components/node/golden/resnet/lib/prose.test.ts`

#### Brief

从 ResNet markdown 全文按 H2/H3 章节切出 prose 段。返回一个 `ProseSections` 对象。TDD：先写测试看红，再实现。

#### Steps

- [ ] **Step 1: 创建目录**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
mkdir -p web/src/components/node/golden/resnet/lib
```

- [ ] **Step 2: 创建测试 `lib/prose.test.ts`**

```typescript
import { describe, it, expect } from "vitest";
import { extractProse } from "./prose";

const sampleMarkdown = `---
name: "ResNet"
year: 2015
---

# ResNet (2015)

## 之前卡在哪

第一段卡点。
第二段卡点。

## 核心思想

总览段落。

### 直觉

直觉段落 1。
直觉段落 2。

### 机制

机制公式 $y = F(x) + x$。

## 训练细节

| 维度 | 值 |
|------|---|
| lr | 0.1 |

## 关键代码

\`\`\`python
import torch
\`\`\`

## 影响 / 后续

→ 链接 1
→ 链接 2
`;

describe("extractProse", () => {
  it("removes frontmatter", () => {
    const result = extractProse(sampleMarkdown);
    expect(result.beforeStuckOn).not.toContain("---");
    expect(result.beforeStuckOn).not.toContain("name:");
  });

  it("extracts 之前卡在哪 section", () => {
    const result = extractProse(sampleMarkdown);
    expect(result.beforeStuckOn).toContain("第一段卡点");
    expect(result.beforeStuckOn).toContain("第二段卡点");
    expect(result.beforeStuckOn).not.toContain("总览段落");
  });

  it("extracts 核心思想 section (only its own prose, before any ###)", () => {
    const result = extractProse(sampleMarkdown);
    expect(result.coreInsight).toContain("总览段落");
    expect(result.coreInsight).not.toContain("直觉段落");
  });

  it("extracts 直觉 sub-section", () => {
    const result = extractProse(sampleMarkdown);
    expect(result.intuition).toContain("直觉段落 1");
    expect(result.intuition).toContain("直觉段落 2");
    expect(result.intuition).not.toContain("机制公式");
  });

  it("extracts 机制 sub-section", () => {
    const result = extractProse(sampleMarkdown);
    expect(result.mechanism).toContain("机制公式");
    expect(result.mechanism).not.toContain("训练细节");
  });

  it("extracts 训练细节 section", () => {
    const result = extractProse(sampleMarkdown);
    expect(result.trainingDetails).toContain("lr");
    expect(result.trainingDetails).toContain("0.1");
  });

  it("extracts 关键代码 section including code block", () => {
    const result = extractProse(sampleMarkdown);
    expect(result.keyCode).toContain("import torch");
  });

  it("extracts 影响 / 后续 section", () => {
    const result = extractProse(sampleMarkdown);
    expect(result.aftermath).toContain("链接 1");
    expect(result.aftermath).toContain("链接 2");
  });

  it("returns empty string for missing section gracefully", () => {
    const minimal = "# Title\n\n## 之前卡在哪\n\nonly this section.\n";
    const result = extractProse(minimal);
    expect(result.beforeStuckOn).toContain("only this section");
    expect(result.coreInsight).toBe("");
    expect(result.trainingDetails).toBe("");
  });
});
```

- [ ] **Step 3: 跑测试看红**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npm test -- --run src/components/node/golden/resnet/lib/prose.test.ts 2>&1 | tail -10
```

期望：FAIL（`./prose` 不存在）。

- [ ] **Step 4: 实现 `lib/prose.ts`**

```typescript
export interface ProseSections {
  /** ## 之前卡在哪 章节正文 */
  beforeStuckOn: string;
  /** ## 核心思想 章节自己的正文（不含 ### 子段） */
  coreInsight: string;
  /** ### 直觉 子段正文 */
  intuition: string;
  /** ### 机制 子段正文 */
  mechanism: string;
  /** ## 训练细节 章节正文（含表格） */
  trainingDetails: string;
  /** ## 关键代码 章节正文（含 fenced code block） */
  keyCode: string;
  /** ## 影响 / 后续 章节正文 */
  aftermath: string;
}

/**
 * 从节点 markdown 全文按 H2/H3 章节切出 prose 段。
 * 1) 先剥除 YAML frontmatter（顶部 --- ... --- 块）
 * 2) 按 H2/H3 标题切分内容
 * 3) 把已知章节名映射到对应字段；未匹配的章节忽略
 */
export function extractProse(markdown: string): ProseSections {
  // 剥除 frontmatter
  const body = markdown.replace(/^---[\s\S]*?---\n?/, "");

  const sections: ProseSections = {
    beforeStuckOn: "",
    coreInsight: "",
    intuition: "",
    mechanism: "",
    trainingDetails: "",
    keyCode: "",
    aftermath: "",
  };

  // 章节名 → ProseSections key
  const h2Map: Record<string, keyof ProseSections> = {
    "之前卡在哪": "beforeStuckOn",
    "核心思想": "coreInsight",
    "训练细节": "trainingDetails",
    "关键代码": "keyCode",
    "影响 / 后续": "aftermath",
    "影响 / 后续 ": "aftermath", // 容忍尾部空格
    "影响/后续": "aftermath",
  };
  const h3Map: Record<string, keyof ProseSections> = {
    "直觉": "intuition",
    "机制": "mechanism",
  };

  // 切分：把 body 按行处理，遇到 H2 / H3 标题切换当前 section
  const lines = body.split("\n");
  let currentKey: keyof ProseSections | null = null;
  let buffer: string[] = [];

  const flush = () => {
    if (currentKey) {
      sections[currentKey] = buffer.join("\n").trim();
    }
    buffer = [];
  };

  for (const line of lines) {
    const h2Match = /^## +(.+?)\s*$/.exec(line);
    const h3Match = /^### +(.+?)\s*$/.exec(line);

    if (h2Match) {
      flush();
      const name = h2Match[1].trim();
      currentKey = h2Map[name] ?? null;
      continue;
    }

    if (h3Match && (currentKey === "coreInsight" || currentKey === "intuition" || currentKey === "mechanism")) {
      // 在 核心思想 / 直觉 / 机制 上下文中遇到 H3 → 切换
      flush();
      const name = h3Match[1].trim();
      currentKey = h3Map[name] ?? null;
      continue;
    }

    if (h3Match) {
      // 其他 H2 上下文中遇到 H3：当成内容一部分而非分隔（保留行）
      buffer.push(line);
      continue;
    }

    if (currentKey) {
      buffer.push(line);
    }
  }
  flush();

  return sections;
}
```

- [ ] **Step 5: 跑测试看绿**

```bash
npm test -- --run src/components/node/golden/resnet/lib/prose.test.ts 2>&1 | tail -10
```

期望：9 个测试全 PASS。

- [ ] **Step 6: 提交**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add web/src/components/node/golden/resnet/lib/prose.ts web/src/components/node/golden/resnet/lib/prose.test.ts
git commit -m "feat(web/resnet): add prose section extractor with 9 tests"
```

---

### Task 2: 合成曲线数据

**Files:**
- Create: `web/src/components/node/golden/resnet/lib/curves.ts`

#### Brief

退化曲线的合成数据生成器。不引用论文真实数据，用公式合成 plain 和 ResNet 两条 training error 曲线。

#### Steps

- [ ] **Step 1: 创建 `lib/curves.ts`**

```typescript
export interface CurvePoint {
  epoch: number;
  loss: number;
}

const EPOCHS = 200;

/**
 * Plain CNN 训练损失曲线。
 * - 浅模型（depth ≤ 30）：单调指数衰减到 ~0.05
 * - 深模型（depth > 30）：先降后升的 "退化" U 形——越深越糟
 */
export function plainCurve(depth: number, epochs = EPOCHS): CurvePoint[] {
  const isDeep = depth > 30;
  const points: CurvePoint[] = [];

  if (!isDeep) {
    // 浅：标准指数衰减
    for (let i = 1; i <= epochs; i++) {
      const loss = 0.6 * Math.exp(-i / 35) + 0.05;
      points.push({ epoch: i, loss });
    }
    return points;
  }

  // 深：min epoch 与 final loss 都随深度恶化
  const minEpoch = 50 + (depth - 30) * 1.0;
  const minLoss = 0.1 + (depth - 30) * 0.0015;
  const finalLoss = minLoss + (depth - 30) * 0.0025;

  for (let i = 1; i <= epochs; i++) {
    // 下降段：到 minEpoch 前指数衰减接近 minLoss
    const downLoss = (0.65 - minLoss) * Math.exp(-i / (minEpoch * 0.4)) + minLoss;
    // 上升段：minEpoch 之后线性增长到 finalLoss
    const rise = Math.max(0, (i - minEpoch) / (epochs - minEpoch));
    const upLoss = minLoss + rise * (finalLoss - minLoss);
    points.push({ epoch: i, loss: Math.max(downLoss, upLoss) });
  }
  return points;
}

/**
 * ResNet 训练损失曲线。
 * - 任何深度都稳定指数衰减到 ~0.04
 * - 深度越深，时间常数稍大（收敛略慢），但最终损失差不多
 */
export function resnetCurve(depth: number, epochs = EPOCHS): CurvePoint[] {
  const finalLoss = 0.04;
  const tau = 30 + Math.sqrt(depth) * 3;
  const points: CurvePoint[] = [];
  for (let i = 1; i <= epochs; i++) {
    const loss = (0.65 - finalLoss) * Math.exp(-i / tau) + finalLoss;
    points.push({ epoch: i, loss });
  }
  return points;
}

/** BasicBlock 参数量（C=64 通道，2 个 3×3 conv） */
export const BASIC_BLOCK_PARAMS = 2 * (3 * 3 * 64 * 64); // 73,728

/** Bottleneck 参数量（1×1↓256→64 + 3×3 64 + 1×1↑64→256） */
export const BOTTLENECK_PARAMS =
  1 * 1 * 256 * 64 + 3 * 3 * 64 * 64 + 1 * 1 * 64 * 256; // 70,144

/** 梯度公路：每经过一个 plain block，梯度幅度衰减系数 */
export const GRADIENT_DECAY_PER_BLOCK = 0.85;
```

- [ ] **Step 2: TS 编译检查**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npx tsc --noEmit 2>&1 | grep "curves" | head -5
```

期望：无 curves 相关错误。

- [ ] **Step 3: 简单 sanity check**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
node --input-type=module -e "
import { plainCurve, resnetCurve, BASIC_BLOCK_PARAMS, BOTTLENECK_PARAMS } from './src/components/node/golden/resnet/lib/curves.ts'
" 2>&1 | head -3 || echo "(node 不能直接 import TS，跳过 runtime sanity)"
```

跳过运行时验证（Vite 处理 TS）。手验逻辑（也可写测试，但合成数据非业务关键路径，省）：

```bash
echo "BASIC_BLOCK_PARAMS = 73728" && python3 -c "print(2*(3*3*64*64))"
echo "BOTTLENECK_PARAMS = 70144" && python3 -c "print(1*1*256*64 + 3*3*64*64 + 1*1*64*256)"
```

期望：分别 73728 / 70144。

- [ ] **Step 4: 提交**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add web/src/components/node/golden/resnet/lib/curves.ts
git commit -m "feat(web/resnet): add synthetic degradation/resnet curve data + block params"
```

---

### Task 3: 三个 SVG 主图 widget

**Files:**
- Create: `web/src/components/node/golden/resnet/widgets/DegradationCurves.tsx`
- Create: `web/src/components/node/golden/resnet/widgets/BottleneckSVG.tsx`
- Create: `web/src/components/node/golden/resnet/widgets/GradientHighwaySVG.tsx`

#### Brief

3 个右面板主图 SVG 组件，受外部 props 驱动状态变化。

#### Steps

- [ ] **Step 1: 创建 `widgets/DegradationCurves.tsx`**

```typescript
import { scaleLinear } from "d3-scale";
import { line } from "d3-shape";
import { plainCurve, resnetCurve } from "../lib/curves";

interface Props {
  /** 当前选中的网络深度 */
  depth: number;
  width?: number;
  height?: number;
}

const PADDING = { top: 30, right: 30, bottom: 50, left: 60 };

export function DegradationCurves({ depth, width = 560, height = 360 }: Props) {
  const plain = plainCurve(depth);
  const resnet = resnetCurve(depth);

  const innerW = width - PADDING.left - PADDING.right;
  const innerH = height - PADDING.top - PADDING.bottom;

  const xScale = scaleLinear().domain([0, 200]).range([0, innerW]);
  const yScale = scaleLinear().domain([0, 0.7]).range([innerH, 0]);

  const lineGen = line<{ epoch: number; loss: number }>()
    .x((d) => xScale(d.epoch))
    .y((d) => yScale(d.loss));

  const plainPath = lineGen(plain) ?? "";
  const resnetPath = lineGen(resnet) ?? "";

  // X 轴 ticks
  const xTicks = [0, 50, 100, 150, 200];
  // Y 轴 ticks
  const yTicks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      height={height}
      role="img"
      aria-label={`训练损失曲线对比 ${depth} 层 plain 与 ResNet`}
    >
      <g transform={`translate(${PADDING.left}, ${PADDING.top})`}>
        {/* Axes */}
        <line x1={0} y1={innerH} x2={innerW} y2={innerH} stroke="var(--border)" />
        <line x1={0} y1={0} x2={0} y2={innerH} stroke="var(--border)" />

        {/* X ticks */}
        {xTicks.map((t) => (
          <g key={t} transform={`translate(${xScale(t)}, ${innerH})`}>
            <line y2={6} stroke="var(--ink-muted)" />
            <text
              y={20}
              textAnchor="middle"
              fontSize={11}
              fill="var(--ink-muted)"
            >
              {t}
            </text>
          </g>
        ))}
        <text
          x={innerW / 2}
          y={innerH + 40}
          textAnchor="middle"
          fontSize={12}
          fill="var(--ink-secondary)"
        >
          epoch
        </text>

        {/* Y ticks */}
        {yTicks.map((t) => (
          <g key={t} transform={`translate(0, ${yScale(t)})`}>
            <line x2={-6} stroke="var(--ink-muted)" />
            <text
              x={-10}
              y={4}
              textAnchor="end"
              fontSize={11}
              fill="var(--ink-muted)"
            >
              {t.toFixed(1)}
            </text>
          </g>
        ))}
        <text
          transform={`translate(-44, ${innerH / 2}) rotate(-90)`}
          textAnchor="middle"
          fontSize={12}
          fill="var(--ink-secondary)"
        >
          training error
        </text>

        {/* Plain curve (red, dashed if it degrades) */}
        <path
          d={plainPath}
          stroke="#dc2626"
          strokeWidth={2}
          fill="none"
          strokeDasharray={depth > 30 ? "5,3" : "none"}
        />
        {/* ResNet curve (blue, solid) */}
        <path d={resnetPath} stroke="#2563eb" strokeWidth={2.5} fill="none" />

        {/* Legend */}
        <g transform={`translate(${innerW - 140}, 10)`}>
          <line x1={0} y1={6} x2={20} y2={6} stroke="#dc2626" strokeWidth={2} />
          <text x={26} y={10} fontSize={12} fill="var(--ink-primary)">
            plain ({depth} 层)
          </text>
          <line
            x1={0}
            y1={26}
            x2={20}
            y2={26}
            stroke="#2563eb"
            strokeWidth={2.5}
          />
          <text x={26} y={30} fontSize={12} fill="var(--ink-primary)">
            ResNet ({depth} 层)
          </text>
        </g>
      </g>
    </svg>
  );
}
```

- [ ] **Step 2: 创建 `widgets/BottleneckSVG.tsx`**

```typescript
import { BASIC_BLOCK_PARAMS, BOTTLENECK_PARAMS } from "../lib/curves";

interface Props {
  blockType: "basic" | "bottleneck";
  width?: number;
  height?: number;
}

export function BottleneckSVG({
  blockType,
  width = 560,
  height = 360,
}: Props) {
  const isBasic = blockType === "basic";
  const params = isBasic ? BASIC_BLOCK_PARAMS : BOTTLENECK_PARAMS;

  // 节点定义：取决于 block 类型
  const layers = isBasic
    ? [
        { label: "Conv 3×3", sub: "64", x: 100 },
        { label: "ReLU", sub: "", x: 200 },
        { label: "Conv 3×3", sub: "64", x: 290 },
      ]
    : [
        { label: "Conv 1×1 ↓", sub: "64", x: 90 },
        { label: "ReLU", sub: "", x: 180 },
        { label: "Conv 3×3", sub: "64", x: 240 },
        { label: "ReLU", sub: "", x: 330 },
        { label: "Conv 1×1 ↑", sub: "256", x: 380 },
      ];

  const inputX = 20;
  const addX = isBasic ? 380 : 470;
  const outputX = isBasic ? 460 : 550;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      height={height}
      role="img"
      aria-label={`${blockType === "basic" ? "BasicBlock" : "Bottleneck"} 残差块结构`}
    >
      {/* Title */}
      <text
        x={width / 2}
        y={30}
        textAnchor="middle"
        fontSize={16}
        fontWeight={600}
        fill="var(--ink-primary)"
      >
        {isBasic ? "BasicBlock" : "Bottleneck"}
      </text>

      {/* Input x */}
      <rect
        x={inputX}
        y={180}
        width={50}
        height={36}
        rx={6}
        fill="#fef3c7"
        stroke="#d97706"
        strokeWidth={1.5}
      />
      <text x={inputX + 25} y={203} textAnchor="middle" fontSize={14} fill="#92400e">
        x
      </text>

      {/* Main path layers */}
      {layers.map((layer, i) => (
        <g key={i}>
          <rect
            x={layer.x}
            y={180}
            width={70}
            height={36}
            rx={6}
            fill="#fce7f3"
            stroke="#db2777"
            strokeWidth={1.5}
          />
          <text
            x={layer.x + 35}
            y={layer.sub ? 197 : 203}
            textAnchor="middle"
            fontSize={12}
            fill="#9d174d"
          >
            {layer.label}
          </text>
          {layer.sub && (
            <text
              x={layer.x + 35}
              y={210}
              textAnchor="middle"
              fontSize={10}
              fill="#9d174d"
            >
              {layer.sub}
            </text>
          )}
        </g>
      ))}

      {/* ⊕ Add node */}
      <circle
        cx={addX}
        cy={198}
        r={16}
        fill="#fef3c7"
        stroke="#d97706"
        strokeWidth={2}
      />
      <text
        x={addX}
        y={204}
        textAnchor="middle"
        fontSize={18}
        fill="#92400e"
        fontWeight={700}
      >
        ⊕
      </text>

      {/* Output y */}
      <rect
        x={outputX}
        y={180}
        width={50}
        height={36}
        rx={6}
        fill="#fef3c7"
        stroke="#d97706"
        strokeWidth={1.5}
      />
      <text x={outputX + 25} y={203} textAnchor="middle" fontSize={14} fill="#92400e">
        y
      </text>

      {/* Shortcut arc: input top → ⊕ top */}
      <path
        d={`M ${inputX + 25} 180 C ${inputX + 25} 80, ${addX} 80, ${addX} 184`}
        stroke="#2563eb"
        strokeWidth={3}
        fill="none"
      />
      <text
        x={(inputX + addX) / 2}
        y={75}
        textAnchor="middle"
        fontSize={13}
        fontStyle="italic"
        fill="#1e40af"
      >
        identity
      </text>

      {/* F(x) + x label */}
      <text
        x={addX + 24}
        y={170}
        textAnchor="start"
        fontSize={14}
        fontStyle="italic"
        fill="var(--ink-primary)"
      >
        F(x) + x
      </text>

      {/* Params display */}
      <g transform={`translate(${width / 2}, 285)`}>
        <text
          textAnchor="middle"
          fontSize={13}
          fill="var(--ink-secondary)"
        >
          参数量
        </text>
        <text
          y={26}
          textAnchor="middle"
          fontSize={22}
          fontWeight={600}
          fill="var(--ink-primary)"
        >
          {params.toLocaleString()}
        </text>
      </g>
    </svg>
  );
}
```

- [ ] **Step 3: 创建 `widgets/GradientHighwaySVG.tsx`**

```typescript
import { GRADIENT_DECAY_PER_BLOCK } from "../lib/curves";

interface Props {
  stackDepth: number; // 2 / 6 / 20 / 50
  width?: number;
  height?: number;
}

const PADDING_X = 30;
const BLOCK_W = 50;
const BLOCK_H = 30;
const BLOCK_Y = 160;

export function GradientHighwaySVG({
  stackDepth,
  width = 560,
  height = 360,
}: Props) {
  const innerW = width - 2 * PADDING_X;

  // 显示 block 数量上限避免画面拥挤；超过则在中间用 "..." 代替
  const visibleCount = Math.min(stackDepth, 8);
  const blockGap =
    visibleCount > 1 ? (innerW - visibleCount * BLOCK_W) / (visibleCount - 1) : 0;

  const blocks = Array.from({ length: visibleCount }, (_, i) => ({
    index: i,
    x: PADDING_X + i * (BLOCK_W + blockGap),
  }));

  // 主路梯度：最右端到达 input 时残留比例 0.85^stackDepth
  const totalDecay = Math.pow(GRADIENT_DECAY_PER_BLOCK, stackDepth);
  // 红虚线分段，每段越往左越细
  const decayWidths = blocks.map((_, i) => {
    const remaining = Math.pow(GRADIENT_DECAY_PER_BLOCK, stackDepth - i);
    return Math.max(0.3, 2.5 * remaining);
  });

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      height={height}
      role="img"
      aria-label={`${stackDepth} 块串联的梯度高速公路`}
    >
      {/* Title */}
      <text
        x={width / 2}
        y={30}
        textAnchor="middle"
        fontSize={16}
        fontWeight={600}
        fill="var(--ink-primary)"
      >
        Gradient Highway · {stackDepth} 块串联
      </text>

      {/* Blocks */}
      {blocks.map((b) => (
        <g key={b.index}>
          <rect
            x={b.x}
            y={BLOCK_Y}
            width={BLOCK_W}
            height={BLOCK_H}
            rx={4}
            fill="#fce7f3"
            stroke="#db2777"
            strokeWidth={1.5}
          />
          <text
            x={b.x + BLOCK_W / 2}
            y={BLOCK_Y + 19}
            textAnchor="middle"
            fontSize={10}
            fill="#9d174d"
          >
            B{b.index + 1}
          </text>
          {/* small shortcut arc */}
          <path
            d={`M ${b.x + 5} ${BLOCK_Y} C ${b.x + 5} ${BLOCK_Y - 20}, ${
              b.x + BLOCK_W - 5
            } ${BLOCK_Y - 20}, ${b.x + BLOCK_W - 5} ${BLOCK_Y}`}
            stroke="#2563eb"
            strokeWidth={1.5}
            fill="none"
          />
          {/* Forward arrow */}
          {b.index < visibleCount - 1 && (
            <line
              x1={b.x + BLOCK_W}
              y1={BLOCK_Y + BLOCK_H / 2}
              x2={b.x + BLOCK_W + blockGap}
              y2={BLOCK_Y + BLOCK_H / 2}
              stroke="#9d174d"
              strokeWidth={1}
            />
          )}
        </g>
      ))}

      {/* "..." in middle if stack is bigger than visible */}
      {stackDepth > visibleCount && (
        <text
          x={width / 2}
          y={BLOCK_Y - 30}
          textAnchor="middle"
          fontSize={11}
          fill="var(--ink-muted)"
        >
          显示前 {visibleCount} 块 · 实际 {stackDepth} 块
        </text>
      )}

      {/* ∂L/∂y starting label */}
      <text
        x={width - PADDING_X + 5}
        y={BLOCK_Y - 5}
        fontSize={12}
        fontWeight={600}
        fill="var(--ink-primary)"
      >
        ∂L/∂y
      </text>

      {/* Main path gradient (red dashed, decaying) */}
      {blocks
        .slice()
        .reverse()
        .map((b, i) => {
          const widthHere = decayWidths[b.index];
          return (
            <line
              key={`grad-main-${b.index}`}
              x1={b.x + BLOCK_W}
              y1={BLOCK_Y + 60}
              x2={b.x}
              y2={BLOCK_Y + 60}
              stroke="#dc2626"
              strokeWidth={widthHere}
              strokeDasharray="4,3"
            />
          );
        })}

      {/* Shortcut gradient (blue solid, constant) */}
      <line
        x1={width - PADDING_X}
        y1={BLOCK_Y + 90}
        x2={PADDING_X}
        y2={BLOCK_Y + 90}
        stroke="#2563eb"
        strokeWidth={2.5}
      />

      {/* Path labels */}
      <text x={PADDING_X} y={BLOCK_Y + 55} fontSize={11} fill="#dc2626">
        主路：到达 input 时残留 {(totalDecay * 100).toFixed(1)}%
      </text>
      <text x={PADDING_X} y={BLOCK_Y + 110} fontSize={11} fill="#2563eb">
        shortcut：恒粗，无衰减
      </text>
    </svg>
  );
}
```

- [ ] **Step 4: 编译验证**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npx tsc --noEmit 2>&1 | grep -E "widgets/(DegradationCurves|BottleneckSVG|GradientHighwaySVG)" | head -10
```

期望：无错。

- [ ] **Step 5: 提交**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add web/src/components/node/golden/resnet/widgets/DegradationCurves.tsx \
        web/src/components/node/golden/resnet/widgets/BottleneckSVG.tsx \
        web/src/components/node/golden/resnet/widgets/GradientHighwaySVG.tsx
git commit -m "feat(web/resnet): add 3 SVG main visualizations (curves / bottleneck / gradient highway)"
```

---

### Task 4: 三个控件 widget

**Files:**
- Create: `web/src/components/node/golden/resnet/widgets/DepthSlider.tsx`
- Create: `web/src/components/node/golden/resnet/widgets/BlockTypeToggle.tsx`
- Create: `web/src/components/node/golden/resnet/widgets/StackDepthSlider.tsx`

#### Brief

3 个左面板内嵌控件。逻辑都是受控组件，回调 `onChange`。

#### Steps

- [ ] **Step 1: 创建 `widgets/DepthSlider.tsx`**

```typescript
interface Props {
  value: number;
  onChange: (depth: number) => void;
}

const DEPTH_OPTIONS = [8, 20, 56, 110, 152];

export function DepthSlider({ value, onChange }: Props) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "var(--space-3)",
        padding: "var(--space-3) var(--space-4)",
        background: "var(--bg-subtle)",
        borderRadius: "var(--radius-md)",
        margin: "var(--space-4) 0",
      }}
    >
      <span
        style={{
          fontSize: "var(--fs-sm)",
          color: "var(--ink-secondary)",
          fontWeight: 500,
        }}
      >
        网络深度：
      </span>
      {DEPTH_OPTIONS.map((d) => (
        <button
          key={d}
          onClick={() => onChange(d)}
          style={{
            padding: "var(--space-1) var(--space-3)",
            borderRadius: "var(--radius-full)",
            fontSize: "var(--fs-sm)",
            fontWeight: 500,
            background:
              value === d ? "var(--accent-link)" : "var(--bg-surface)",
            color:
              value === d ? "var(--bg-surface)" : "var(--ink-secondary)",
            border: `1px solid ${
              value === d ? "var(--accent-link)" : "var(--border)"
            }`,
            transition: "all var(--dur-fast) var(--ease-out)",
          }}
        >
          {d}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: 创建 `widgets/BlockTypeToggle.tsx`**

```typescript
interface Props {
  value: "basic" | "bottleneck";
  onChange: (blockType: "basic" | "bottleneck") => void;
}

export function BlockTypeToggle({ value, onChange }: Props) {
  return (
    <div
      style={{
        display: "inline-flex",
        background: "var(--bg-subtle)",
        borderRadius: "var(--radius-full)",
        padding: "var(--space-1)",
        margin: "var(--space-4) 0",
      }}
    >
      <button
        onClick={() => onChange("basic")}
        style={{
          padding: "var(--space-2) var(--space-4)",
          borderRadius: "var(--radius-full)",
          fontSize: "var(--fs-sm)",
          fontWeight: 500,
          color:
            value === "basic"
              ? "var(--ink-primary)"
              : "var(--ink-secondary)",
          background:
            value === "basic" ? "var(--bg-surface)" : "transparent",
          boxShadow:
            value === "basic" ? "var(--shadow-sm)" : "none",
        }}
      >
        BasicBlock
      </button>
      <button
        onClick={() => onChange("bottleneck")}
        style={{
          padding: "var(--space-2) var(--space-4)",
          borderRadius: "var(--radius-full)",
          fontSize: "var(--fs-sm)",
          fontWeight: 500,
          color:
            value === "bottleneck"
              ? "var(--ink-primary)"
              : "var(--ink-secondary)",
          background:
            value === "bottleneck" ? "var(--bg-surface)" : "transparent",
          boxShadow:
            value === "bottleneck" ? "var(--shadow-sm)" : "none",
        }}
      >
        Bottleneck
      </button>
    </div>
  );
}
```

- [ ] **Step 3: 创建 `widgets/StackDepthSlider.tsx`**

```typescript
interface Props {
  value: number;
  onChange: (depth: number) => void;
}

const STACK_OPTIONS = [2, 6, 20, 50];

export function StackDepthSlider({ value, onChange }: Props) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "var(--space-3)",
        padding: "var(--space-3) var(--space-4)",
        background: "var(--bg-subtle)",
        borderRadius: "var(--radius-md)",
        margin: "var(--space-4) 0",
      }}
    >
      <span
        style={{
          fontSize: "var(--fs-sm)",
          color: "var(--ink-secondary)",
          fontWeight: 500,
        }}
      >
        堆叠块数：
      </span>
      {STACK_OPTIONS.map((d) => (
        <button
          key={d}
          onClick={() => onChange(d)}
          style={{
            padding: "var(--space-1) var(--space-3)",
            borderRadius: "var(--radius-full)",
            fontSize: "var(--fs-sm)",
            fontWeight: 500,
            background:
              value === d ? "var(--accent-link)" : "var(--bg-surface)",
            color:
              value === d ? "var(--bg-surface)" : "var(--ink-secondary)",
            border: `1px solid ${
              value === d ? "var(--accent-link)" : "var(--border)"
            }`,
            transition: "all var(--dur-fast) var(--ease-out)",
          }}
        >
          {d}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 4: 编译验证 + 提交**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npx tsc --noEmit 2>&1 | grep -E "widgets/(Depth|Block|Stack)" | head -5
```

期望：无错。

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add web/src/components/node/golden/resnet/widgets/DepthSlider.tsx \
        web/src/components/node/golden/resnet/widgets/BlockTypeToggle.tsx \
        web/src/components/node/golden/resnet/widgets/StackDepthSlider.tsx
git commit -m "feat(web/resnet): add 3 interactive controls (depth slider / block toggle / stack slider)"
```

---

### Task 5: 三个 Stage 组装

**Files:**
- Create: `web/src/components/node/golden/resnet/stages/DegradationStage.tsx`
- Create: `web/src/components/node/golden/resnet/stages/ResidualBlockStage.tsx`
- Create: `web/src/components/node/golden/resnet/stages/GradientHighwayStage.tsx`

#### Brief

每个 stage = 左侧 prose（react-markdown 渲染）+ 嵌入的 widget 控件 + 右侧 SVG 主图。组件签名一致：接收 prose 文本 + state + onChange。

#### Steps

- [ ] **Step 1: 创建 `stages/DegradationStage.tsx`**

```typescript
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { DepthSlider } from "../widgets/DepthSlider";
import { DegradationCurves } from "../widgets/DegradationCurves";

interface Props {
  beforeStuckOnProse: string;
  coreInsightProse: string;
  depth: number;
  onDepthChange: (d: number) => void;
}

export function DegradationStage({
  beforeStuckOnProse,
  coreInsightProse,
  depth,
  onDepthChange,
}: Props) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: "var(--space-8)",
        alignItems: "start",
      }}
    >
      {/* 左：prose + 控件 */}
      <div>
        <h2 style={{ fontSize: "var(--fs-2xl)", marginBottom: "var(--space-4)" }}>
          之前卡在哪
        </h2>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
          >
            {beforeStuckOnProse}
          </ReactMarkdown>
        </div>

        <h2
          style={{
            fontSize: "var(--fs-2xl)",
            marginTop: "var(--space-8)",
            marginBottom: "var(--space-4)",
          }}
        >
          核心思想
        </h2>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
          >
            {coreInsightProse}
          </ReactMarkdown>
        </div>

        <DepthSlider value={depth} onChange={onDepthChange} />

        <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
          滑动选择网络深度，右图实时变化。深度超过 30 层后，plain 网络会出现"先降后升"的退化形态。
        </p>
      </div>

      {/* 右：可视化 */}
      <div style={{ position: "sticky", top: "var(--space-8)" }}>
        <DegradationCurves depth={depth} />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 创建 `stages/ResidualBlockStage.tsx`**

```typescript
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { BlockTypeToggle } from "../widgets/BlockTypeToggle";
import { BottleneckSVG } from "../widgets/BottleneckSVG";

interface Props {
  intuitionProse: string;
  blockType: "basic" | "bottleneck";
  onBlockTypeChange: (t: "basic" | "bottleneck") => void;
}

export function ResidualBlockStage({
  intuitionProse,
  blockType,
  onBlockTypeChange,
}: Props) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: "var(--space-8)",
        alignItems: "start",
      }}
    >
      <div>
        <h2 style={{ fontSize: "var(--fs-2xl)", marginBottom: "var(--space-4)" }}>
          残差的直觉
        </h2>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
          >
            {intuitionProse}
          </ReactMarkdown>
        </div>

        <BlockTypeToggle value={blockType} onChange={onBlockTypeChange} />

        <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
          切换两种 block 结构，看参数量与内部结构变化。Bottleneck 用 1×1 降维节省 ~75% 参数。
        </p>
      </div>

      <div style={{ position: "sticky", top: "var(--space-8)" }}>
        <BottleneckSVG blockType={blockType} />
      </div>
    </div>
  );
}
```

- [ ] **Step 3: 创建 `stages/GradientHighwayStage.tsx`**

```typescript
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { StackDepthSlider } from "../widgets/StackDepthSlider";
import { GradientHighwaySVG } from "../widgets/GradientHighwaySVG";

interface Props {
  mechanismProse: string;
  stackDepth: number;
  onStackDepthChange: (d: number) => void;
}

export function GradientHighwayStage({
  mechanismProse,
  stackDepth,
  onStackDepthChange,
}: Props) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: "var(--space-8)",
        alignItems: "start",
      }}
    >
      <div>
        <h2 style={{ fontSize: "var(--fs-2xl)", marginBottom: "var(--space-4)" }}>
          梯度高速公路
        </h2>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
          >
            {mechanismProse}
          </ReactMarkdown>
        </div>

        <StackDepthSlider
          value={stackDepth}
          onChange={onStackDepthChange}
        />

        <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
          滑动改变堆叠块数。主路梯度按 0.85^N 衰减，shortcut 路径无衰减直达底层。
        </p>
      </div>

      <div style={{ position: "sticky", top: "var(--space-8)" }}>
        <GradientHighwaySVG stackDepth={stackDepth} />
      </div>
    </div>
  );
}
```

- [ ] **Step 4: 编译 + 提交**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npx tsc --noEmit 2>&1 | grep "stages/" | head -5
```

期望：无错。

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add web/src/components/node/golden/resnet/stages/
git commit -m "feat(web/resnet): add 3 stage components combining prose + control + SVG"
```

---

### Task 6: 主组件 `NodePageResNet`

**Files:**
- Create: `web/src/components/node/golden/resnet/NodePageResNet.tsx`
- Create: `web/src/components/node/golden/resnet/NodePageResNet.module.css`

#### Brief

页面入口：hero + 3 stage（垂直堆叠）+ 训练细节 + 关键代码 + 影响后续。所有 widget state 在此 useState。

注意：MVP 不实现"sticky 切换"——3 个 stage 垂直顺序排列，每个 stage 内部右图 sticky 在 stage 高度内。整页滚动自然出现 3 段连续的"右图固定一段时间"。这比同步 scroll 状态切换更简单，效果接近 distill。

#### Steps

- [ ] **Step 1: 创建 `NodePageResNet.module.css`**

```css
.container {
  padding: 0;
  background: var(--bg-canvas);
}

.hero {
  max-width: 1000px;
  margin: 0 auto;
  padding: var(--space-12) var(--space-8);
  text-align: center;
}

.back {
  display: inline-block;
  margin-bottom: var(--space-6);
  font-size: var(--fs-sm);
  color: var(--ink-secondary);
}

.title {
  font-size: var(--fs-5xl);
  margin: var(--space-4) 0;
  background: linear-gradient(
    90deg,
    var(--family-01) 0%,
    var(--accent-link) 100%
  );
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.metaLine {
  font-size: var(--fs-sm);
  color: var(--ink-muted);
  margin: var(--space-1) 0;
}

.keyIdea {
  font-size: var(--fs-lg);
  color: var(--ink-secondary);
  font-style: italic;
  max-width: 700px;
  margin: var(--space-6) auto 0;
  line-height: 1.5;
}

.stage {
  max-width: 1300px;
  margin: 0 auto;
  padding: var(--space-24) var(--space-8);
  min-height: 100vh;
}

.stageAlt {
  background: var(--bg-surface);
}

.footer {
  max-width: 800px;
  margin: 0 auto;
  padding: var(--space-16) var(--space-8);
  font-family: var(--font-serif);
  font-size: var(--fs-md);
  line-height: 1.7;
}

.footerSection {
  margin: var(--space-12) 0;
}

.footerSection h2 {
  font-family: var(--font-sans);
  font-size: var(--fs-2xl);
  margin-bottom: var(--space-4);
}
```

- [ ] **Step 2: 创建 `NodePageResNet.tsx`**

```typescript
import { useState } from "react";
import { Link } from "react-router";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import "highlight.js/styles/github.css";

import resnetMarkdown from "../../../../../../01-cnn/05-resnet.md?raw";
import { extractProse } from "./lib/prose";
import { DegradationStage } from "./stages/DegradationStage";
import { ResidualBlockStage } from "./stages/ResidualBlockStage";
import { GradientHighwayStage } from "./stages/GradientHighwayStage";
import styles from "./NodePageResNet.module.css";

const prose = extractProse(resnetMarkdown);

export default function NodePageResNet() {
  const [depth, setDepth] = useState(56);
  const [blockType, setBlockType] = useState<"basic" | "bottleneck">(
    "bottleneck"
  );
  const [stackDepth, setStackDepth] = useState(6);

  return (
    <div className={styles.container}>
      {/* Hero */}
      <section className={styles.hero}>
        <Link to="/families/01-cnn" className={styles.back}>
          ← 返回 CNN 卷积神经网络
        </Link>
        <h1 className={styles.title}>ResNet (2015)</h1>
        <div className={styles.metaLine}>
          作者：Kaiming He · Xiangyu Zhang · Shaoqing Ren · Jian Sun
        </div>
        <div className={styles.metaLine}>
          论文：Deep Residual Learning for Image Recognition
        </div>
        <p className={styles.keyIdea}>
          用 shortcut 让网络只学残差修正而不是从零重建映射，把 152 层稳定训练变成可能
        </p>
      </section>

      {/* Stage 1: 退化曲线 */}
      <section className={styles.stage}>
        <DegradationStage
          beforeStuckOnProse={prose.beforeStuckOn}
          coreInsightProse={prose.coreInsight}
          depth={depth}
          onDepthChange={setDepth}
        />
      </section>

      {/* Stage 2: 残差块拆解 */}
      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <ResidualBlockStage
          intuitionProse={prose.intuition}
          blockType={blockType}
          onBlockTypeChange={setBlockType}
        />
      </section>

      {/* Stage 3: 梯度高速公路 */}
      <section className={styles.stage}>
        <GradientHighwayStage
          mechanismProse={prose.mechanism}
          stackDepth={stackDepth}
          onStackDepthChange={setStackDepth}
        />
      </section>

      {/* 训练细节 / 关键代码 / 影响后续 */}
      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeHighlight, rehypeKatex]}
          >
            {prose.trainingDetails}
          </ReactMarkdown>
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeHighlight, rehypeKatex]}
          >
            {prose.keyCode}
          </ReactMarkdown>
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeKatex]}
          >
            {prose.aftermath}
          </ReactMarkdown>
        </div>
      </section>
    </div>
  );
}
```

注意：默认 export（不是 named）—— 因为金标本注册表用 `React.lazy(() => import(...))` 需要 default export。

- [ ] **Step 3: 编译**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npx tsc --noEmit 2>&1 | tail -10
```

期望：无错。如果 `.md?raw` 类型报 "Cannot find module"，在 `web/src/vite-env.d.ts` 加（如果还没声明）：

```typescript
declare module "*.md?raw" {
  const content: string;
  export default content;
}
```

- [ ] **Step 4: 提交**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add web/src/components/node/golden/resnet/NodePageResNet.tsx \
        web/src/components/node/golden/resnet/NodePageResNet.module.css \
        web/src/vite-env.d.ts
git commit -m "feat(web/resnet): NodePageResNet main component with 3 stages + footer"
```

---

### Task 7: 金标本注册表 + NodePage 路由分流

**Files:**
- Create: `web/src/components/node/golden/index.ts`
- Modify: `web/src/components/node/NodePage.tsx`

#### Brief

NodePage 检测当前路由是否在金标本注册表中。若是，lazy load 对应组件；若否，走原 W3 markdown 渲染。

#### Steps

- [ ] **Step 1: 创建 `golden/index.ts`**

```typescript
import { lazy } from "react";
import type { ComponentType, LazyExoticComponent } from "react";

export const goldenSamples: Record<
  string,
  LazyExoticComponent<ComponentType>
> = {
  "01-cnn/05-resnet": lazy(() => import("./resnet/NodePageResNet")),
  // 未来新增其他金标本：
  // "05-transformer/01-transformer": lazy(() => import("./transformer/NodePageTransformer")),
};
```

- [ ] **Step 2: 修改 `NodePage.tsx` 加路由分流**

读取现有 `web/src/components/node/NodePage.tsx`。**在顶部 imports 区**追加：

```typescript
import { Suspense } from "react";
import { goldenSamples } from "./golden";
```

**在 `function NodePage()` 体内最前面、`const family = ...` 之前**插入：

```typescript
  const goldenKey = `${familyId}/${nodeSlug}`;
  const GoldenComponent = goldenSamples[goldenKey];

  if (GoldenComponent) {
    return (
      <Suspense
        fallback={
          <div style={{ padding: "var(--space-16)", textAlign: "center" }}>
            Loading golden sample…
          </div>
        }
      >
        <GoldenComponent />
      </Suspense>
    );
  }
```

放在 useParams 拿到 familyId / nodeSlug 之后即可。其余原逻辑（W3 markdown 渲染）保留作为 fallback。

- [ ] **Step 3: 编译**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npx tsc --noEmit 2>&1 | tail -10
```

期望：无错。

- [ ] **Step 4: 提交**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add web/src/components/node/golden/index.ts web/src/components/node/NodePage.tsx
git commit -m "feat(web): wire NodePage to golden sample registry (ResNet → distill page)"
```

---

### Task 8: 测试

**Files:**
- Create: `web/src/components/node/golden/resnet/NodePageResNet.test.tsx`

#### Brief

简单测试：组件 mount 不崩，三个 stage 都渲染，控件响应。不测 scrollytelling 视觉。

#### Steps

- [ ] **Step 1: 创建测试**

```typescript
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import NodePageResNet from "./NodePageResNet";

function renderPage() {
  return render(
    <MemoryRouter>
      <NodePageResNet />
    </MemoryRouter>
  );
}

describe("NodePageResNet", () => {
  it("renders hero with title and key idea", () => {
    renderPage();
    expect(screen.getByText("ResNet (2015)")).toBeInTheDocument();
    expect(
      screen.getByText(/shortcut.*只学残差修正/)
    ).toBeInTheDocument();
  });

  it("renders all 3 stage h2 headings", () => {
    renderPage();
    expect(screen.getByText("之前卡在哪")).toBeInTheDocument();
    expect(screen.getByText("残差的直觉")).toBeInTheDocument();
    expect(screen.getByText("梯度高速公路")).toBeInTheDocument();
  });

  it("DepthSlider toggles depth and updates legend", () => {
    renderPage();
    // 默认 56 层，应该看到 "plain (56 层)"
    expect(screen.getByText(/plain \(56 层\)/)).toBeInTheDocument();
    // 点击 152 层按钮
    fireEvent.click(screen.getByRole("button", { name: "152" }));
    expect(screen.getByText(/plain \(152 层\)/)).toBeInTheDocument();
  });

  it("BlockTypeToggle changes block params display", () => {
    renderPage();
    // 默认 bottleneck，显示 70,144
    expect(screen.getByText("70,144")).toBeInTheDocument();
    // 点击 BasicBlock
    fireEvent.click(screen.getByRole("button", { name: "BasicBlock" }));
    expect(screen.getByText("73,728")).toBeInTheDocument();
  });

  it("StackDepthSlider changes stack count display", () => {
    renderPage();
    // 默认 6 块
    expect(screen.getByText(/6 块串联/)).toBeInTheDocument();
    // 切到 20 块
    fireEvent.click(screen.getByRole("button", { name: "20" }));
    expect(screen.getByText(/20 块串联/)).toBeInTheDocument();
  });

  it("renders footer sections (training / code / aftermath)", () => {
    renderPage();
    expect(screen.getByText("训练细节")).toBeInTheDocument();
    expect(screen.getByText("关键代码")).toBeInTheDocument();
    expect(screen.getByText(/影响.*后续/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: 跑测试**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npm test 2>&1 | tail -10
```

期望：原 14 测试 + prose 9 测试 + 6 新测试 = 29 测试全过。

如果某些测试因 jsdom 找 SVG text 文本失败，把 `getByText` 改用 `findByText` 异步或调整文本断言（用 partial regex）。

- [ ] **Step 3: 提交**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM
git add web/src/components/node/golden/resnet/NodePageResNet.test.tsx
git commit -m "test(web/resnet): add 6 tests covering hero, stages, and 3 control interactions"
```

---

### Task 9: 端到端冒烟（spec §12 验收）

**Files:** 无新增

#### Steps

- [ ] **Step 1: TS 编译**

```bash
cd /Users/lauzanhing/Desktop/Daily-LLM/web
npx tsc --noEmit && echo "TS OK"
```

- [ ] **Step 2: Build**

```bash
npm run build 2>&1 | tail -10
```

期望：成功。

```bash
gzip -c dist/assets/index-*.js | wc -c
```

期望：主 bundle gzip 仍 < 500 KB（ResNet chunk lazy load 不进主 bundle）。

```bash
ls dist/assets/ | grep -i resnet | head -5
```

期望：能看到 ResNet 相关的 chunk（lazy 输出独立文件）。

- [ ] **Step 3: 测试**

```bash
npm test 2>&1 | tail -5
```

期望：所有测试 PASS（约 29 个）。

- [ ] **Step 4: dev server 烟测**

```bash
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort &
sleep 4
curl -s http://127.0.0.1:5173/families/01-cnn/05-resnet | grep -c "<div id=\"root\">"
pkill -f "vite" 2>/dev/null || true
```

期望：grep 返回 `1`（HTML 含 root div，SPA 路由前服务端只提供 shell）。

- [ ] **Step 5: 工作树检查**

```bash
git status -s
```

期望：除 `.claude/` 外无未追踪。

---

## 完成判定（对齐 spec §12）

| # | 验收 | 落实任务 |
|---|------|--------|
| 1 | 访问 ResNet 路由加载 `NodePageResNet` | Task 7 + Task 9 |
| 2 | 访问其他 CNN 节点仍走 W3 markdown | Task 7（fallback 路径未动）|
| 3 | 3 sticky 阶段 + 滚动切换 | Task 5 + Task 6 |
| 4 | DepthSlider → 退化曲线响应 | Task 3 + Task 4 + Task 8 |
| 5 | BlockTypeToggle → BottleneckSVG 响应 + 参数量更新 | Task 3 + Task 4 + Task 8 |
| 6 | StackDepthSlider → 梯度公路响应 | Task 3 + Task 4 + Task 8 |
| 7 | Hero / 训练细节 / 关键代码 / 影响后续 4 段非 sticky | Task 6 |
| 8 | `extractProse` 切 7 个段 | Task 1 |
| 9 | TS 编译零错 | Task 9 |
| 10 | Build 成功 + 主 bundle < 500 KB gzip + ResNet 独立 chunk | Task 9 |
| 11 | 测试全过（~29 个） | Task 8 + Task 9 |
| 12 | 浏览器手动验证 | 用户负责（dev server 起来后人工） |

---

## 后续 plan（不在本计划范围内）

- **W4.5**：MVP 之外的完整套餐 5 个交互（hover loss / 论文数据点 / 逐层 shape hover / F(x) toggle / shortcut on-off）
- **W4.9**：顶配套餐 2 个高级交互（梯度反传动画 / 数学浮窗）
- **W5+**：其他金标本节点（Transformer / CLIP / o1-R1 等）
- **markdown ↔ React 自动同步检测**
- **暗色模式 / 移动端深度响应式 / a11y 审计**
