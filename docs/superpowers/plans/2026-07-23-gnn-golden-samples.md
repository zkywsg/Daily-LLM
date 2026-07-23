# GNN 家族金标本交互页 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为第 17 个家族(`17-graph-neural-networks`)的 5 个节点(GCN/GraphSAGE/GAT/GIN/Graphormer)各自补一个交互金标本页,复用现有 `web/src/components/node/golden/{slug}/` 结构约定。

**Architecture:** 每节点 `lib/data.ts`(确定性 demo 数据/函数)+ `lib/prose.ts`(从 markdown 提取正文段落)+ 3 个 `stages/*.tsx`(对应机制一/二/三,每个内含 2 个 `widgets/*.tsx`)+ `NodePage{Name}.tsx`(hero+3 stage+footer)+ `NodePage{Name}.module.css`,注册进 `web/src/components/node/golden/index.ts`。

**Tech Stack:** React + TypeScript,内联 SVG(参照 `golden/mixtral`、`golden/lstm` 的既有写法),CSS Modules,Vitest + Testing Library。

---

## 关键背景(所有任务共用,不要重新调研)

**GNN 家族的 markdown 结构与其他家族不同**:`17-graph-neural-networks/*.md` 里"核心思想 + 直觉"、"机制一"、"机制二"、"机制三"、"三件套协同" 全部是**扁平的 `##` 二级标题**,不像 mixtral 等节点那样把"直觉/机制一/机制二/机制三/三件套协同"嵌套在"## 核心思想"下面的 `###` 三级标题里。因此**不能照抄** `mixtral/lib/prose.ts` 的 `H3_KEYS`/`H2_KEYS` 双层正则逻辑——GNN 节点的 `extractProse()` 只需要一层 `##` 匹配即可(见下方每个任务里给出的精确实现)。

5 个节点的真实标题(已用 `grep` 逐一核对,后续任务直接照抄这些正则,不要自己猜):

| 节点 | md 路径 | 机制一标题 | 机制二标题 | 机制三标题 |
|---|---|---|---|---|
| GCN | `17-graph-neural-networks/01-gcn.md` | `## 机制一:重整化技巧(Ã = A + I)` | `## 机制二:对称归一化(D̃^(-1/2) Ã D̃^(-1/2))` | `## 机制三:逐层传播规则` |
| GraphSAGE | `17-graph-neural-networks/02-graphsage.md` | `## 机制一:固定大小邻域采样` | `## 机制二:可学习聚合函数(mean / LSTM / pooling)` | `## 机制三:逐层采样-聚合-拼接(K 层堆叠)` |
| GAT | `17-graph-neural-networks/03-gat.md` | `## 机制一:自注意力系数计算` | `## 机制二:softmax 归一化 + 加权聚合` | `## 机制三:多头注意力` |
| GIN | `17-graph-neural-networks/04-gin.md` | `## 机制一:为什么 mean / max 聚合不是单射的` | `## 机制二:GIN 的求和聚合 + MLP` | `## 机制三:图级别读出函数(多层拼接而非只用最后一层)` |
| Graphormer | `17-graph-neural-networks/05-graphormer.md` | `## 机制一:中心性编码(Centrality Encoding)` | `## 机制二:空间编码(Spatial Encoding)——核心创新` | `## 机制三:边编码(Edge Encoding)` |

所有 5 个节点共有的其余标题(拼写完全一致,可直接复用正则):`## 前作进展`、`## 核心思想 + 直觉`、`## 三件套协同`、`## 关键代码`、`## 性能数据`、`## 影响 / 后续`。

**通用 `extractProse()` 模板**(每个节点的 `lib/prose.ts` 都用这个模板,只改 `H2_KEYS` 里机制一/二/三 的正则前缀和 `_SOURCE_PATH` 常量):

```typescript
export interface ProseSections {
  previousWork: string;
  intuition: string;
  mechanism1: string;
  mechanism2: string;
  mechanism3: string;
  synergy: string;
  keyCode: string;
  performance: string;
  aftermath: string;
}

const H2_KEYS: Array<{ test: RegExp; key: keyof ProseSections }> = [
  { test: /^前作进展/, key: "previousWork" },
  { test: /^核心思想/, key: "intuition" },
  { test: /^机制一/, key: "mechanism1" },
  { test: /^机制二/, key: "mechanism2" },
  { test: /^机制三/, key: "mechanism3" },
  { test: /^三件套协同/, key: "synergy" },
  { test: /^关键代码/, key: "keyCode" },
  { test: /^性能数据/, key: "performance" },
  { test: /^影响/, key: "aftermath" },
];

export function extractProse(markdown: string): ProseSections {
  const body = markdown
    .replace(/^---[\s\S]*?---\n?/, "")
    .replace(/```mermaid\n[\s\S]*?```\n(\*图 ?\d[^\n]*\*\n?)?/g, "");

  const sections: ProseSections = {
    previousWork: "", intuition: "", mechanism1: "", mechanism2: "",
    mechanism3: "", synergy: "", keyCode: "", performance: "", aftermath: "",
  };

  let currentKey: keyof ProseSections | null = null;
  let buffer: string[] = [];
  const flush = () => {
    if (currentKey) sections[currentKey] = buffer.join("\n").trim();
    buffer = [];
  };

  for (const line of body.split("\n")) {
    const h2 = /^## +(.+?)\s*$/.exec(line);
    if (h2) {
      flush();
      const m = H2_KEYS.find((x) => x.test.test(h2[1].trim()));
      currentKey = m ? m.key : null;
      continue;
    }
    if (currentKey) buffer.push(line);
  }
  flush();
  return sections;
}
```

**路由注册 key 格式**:`web/src/components/node/golden/index.ts` 的 `goldenSamples` map 里,key 必须精确等于 `"17-graph-neural-networks/{NN-slug}"`(如 `"17-graph-neural-networks/01-gcn"`),否则该节点会静默走回普通 markdown 渲染(`NodePage.tsx` 用 `` `${familyId}/${nodeSlug}` `` 在 map 里查找,查不到就不是 golden)。

**标题渐变色**:每个 `NodePage{Name}.module.css` 的 `.title` 渐变统一用 `var(--family-17)` 到 `var(--accent-link)`(参照 mixtral 用 `var(--family-13)` 的写法,把 13 换成 17)。

**通用 `NodePage{Name}.module.css` 模板**(5 个节点完全复制这份,不用改任何值):

```css
.container {
  padding: 0;
  background: var(--bg-canvas);
}

.hero {
  max-width: 1000px;
  margin: 0 auto;
  padding: var(--space-8) var(--space-4);
  text-align: center;
}

@media (min-width: 768px) {
  .hero {
    padding: var(--space-12) var(--space-8);
  }
}

.back {
  display: inline-block;
  margin-bottom: var(--space-6);
  font-size: var(--fs-sm);
  color: var(--ink-secondary);
}

.title {
  font-size: clamp(2rem, 5vw + 1rem, 4rem);
  margin: var(--space-4) 0;
  background: linear-gradient(90deg, var(--family-17) 0%, var(--accent-link) 100%);
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
  padding: var(--space-12) var(--space-4);
  min-height: 100vh;
}

@media (min-width: 768px) {
  .stage {
    padding: var(--space-24) var(--space-8);
  }
}

.stageAlt {
  background: var(--bg-surface);
}

.footer {
  max-width: 800px;
  margin: 0 auto;
  padding: var(--space-12) var(--space-4);
  font-family: var(--font-serif);
  font-size: var(--fs-md);
  line-height: 1.7;
}

@media (min-width: 768px) {
  .footer {
    padding: var(--space-16) var(--space-8);
  }
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

**通用 `stages/Stage.module.css` 模板**(5 个节点完全复制这份):

```css
.grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: var(--space-6);
  align-items: start;
}

.stickyPanel {
  position: static;
  width: 100%;
}

@media (min-width: 1024px) {
  .grid {
    grid-template-columns: 1fr 1fr;
    gap: var(--space-8);
  }

  .stickyPanel {
    position: sticky;
    top: var(--space-8);
  }
}

.caption {
  font-size: var(--fs-sm);
  color: var(--ink-muted);
  margin-top: var(--space-2);
  line-height: 1.5;
}
```

**通用 toy 图**(GCN / GAT / GIN 三个节点共用同一张 6 节点小图的坐标和边,方便 viewer 认出"这是同一张图在演示不同机制"——**注意:这不是共享文件,是同样的常量数据在 3 个节点各自的 `lib/data.ts` 里独立复制一份**,因为既有约定是每个节点的 lib 自成一体、不跨节点 import):

```typescript
export const NODES = [0, 1, 2, 3, 4, 5];
export const EDGES: Array<{ a: number; b: number }> = [
  { a: 0, b: 1 }, { a: 0, b: 2 }, { a: 1, b: 2 },
  { a: 1, b: 3 }, { a: 3, b: 4 }, { a: 3, b: 5 },
];
export const POSITIONS: Record<number, [number, number]> = {
  0: [120, 200], 1: [260, 110], 2: [260, 290], 3: [420, 200], 4: [560, 110], 5: [560, 290],
};
export function rawNeighbors(node: number): number[] {
  return EDGES.filter((e) => e.a === node || e.b === node).map((e) => (e.a === node ? e.b : e.a));
}
```

---

## Task 1: GCN — lib + Stage1(自环)

**Files:**
- Create: `web/src/components/node/golden/gcn/lib/data.ts`
- Create: `web/src/components/node/golden/gcn/lib/prose.ts`
- Create: `web/src/components/node/golden/gcn/stages/Stage.module.css`
- Create: `web/src/components/node/golden/gcn/stages/SelfLoopStage.tsx`
- Create: `web/src/components/node/golden/gcn/widgets/GraphSelfLoopWidget.tsx`
- Create: `web/src/components/node/golden/gcn/widgets/AdjacencyMatrixWidget.tsx`
- Test: `web/src/components/node/golden/ProseCompleteness.test.tsx`(已存在,自动 glob 扫描,无需修改)

- [ ] **Step 1: 写 `lib/data.ts`**

```typescript
// GCN demo 数据:6 节点 toy 图 + 自环/归一化相关的纯函数。
// 不真跑训练,所有数值都是确定性计算。

export const NODES = [0, 1, 2, 3, 4, 5];
export const EDGES: Array<{ a: number; b: number }> = [
  { a: 0, b: 1 }, { a: 0, b: 2 }, { a: 1, b: 2 },
  { a: 1, b: 3 }, { a: 3, b: 4 }, { a: 3, b: 5 },
];
export const POSITIONS: Record<number, [number, number]> = {
  0: [120, 200], 1: [260, 110], 2: [260, 290], 3: [420, 200], 4: [560, 110], 5: [560, 290],
};

export function rawNeighbors(node: number): number[] {
  return EDGES.filter((e) => e.a === node || e.b === node).map((e) => (e.a === node ? e.b : e.a));
}

/** 度数,可选是否计入自环(Ã = A + I 会让每个节点度数 +1) */
export function degree(node: number, withSelfLoop: boolean): number {
  return rawNeighbors(node).length + (withSelfLoop ? 1 : 0);
}

/** 邻接矩阵一格的值(1 = 有边,withSelfLoop 时对角线也是 1) */
export function adjacencyCell(i: number, j: number, withSelfLoop: boolean): number {
  if (i === j) return withSelfLoop ? 1 : 0;
  return rawNeighbors(i).includes(j) ? 1 : 0;
}

/** 未归一化的求和聚合权重:边存在就是 1 */
export function rawWeight(i: number, j: number): number {
  return rawNeighbors(i).includes(j) ? 1 : 0;
}

/** D̃^(-1/2) Ã D̃^(-1/2) 对称归一化权重 */
export function normWeight(i: number, j: number, withSelfLoop: boolean): number {
  if (adjacencyCell(i, j, withSelfLoop) === 0) return 0;
  const di = degree(i, withSelfLoop);
  const dj = degree(j, withSelfLoop);
  return 1 / Math.sqrt(di * dj);
}

/** k-hop 内可达的节点集合(含自身),用于展示感受野随层数扩大 */
export function reachableWithinHops(center: number, hops: number): Set<number> {
  let frontier = new Set([center]);
  const visited = new Set([center]);
  for (let h = 0; h < hops; h++) {
    const next = new Set<number>();
    for (const n of frontier) {
      for (const nb of rawNeighbors(n)) {
        if (!visited.has(nb)) { next.add(nb); visited.add(nb); }
      }
    }
    frontier = next;
  }
  return visited;
}
```

- [ ] **Step 2: 写 `lib/prose.ts`**

用本文档"关键背景"里给出的通用模板,`H2_KEYS` 里机制一/二/三分别用 `/^机制一/`、`/^机制二/`、`/^机制三/`(GCN 三个标题都直接以"机制一"/"机制二"/"机制三"开头,无需更精细的正则),并加:

```typescript
export const GCN_SOURCE_PATH = "17-graph-neural-networks/01-gcn.md";
```

- [ ] **Step 3: 写 `stages/Stage.module.css`**

原样复制本文档"关键背景"里的 `Stage.module.css` 模板内容。

- [ ] **Step 4: 写 `widgets/GraphSelfLoopWidget.tsx`**

```tsx
import { EDGES, NODES, POSITIONS, degree } from "../lib/data";

interface Props {
  withSelfLoop: boolean;
}

const W = 680;
const H = 360;

// 6 节点 toy 图,切换 "原始邻接 A" vs "加自环 Ã = A + I"。
// 打开自环时每个节点旁边画一个小圆环,并在节点内显示更新后的度数。

export function GraphSelfLoopWidget({ withSelfLoop }: Props) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`GCN toy 图,自环${withSelfLoop ? "已开启" : "未开启"}`}>
      <text x={W / 2} y={24} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        {withSelfLoop ? "Ã = A + I(每个节点加一条指向自己的边)" : "原始邻接矩阵 A(只有真实的边)"}
      </text>

      {EDGES.map((e, idx) => {
        const [x1, y1] = POSITIONS[e.a];
        const [x2, y2] = POSITIONS[e.b];
        return <line key={idx} x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--border)" strokeWidth={2} />;
      })}

      {NODES.map((n) => {
        const [x, y] = POSITIONS[n];
        return (
          <g key={n}>
            {withSelfLoop && (
              <circle cx={x + 26} cy={y - 26} r={14} fill="none" stroke="#ec4899" strokeWidth={2} strokeDasharray="3 2" />
            )}
            <circle cx={x} cy={y} r={22} fill="var(--bg-surface)" stroke="#ec4899" strokeWidth={2} />
            <text x={x} y={y + 5} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
              {n}
            </text>
            <text x={x} y={y + 40} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">
              deg={degree(n, withSelfLoop)}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        {withSelfLoop
          ? "每个节点度数都 +1 —— 后面聚合时,节点会把自己的旧特征也算进新特征里"
          : "此时聚合只用邻居信息,节点自身的旧特征在下一层会被完全覆盖"}
      </text>
    </svg>
  );
}
```

- [ ] **Step 5: 写 `widgets/AdjacencyMatrixWidget.tsx`**

```tsx
import { NODES, adjacencyCell } from "../lib/data";

interface Props {
  withSelfLoop: boolean;
}

// 6x6 邻接矩阵网格,加自环时对角线格子会从 0 变成 1(高亮)。

export function AdjacencyMatrixWidget({ withSelfLoop }: Props) {
  return (
    <div style={{ display: "inline-block" }}>
      <table style={{ borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={{ width: 28 }} />
            {NODES.map((j) => (
              <th key={j} style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", width: 32 }}>
                {j}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {NODES.map((i) => (
            <tr key={i}>
              <td style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", textAlign: "right", paddingRight: 6 }}>
                {i}
              </td>
              {NODES.map((j) => {
                const v = adjacencyCell(i, j, withSelfLoop);
                const isDiag = i === j;
                return (
                  <td
                    key={j}
                    style={{
                      width: 32,
                      height: 32,
                      textAlign: "center",
                      border: "1px solid var(--border)",
                      background: v ? (isDiag ? "#ec4899" : "var(--bg-subtle)") : "var(--bg-surface)",
                      color: v && isDiag ? "#fff" : "var(--ink-primary)",
                      fontWeight: isDiag ? 700 : 400,
                      fontSize: "var(--fs-sm)",
                    }}
                  >
                    {v}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 6: 写 `stages/SelfLoopStage.tsx`**

```tsx
import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GCN_SOURCE_PATH } from "../lib/prose";
import { GraphSelfLoopWidget } from "../widgets/GraphSelfLoopWidget";
import { AdjacencyMatrixWidget } from "../widgets/AdjacencyMatrixWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function SelfLoopStage({ intuitionProse, mechanism1Prose }: Props) {
  const [withSelfLoop, setWithSelfLoop] = useState(false);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:重整化技巧 — Ã = A + I
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        给邻接矩阵加上单位矩阵,让每个节点在聚合时也把自己的特征算进去 —— 否则每一层传播都会把节点自己的信息完全丢掉,只剩邻居信息。
      </p>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={GCN_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={GCN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <button
            type="button"
            onClick={() => setWithSelfLoop((v) => !v)}
            style={{
              padding: "6px 16px", borderRadius: "var(--radius-sm)",
              border: `1px solid ${withSelfLoop ? "#ec4899" : "var(--border)"}`,
              background: withSelfLoop ? "#ec4899" : "var(--bg-surface)",
              color: withSelfLoop ? "#fff" : "var(--ink-secondary)",
              cursor: "pointer", marginBottom: "var(--space-4)",
            }}
          >
            {withSelfLoop ? "✓ 已加自环 Ã = A + I" : "点击加自环"}
          </button>
          <GraphSelfLoopWidget withSelfLoop={withSelfLoop} />
          <p className={styles.caption}>↑ 图结构与每个节点度数的变化</p>
          <div style={{ marginTop: "var(--space-6)" }}>
            <AdjacencyMatrixWidget withSelfLoop={withSelfLoop} />
          </div>
          <p className={styles.caption}>↑ 6×6 邻接矩阵,对角线(粉色)代表自环</p>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: 跑 vitest 确认这几个新文件不报错(此时还没注册进 index.ts,先跑 typecheck)**

Run: `cd web && npx tsc --noEmit`
Expected: 无新增错误(GCN 相关文件目前未被任何地方 import,不会报错;若报错说明上面代码有拼写问题,照抄修正)

- [ ] **Step 8: Commit**

```bash
git add web/src/components/node/golden/gcn/lib web/src/components/node/golden/gcn/stages web/src/components/node/golden/gcn/widgets
git commit -m "feat: GCN 金标本 lib + Stage1(自环)"
```

---

## Task 2: GCN — Stage2(归一化)+ Stage3(逐层传播)+ NodePage + 注册

**Files:**
- Create: `web/src/components/node/golden/gcn/widgets/NormalizationCompareWidget.tsx`
- Create: `web/src/components/node/golden/gcn/widgets/DegreeSelectorWidget.tsx`
- Create: `web/src/components/node/golden/gcn/stages/NormalizationStage.tsx`
- Create: `web/src/components/node/golden/gcn/widgets/ReceptiveFieldWidget.tsx`
- Create: `web/src/components/node/golden/gcn/widgets/LayerToggleWidget.tsx`
- Create: `web/src/components/node/golden/gcn/stages/PropagationStage.tsx`
- Create: `web/src/components/node/golden/gcn/NodePageGCN.tsx`
- Create: `web/src/components/node/golden/gcn/NodePageGCN.module.css`
- Modify: `web/src/components/node/golden/index.ts`
- Test: `web/src/components/node/golden/AllGoldenSamples.smoke.test.tsx`(已存在,自动扫描,无需修改)

- [ ] **Step 1: 写 `widgets/DegreeSelectorWidget.tsx`**

```tsx
import { NODES } from "../lib/data";

interface Props {
  selected: number;
  onSelect: (n: number) => void;
}

export function DegreeSelectorWidget({ selected, onSelect }: Props) {
  return (
    <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-3)" }}>
      {NODES.map((n) => (
        <button
          key={n}
          type="button"
          onClick={() => onSelect(n)}
          style={{
            width: 32, height: 32, borderRadius: "var(--radius-sm)",
            border: `1px solid ${n === selected ? "#ec4899" : "var(--border)"}`,
            background: n === selected ? "#ec4899" : "var(--bg-surface)",
            color: n === selected ? "#fff" : "var(--ink-secondary)",
            cursor: "pointer", fontSize: "var(--fs-sm)",
          }}
        >
          {n}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: 写 `widgets/NormalizationCompareWidget.tsx`**

```tsx
import { rawNeighbors, rawWeight, normWeight, degree } from "../lib/data";

interface Props {
  center: number;
}

const W = 680;
const H = 320;

// 选中一个中心节点,对比它每个邻居在 "未归一化求和(权重恒为 1)"
// vs "对称归一化 D̃^(-1/2)ÃD̃^(-1/2)" 下的聚合权重差异 —— 度数越高的
// 邻居,归一化权重被压得越低,避免它在聚合里占主导。

export function NormalizationCompareWidget({ center }: Props) {
  const neighbors = rawNeighbors(center);
  const PAD = { left: 50, right: 20, top: 60, bottom: 60 };
  const innerW = W - PAD.left - PAD.right;
  const groupW = innerW / Math.max(neighbors.length, 1);
  const barW = groupW * 0.32;
  const maxH = H - PAD.top - PAD.bottom;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`节点 ${center} 的邻居聚合权重对比`}>
      <text x={W / 2} y={22} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        节点 {center} 的邻居聚合权重:原始求和 vs 对称归一化
      </text>
      <text x={W / 2} y={38} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        deg({center}) = {degree(center, true)}(含自环)
      </text>

      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />

      {neighbors.map((nb, idx) => {
        const raw = rawWeight(center, nb);
        const norm = normWeight(center, nb, true);
        const gx = PAD.left + idx * groupW + groupW / 2;
        const rawH = raw * maxH * 0.8;
        const normH = norm * maxH * 3; // norm 值较小,放大方便比较视觉高度
        return (
          <g key={nb}>
            <rect x={gx - barW - 2} y={H - PAD.bottom - rawH} width={barW} height={rawH} fill="#9ca3af" />
            <text x={gx - barW / 2 - 2} y={H - PAD.bottom - rawH - 6} textAnchor="middle" fontSize={10} fill="var(--ink-primary)">
              {raw.toFixed(2)}
            </text>
            <rect x={gx + 2} y={H - PAD.bottom - normH} width={barW} height={normH} fill="#ec4899" />
            <text x={gx + barW / 2 + 2} y={H - PAD.bottom - normH - 6} textAnchor="middle" fontSize={10} fill="var(--ink-primary)">
              {norm.toFixed(2)}
            </text>
            <text x={gx} y={H - PAD.bottom + 18} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--ink-secondary)">
              邻居 {nb}(deg={degree(nb, true)})
            </text>
          </g>
        );
      })}

      <g transform={`translate(${W - 170}, ${PAD.top - 30})`}>
        <rect x={0} y={0} width={12} height={12} fill="#9ca3af" />
        <text x={18} y={10} fontSize={10} fill="var(--ink-secondary)">原始求和(恒为 1)</text>
        <rect x={0} y={16} width={12} height={12} fill="#ec4899" />
        <text x={18} y={26} fontSize={10} fill="var(--ink-secondary)">对称归一化</text>
      </g>
    </svg>
  );
}
```

- [ ] **Step 3: 写 `stages/NormalizationStage.tsx`**

```tsx
import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GCN_SOURCE_PATH } from "../lib/prose";
import { DegreeSelectorWidget } from "../widgets/DegreeSelectorWidget";
import { NormalizationCompareWidget } from "../widgets/NormalizationCompareWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function NormalizationStage({ mechanism2Prose }: Props) {
  const [center, setCenter] = useState(1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:对称归一化 — D̃^(-1/2) Ã D̃^(-1/2)
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        如果只是把邻居特征简单求和,度数越高的节点会在聚合里贡献越多、数值也越容易爆炸。对称归一化按两端度数的几何平均缩放每条边的权重,压低高度数邻居的影响。
      </p>

      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={GCN_SOURCE_PATH} />
        </div>

        <div className={styles.stickyPanel}>
          <DegreeSelectorWidget selected={center} onSelect={setCenter} />
          <NormalizationCompareWidget center={center} />
          <p className={styles.caption}>
            ↑ 灰色 = 未归一化(每条边权重恒为 1);粉色 = 对称归一化后的实际权重。
            切换中心节点看不同度数组合下权重被压缩的幅度。
          </p>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: 写 `widgets/LayerToggleWidget.tsx`**

```tsx
interface Props {
  hops: number;
  onChange: (h: number) => void;
}

export function LayerToggleWidget({ hops, onChange }: Props) {
  return (
    <div style={{ display: "flex", gap: 8, marginBottom: "var(--space-3)" }}>
      {[1, 2].map((h) => (
        <button
          key={h}
          type="button"
          onClick={() => onChange(h)}
          style={{
            padding: "4px 14px", borderRadius: "var(--radius-sm)",
            border: `1px solid ${hops === h ? "#ec4899" : "var(--border)"}`,
            background: hops === h ? "#ec4899" : "var(--bg-surface)",
            color: hops === h ? "#fff" : "var(--ink-secondary)",
            cursor: "pointer", fontSize: "var(--fs-sm)",
          }}
        >
          L = {h} 层
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 5: 写 `widgets/ReceptiveFieldWidget.tsx`**

```tsx
import { EDGES, NODES, POSITIONS, reachableWithinHops } from "../lib/data";

interface Props {
  center: number;
  hops: number;
}

const W = 680;
const H = 360;

// 高亮中心节点在 L 层堆叠后感受野覆盖到的节点集合:
// L=1 只覆盖直接邻居,L=2 能扩展到邻居的邻居。

export function ReceptiveFieldWidget({ center, hops }: Props) {
  const reached = reachableWithinHops(center, hops);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`节点 ${center} 在 ${hops} 层 GCN 下的感受野`}>
      <text x={W / 2} y={24} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        节点 {center} 的感受野(L = {hops} 层)
      </text>

      {EDGES.map((e, idx) => {
        const [x1, y1] = POSITIONS[e.a];
        const [x2, y2] = POSITIONS[e.b];
        const active = reached.has(e.a) && reached.has(e.b);
        return <line key={idx} x1={x1} y1={y1} x2={x2} y2={y2} stroke={active ? "#ec4899" : "var(--border)"} strokeWidth={active ? 2.5 : 1.5} />;
      })}

      {NODES.map((n) => {
        const [x, y] = POSITIONS[n];
        const active = reached.has(n);
        const isCenter = n === center;
        return (
          <g key={n}>
            <circle cx={x} cy={y} r={isCenter ? 26 : 22} fill={active ? (isCenter ? "#ec4899" : "#fce7f3") : "var(--bg-surface)"} stroke={active ? "#ec4899" : "var(--border)"} strokeWidth={2} />
            <text x={x} y={y + 5} textAnchor="middle" fontSize={13} fontWeight={700} fill={isCenter ? "#fff" : "var(--ink-primary)"}>
              {n}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        粉色节点/边 = 这一层堆叠后节点 {center} 的特征里已经包含的信息来源
      </text>
    </svg>
  );
}
```

- [ ] **Step 6: 写 `stages/PropagationStage.tsx`**

```tsx
import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GCN_SOURCE_PATH } from "../lib/prose";
import { DegreeSelectorWidget } from "../widgets/DegreeSelectorWidget";
import { LayerToggleWidget } from "../widgets/LayerToggleWidget";
import { ReceptiveFieldWidget } from "../widgets/ReceptiveFieldWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function PropagationStage({ mechanism3Prose, synergyProse }: Props) {
  const [center, setCenter] = useState(3);
  const [hops, setHops] = useState(1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:逐层传播规则
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        单层 GCN 只能看到 1-hop 邻居。堆叠 L 层后,每个节点的特征里累积了 L-hop 内所有节点的信息 —— 感受野随层数线性扩大。
      </p>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={GCN_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={GCN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <DegreeSelectorWidget selected={center} onSelect={setCenter} />
          <LayerToggleWidget hops={hops} onChange={setHops} />
          <ReceptiveFieldWidget center={center} hops={hops} />
          <p className={styles.caption}>↑ 切换 L=1/2 看感受野从直接邻居扩展到二跳邻居</p>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: 写 `NodePageGCN.module.css`**

原样复制本文档"关键背景"里的 `NodePage{Name}.module.css` 模板。

- [ ] **Step 8: 写 `NodePageGCN.tsx`**

```tsx
import { Link } from "react-router";
import gcnMarkdown from "../../../../../../17-graph-neural-networks/01-gcn.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, GCN_SOURCE_PATH } from "./lib/prose";
import { SelfLoopStage } from "./stages/SelfLoopStage";
import { NormalizationStage } from "./stages/NormalizationStage";
import { PropagationStage } from "./stages/PropagationStage";
import styles from "./NodePageGCN.module.css";

const prose = extractProse(gcnMarkdown);

export default function NodePageGCN() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/17-graph-neural-networks" className={styles.back}>
          ← 返回图神经网络(GNN)
        </Link>
        <h1 className={styles.title}>GCN (2017)</h1>
        <div className={styles.metaLine}>作者:Thomas N. Kipf · Max Welling</div>
        <div className={styles.metaLine}>论文:Semi-Supervised Classification with Graph Convolutional Networks</div>
        <p className={styles.keyIdea}>
          把谱图卷积简化到一阶邻域聚合,一层 D̃^(-1/2) Ã D̃^(-1/2) H W 传播规则定义了"现代 GNN"这个范式的起点
        </p>
      </section>

      <section className={styles.stage}>
        <SelfLoopStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <NormalizationStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <PropagationStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={GCN_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={GCN_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={GCN_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
```

- [ ] **Step 9: 注册进 `index.ts`**

在 `web/src/components/node/golden/index.ts` 的 `goldenSamples` 对象里加一行(位置任意,建议加在文件末尾附近):

```typescript
  "17-graph-neural-networks/01-gcn": lazy(() => import("./gcn/NodePageGCN")),
```

- [ ] **Step 10: 跑测试确认注册生效**

Run: `cd web && npx vitest run AllGoldenSamples.smoke ProseCompleteness`
Expected: 全部通过,新增 `17-graph-neural-networks/01-gcn` 冒烟测试用例 PASS,`gcn/lib/prose.ts` 的 prose 完整性用例 PASS(所有字段非空)

Run: `cd web && npx tsc --noEmit`
Expected: 无错误

- [ ] **Step 11: Commit**

```bash
git add web/src/components/node/golden/gcn web/src/components/node/golden/index.ts
git commit -m "feat: GCN 金标本 Stage2/3 + NodePage + 注册"
```

---

## Task 3: GraphSAGE — lib + Stage1(邻居采样)+ Stage2(聚合器对比)

**Files:**
- Create: `web/src/components/node/golden/graphsage/lib/data.ts`
- Create: `web/src/components/node/golden/graphsage/lib/prose.ts`
- Create: `web/src/components/node/golden/graphsage/stages/Stage.module.css`
- Create: `web/src/components/node/golden/graphsage/widgets/NeighborSetWidget.tsx`
- Create: `web/src/components/node/golden/graphsage/widgets/SampleControlWidget.tsx`
- Create: `web/src/components/node/golden/graphsage/stages/SamplingStage.tsx`
- Create: `web/src/components/node/golden/graphsage/widgets/AggregatorCompareWidget.tsx`
- Create: `web/src/components/node/golden/graphsage/widgets/ShuffleButtonWidget.tsx`
- Create: `web/src/components/node/golden/graphsage/stages/AggregatorStage.tsx`

- [ ] **Step 1: 写 `lib/data.ts`**

```typescript
// GraphSAGE demo 数据:8 节点图(比 GCN 的 toy 图大,便于演示"采样子集"),
// 每个节点有一个 2 维特征向量(纯 demo 数值,不代表真实语义)。

export const NODES = [0, 1, 2, 3, 4, 5, 6, 7];
export const EDGES: Array<{ a: number; b: number }> = [
  { a: 0, b: 1 }, { a: 0, b: 2 }, { a: 0, b: 3 }, { a: 0, b: 4 },
  { a: 0, b: 5 }, { a: 0, b: 6 }, { a: 1, b: 2 }, { a: 3, b: 7 },
];
export const POSITIONS: Record<number, [number, number]> = {
  0: [340, 200], 1: [180, 80], 2: [180, 320], 3: [500, 80], 4: [500, 320],
  5: [220, 200], 6: [460, 200], 7: [640, 80],
};

export const NODE_FEATURES: Record<number, [number, number]> = {
  0: [0.5, 0.5], 1: [0.9, 0.1], 2: [0.1, 0.9], 3: [0.8, 0.8],
  4: [0.2, 0.3], 5: [0.6, 0.2], 6: [0.3, 0.7], 7: [0.95, 0.9],
};

export function fullNeighbors(node: number): number[] {
  return EDGES.filter((e) => e.a === node || e.b === node).map((e) => (e.a === node ? e.b : e.a));
}

/** 确定性"随机"采样:用简单 hash 决定选哪 k 个邻居,同一 seed 结果稳定 */
export function sampleNeighbors(node: number, k: number, seed: number): number[] {
  const all = fullNeighbors(node);
  if (all.length <= k) return all;
  const scored = all.map((n) => {
    let h = seed;
    h = (h * 131 + n * 977) >>> 0;
    return { n, score: h % 1000 };
  });
  scored.sort((a, b) => a.score - b.score);
  return scored.slice(0, k).map((s) => s.n).sort((a, b) => a - b);
}

export function meanAggregate(vectors: Array<[number, number]>): [number, number] {
  if (vectors.length === 0) return [0, 0];
  const sx = vectors.reduce((s, v) => s + v[0], 0);
  const sy = vectors.reduce((s, v) => s + v[1], 0);
  return [sx / vectors.length, sy / vectors.length];
}

export function maxPoolAggregate(vectors: Array<[number, number]>): [number, number] {
  if (vectors.length === 0) return [0, 0];
  return [Math.max(...vectors.map((v) => v[0])), Math.max(...vectors.map((v) => v[1]))];
}

/** 简化版"顺序敏感"聚合(代表 LSTM 聚合器的关键性质):
 * 越靠前的邻居权重越高,所以打乱输入顺序会改变结果 —— 这正是 mean/max 不具备的性质。 */
export function orderSensitiveAggregate(vectors: Array<[number, number]>): [number, number] {
  if (vectors.length === 0) return [0, 0];
  let wsum = 0;
  let sx = 0;
  let sy = 0;
  vectors.forEach((v, idx) => {
    const w = 1 / (idx + 1);
    sx += v[0] * w;
    sy += v[1] * w;
    wsum += w;
  });
  return [sx / wsum, sy / wsum];
}

/** 图 B:一个训练时不存在的新节点(用于演示归纳式泛化) */
export const GRAPH_B_NEW_NODE = 100;
export const GRAPH_B_NEW_NODE_FEATURE: [number, number] = [0.4, 0.6];
export const GRAPH_B_NEW_NODE_NEIGHBORS = [201, 202, 203];
export const GRAPH_B_NEIGHBOR_FEATURES: Record<number, [number, number]> = {
  201: [0.7, 0.3], 202: [0.5, 0.5], 203: [0.2, 0.8],
};
```

- [ ] **Step 2: 写 `lib/prose.ts`**

用通用模板,`GRAPHSAGE_SOURCE_PATH = "17-graph-neural-networks/02-graphsage.md"`。

- [ ] **Step 3: 写 `stages/Stage.module.css`**

复制通用模板。

- [ ] **Step 4: 写 `widgets/NeighborSetWidget.tsx`**

```tsx
import { EDGES, NODES, POSITIONS } from "../lib/data";

interface Props {
  center: number;
  sampled: number[];
}

const W = 700;
const H = 380;

// 高亮中心节点的全部邻居(灰色描边)vs 被采样到的子集(粉色实心)。

export function NeighborSetWidget({ center, sampled }: Props) {
  const sampledSet = new Set(sampled);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`节点 ${center} 的邻居采样`}>
      {EDGES.map((e, idx) => {
        const [x1, y1] = POSITIONS[e.a];
        const [x2, y2] = POSITIONS[e.b];
        const touchesCenter = e.a === center || e.b === center;
        const other = e.a === center ? e.b : e.a;
        const active = touchesCenter && sampledSet.has(other);
        return <line key={idx} x1={x1} y1={y1} x2={x2} y2={y2} stroke={active ? "#ec4899" : "var(--border)"} strokeWidth={active ? 2.5 : 1.5} />;
      })}

      {NODES.map((n) => {
        const [x, y] = POSITIONS[n];
        const isCenter = n === center;
        const isNeighbor = EDGES.some((e) => (e.a === center && e.b === n) || (e.b === center && e.a === n));
        const isSampled = sampledSet.has(n);
        const fill = isCenter ? "#ec4899" : isSampled ? "#fce7f3" : "var(--bg-surface)";
        const stroke = isCenter || isSampled ? "#ec4899" : isNeighbor ? "var(--ink-muted)" : "var(--border)";
        return (
          <g key={n}>
            <circle cx={x} cy={y} r={isCenter ? 24 : 20} fill={fill} stroke={stroke} strokeWidth={2} strokeDasharray={isNeighbor && !isSampled ? "3 2" : undefined} />
            <text x={x} y={y + 5} textAnchor="middle" fontSize={12} fontWeight={700} fill={isCenter ? "#fff" : "var(--ink-primary)"}>
              {n}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        虚线描边 = 未被采样到的邻居(本轮聚合完全不参与运算)
      </text>
    </svg>
  );
}
```

- [ ] **Step 5: 写 `widgets/SampleControlWidget.tsx`**

```tsx
import { useState } from "react";
import { NODES, fullNeighbors, sampleNeighbors } from "../lib/data";
import { NeighborSetWidget } from "./NeighborSetWidget";

// 自带状态的采样控制面板:选中心节点 + k 值 + "重新采样"按钮(换 seed)。

export function SampleControlWidget() {
  const [center, setCenter] = useState(0);
  const [k, setK] = useState(3);
  const [seed, setSeed] = useState(1);

  const all = fullNeighbors(center);
  const sampled = sampleNeighbors(center, k, seed);

  return (
    <div>
      <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-3)", flexWrap: "wrap" }}>
        {NODES.filter((n) => fullNeighbors(n).length > 0).map((n) => (
          <button
            key={n}
            type="button"
            onClick={() => setCenter(n)}
            style={{
              width: 30, height: 30, borderRadius: "var(--radius-sm)",
              border: `1px solid ${n === center ? "#ec4899" : "var(--border)"}`,
              background: n === center ? "#ec4899" : "var(--bg-surface)",
              color: n === center ? "#fff" : "var(--ink-secondary)", cursor: "pointer",
            }}
          >
            {n}
          </button>
        ))}
      </div>
      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: "var(--space-4)" }}>
        <label style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)" }}>
          采样数 k = {k}
          <input type="range" min={1} max={Math.max(all.length, 1)} value={k} onChange={(e) => setK(Number(e.target.value))} style={{ marginLeft: 8 }} />
        </label>
        <button type="button" onClick={() => setSeed((s) => s + 1)} style={{ padding: "4px 12px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: "pointer", fontSize: "var(--fs-sm)" }}>
          重新采样
        </button>
      </div>
      <NeighborSetWidget center={center} sampled={sampled} />
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-2)" }}>
        节点 {center} 共有 {all.length} 个邻居,本次采样到 {sampled.length} 个:{sampled.join(", ") || "无"}
      </p>
    </div>
  );
}
```

- [ ] **Step 6: 写 `stages/SamplingStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GRAPHSAGE_SOURCE_PATH } from "../lib/prose";
import { SampleControlWidget } from "../widgets/SampleControlWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function SamplingStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:固定大小邻域采样
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        度数很高的节点(比如社交网络里的大 V)如果每次都聚合全部邻居,计算量会随度数线性增长且不可预测。GraphSAGE 每次只随机采样固定数量 k 个邻居,把每层计算量的上界锁死。
      </p>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={GRAPHSAGE_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={GRAPHSAGE_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <SampleControlWidget />
          <p className={styles.caption}>↑ 选节点、拖 k、点"重新采样"看不同采样子集</p>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: 写 `widgets/ShuffleButtonWidget.tsx`**

```tsx
interface Props {
  onShuffle: () => void;
}

export function ShuffleButtonWidget({ onShuffle }: Props) {
  return (
    <button
      type="button"
      onClick={onShuffle}
      style={{ padding: "4px 12px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: "pointer", fontSize: "var(--fs-sm)", marginBottom: "var(--space-3)" }}
    >
      打乱邻居顺序
    </button>
  );
}
```

- [ ] **Step 8: 写 `widgets/AggregatorCompareWidget.tsx`**

```tsx
import { NODE_FEATURES, meanAggregate, maxPoolAggregate, orderSensitiveAggregate } from "../lib/data";

interface Props {
  neighborIds: number[];
}

const W = 680;
const H = 320;

// 同一组邻居特征向量,分别用 mean / max-pool / order-sensitive 三种
// 聚合器计算,画在 2D 平面上(x/y 是特征的两个维度)。
// order-sensitive 的结果会随 neighborIds 顺序变化,mean/max 不会。

export function AggregatorCompareWidget({ neighborIds }: Props) {
  const vectors = neighborIds.map((id) => NODE_FEATURES[id]);
  const mean = meanAggregate(vectors);
  const maxp = maxPoolAggregate(vectors);
  const order = orderSensitiveAggregate(vectors);

  const scale = 260;
  const originX = 60;
  const originY = H - 50;
  const toXY = (v: [number, number]) => [originX + v[0] * scale, originY - v[1] * scale];

  const points: Array<{ v: [number, number]; label: string; color: string }> = [
    ...vectors.map((v, i) => ({ v, label: `n${neighborIds[i]}`, color: "var(--ink-muted)" })),
    { v: mean, label: "mean", color: "#3b82f6" },
    { v: maxp, label: "max", color: "#10b981" },
    { v: order, label: "order-sensitive", color: "#ec4899" },
  ];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="三种聚合器输出对比">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        同一组邻居,三种聚合器的输出位置
      </text>

      <line x1={originX} y1={originY} x2={originX + scale + 20} y2={originY} stroke="var(--border)" />
      <line x1={originX} y1={originY} x2={originX} y2={originY - scale - 20} stroke="var(--border)" />

      {points.map((p, idx) => {
        const [x, y] = toXY(p.v);
        const isAgg = idx >= vectors.length;
        return (
          <g key={idx}>
            <circle cx={x} cy={y} r={isAgg ? 7 : 5} fill={p.color} opacity={isAgg ? 1 : 0.6} />
            <text x={x + 8} y={y + 4} fontSize={10} fontWeight={isAgg ? 700 : 400} fill={p.color}>
              {p.label}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        灰点 = 各邻居原始特征 · 蓝/绿/粉 = mean / max-pool / order-sensitive 聚合结果
      </text>
    </svg>
  );
}
```

- [ ] **Step 9: 写 `stages/AggregatorStage.tsx`**

```tsx
import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GRAPHSAGE_SOURCE_PATH } from "../lib/prose";
import { AggregatorCompareWidget } from "../widgets/AggregatorCompareWidget";
import { ShuffleButtonWidget } from "../widgets/ShuffleButtonWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

const BASE_NEIGHBORS = [1, 3, 6];

function shuffled(arr: number[]): number[] {
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

export function AggregatorStage({ mechanism2Prose }: Props) {
  const [order, setOrder] = useState(BASE_NEIGHBORS);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:可学习聚合函数
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        mean / max-pool 对邻居的输入顺序不敏感(图本身没有顺序),而 LSTM 聚合器本质上是顺序敏感的 —— 这是它在图任务里的一个已知局限,通常需要先随机打乱邻居顺序来缓解。
      </p>

      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={GRAPHSAGE_SOURCE_PATH} />
        </div>

        <div className={styles.stickyPanel}>
          <ShuffleButtonWidget onShuffle={() => setOrder(shuffled(order))} />
          <AggregatorCompareWidget neighborIds={order} />
          <p className={styles.caption}>
            当前邻居顺序:{order.join(" → ")}。点"打乱邻居顺序"看 order-sensitive(粉点)会跟着移动,但 mean(蓝)/max(绿)始终不变。
          </p>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 10: 跑 typecheck**

Run: `cd web && npx tsc --noEmit`
Expected: 无错误

- [ ] **Step 11: Commit**

```bash
git add web/src/components/node/golden/graphsage/lib web/src/components/node/golden/graphsage/stages web/src/components/node/golden/graphsage/widgets
git commit -m "feat: GraphSAGE 金标本 lib + Stage1/2"
```

---

## Task 4: GraphSAGE — Stage3(归纳式泛化)+ NodePage + 注册

**Files:**
- Create: `web/src/components/node/golden/graphsage/widgets/InductiveCompareWidget.tsx`
- Create: `web/src/components/node/golden/graphsage/stages/InductiveStage.tsx`
- Create: `web/src/components/node/golden/graphsage/NodePageGraphSAGE.tsx`
- Create: `web/src/components/node/golden/graphsage/NodePageGraphSAGE.module.css`
- Modify: `web/src/components/node/golden/index.ts`

- [ ] **Step 1: 写 `widgets/InductiveCompareWidget.tsx`**

```tsx
import {
  GRAPH_B_NEW_NODE,
  GRAPH_B_NEW_NODE_FEATURE,
  GRAPH_B_NEW_NODE_NEIGHBORS,
  GRAPH_B_NEIGHBOR_FEATURES,
  meanAggregate,
} from "../lib/data";

// 图 B 里的新节点(id=100)在"训练时"根本不存在。
// GraphSAGE 的聚合函数(这里用 mean 举例)不依赖任何"记住哪个节点是哪个"
// 的查表操作,纯粹是邻居特征的函数 —— 所以可以直接对这个新节点算出 embedding。
// GCN 做不到:它的传播矩阵是针对固定邻接矩阵 A 求逆/归一化的,换一张图(哪怕
// 只加一个节点)整个矩阵都要重新定义。

export function InductiveCompareWidget() {
  const neighborVectors = GRAPH_B_NEW_NODE_NEIGHBORS.map((id) => GRAPH_B_NEIGHBOR_FEATURES[id]);
  const newEmbedding = meanAggregate(neighborVectors);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
      <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
        <div style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginBottom: 6 }}>
          图 B 里训练时从未出现过的新节点
        </div>
        <div style={{ fontSize: "var(--fs-md)", fontWeight: 600 }}>
          节点 #{GRAPH_B_NEW_NODE},原始特征 = [{GRAPH_B_NEW_NODE_FEATURE.join(", ")}]
        </div>
        <div style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginTop: 4 }}>
          邻居:{GRAPH_B_NEW_NODE_NEIGHBORS.map((id) => `#${id}[${GRAPH_B_NEIGHBOR_FEATURES[id].join(",")}]`).join(" · ")}
        </div>
      </div>

      <div style={{ padding: "var(--space-4)", border: "1px solid #ec4899", borderRadius: "var(--radius-md)", background: "#fce7f3" }}>
        <div style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginBottom: 6 }}>
          直接套用同一个(训练好的)mean 聚合函数
        </div>
        <div style={{ fontSize: "var(--fs-lg)", fontWeight: 700, color: "#9d174d" }}>
          embedding(#{GRAPH_B_NEW_NODE}) = mean(邻居特征) = [{newEmbedding[0].toFixed(2)}, {newEmbedding[1].toFixed(2)}]
        </div>
      </div>

      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", lineHeight: 1.6 }}>
        这一步没有查任何"节点 #{GRAPH_B_NEW_NODE} 的专属参数"——聚合函数只认邻居的特征向量,不认节点 id。
        这正是"归纳式(inductive)"的含义:同一套函数可以直接应用到任意新图、新节点。
        GCN 的 D̃^(-1/2)ÃD̃^(-1/2) 是针对固定图算出来的一个具体矩阵,图变了矩阵就要重新算,没法直接套用到没见过的节点上——这是"直推式(transductive)"的局限。
      </p>
    </div>
  );
}
```

- [ ] **Step 2: 写 `stages/InductiveStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GRAPHSAGE_SOURCE_PATH } from "../lib/prose";
import { InductiveCompareWidget } from "../widgets/InductiveCompareWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function InductiveStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:逐层采样-聚合-拼接(归纳式泛化)
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        K 层堆叠,每层都是"采样邻居 → 聚合 → 和自身特征拼接 → 非线性变换"。因为聚合函数不绑定具体节点 id,整套流程可以直接搬到训练时从未见过的新节点/新图上。
      </p>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={GRAPHSAGE_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={GRAPHSAGE_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <InductiveCompareWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: 写 `NodePageGraphSAGE.module.css`**

复制通用 `NodePage{Name}.module.css` 模板(`.title` 渐变继续用 `var(--family-17)`)。

- [ ] **Step 4: 写 `NodePageGraphSAGE.tsx`**

```tsx
import { Link } from "react-router";
import graphsageMarkdown from "../../../../../../17-graph-neural-networks/02-graphsage.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, GRAPHSAGE_SOURCE_PATH } from "./lib/prose";
import { SamplingStage } from "./stages/SamplingStage";
import { AggregatorStage } from "./stages/AggregatorStage";
import { InductiveStage } from "./stages/InductiveStage";
import styles from "./NodePageGraphSAGE.module.css";

const prose = extractProse(graphsageMarkdown);

export default function NodePageGraphSAGE() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/17-graph-neural-networks" className={styles.back}>
          ← 返回图神经网络(GNN)
        </Link>
        <h1 className={styles.title}>GraphSAGE (2017)</h1>
        <div className={styles.metaLine}>作者:William L. Hamilton · Rex Ying · Jure Leskovec</div>
        <div className={styles.metaLine}>论文:Inductive Representation Learning on Large Graphs</div>
        <p className={styles.keyIdea}>
          SAmple + aggreGatE:固定大小邻域采样 + 可学习聚合函数,让 GNN 第一次能泛化到训练时没见过的节点/图
        </p>
      </section>

      <section className={styles.stage}>
        <SamplingStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <AggregatorStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <InductiveStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={GRAPHSAGE_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={GRAPHSAGE_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={GRAPHSAGE_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
```

- [ ] **Step 5: 注册进 `index.ts`**

```typescript
  "17-graph-neural-networks/02-graphsage": lazy(() => import("./graphsage/NodePageGraphSAGE")),
```

- [ ] **Step 6: 跑测试**

Run: `cd web && npx vitest run AllGoldenSamples.smoke ProseCompleteness && npx tsc --noEmit`
Expected: 全部通过

- [ ] **Step 7: Commit**

```bash
git add web/src/components/node/golden/graphsage web/src/components/node/golden/index.ts
git commit -m "feat: GraphSAGE 金标本 Stage3 + NodePage + 注册"
```

---

## Task 5: GAT — lib + Stage1(注意力系数)+ Stage2(mask+softmax)

**Files:**
- Create: `web/src/components/node/golden/gat/lib/data.ts`
- Create: `web/src/components/node/golden/gat/lib/prose.ts`
- Create: `web/src/components/node/golden/gat/stages/Stage.module.css`
- Create: `web/src/components/node/golden/gat/widgets/AttentionBarWidget.tsx`
- Create: `web/src/components/node/golden/gat/widgets/TemperatureSliderWidget.tsx`
- Create: `web/src/components/node/golden/gat/stages/AttentionCoeffStage.tsx`
- Create: `web/src/components/node/golden/gat/widgets/MaskSoftmaxPipelineWidget.tsx`
- Create: `web/src/components/node/golden/gat/widgets/CenterNodeSelectorWidget.tsx`
- Create: `web/src/components/node/golden/gat/stages/MaskSoftmaxStage.tsx`

- [ ] **Step 1: 写 `lib/data.ts`**

```typescript
// GAT demo 数据:复用 GCN 同款 6 节点 toy 图(方便 viewer 认出同一张图在
// 演示不同机制),额外加确定性的"注意力 logits"计算函数。

export const NODES = [0, 1, 2, 3, 4, 5];
export const EDGES: Array<{ a: number; b: number }> = [
  { a: 0, b: 1 }, { a: 0, b: 2 }, { a: 1, b: 2 },
  { a: 1, b: 3 }, { a: 3, b: 4 }, { a: 3, b: 5 },
];
export const POSITIONS: Record<number, [number, number]> = {
  0: [120, 200], 1: [260, 110], 2: [260, 290], 3: [420, 200], 4: [560, 110], 5: [560, 290],
};

export function rawNeighbors(node: number): number[] {
  return EDGES.filter((e) => e.a === node || e.b === node).map((e) => (e.a === node ? e.b : e.a));
}

/** 用简单 hash 模拟"共享前馈网络"算出的原始 attention logit(未 softmax) */
function baseLogit(i: number, j: number, headSeed: number): number {
  let h = headSeed;
  h = (h * 131 + (i + 1) * 977 + (j + 1) * 331) >>> 0;
  return ((h % 1000) / 1000) * 3 - 1; // 映射到 [-1, 2]
}

/** temperature 越大分布越尖锐(除以 temperature 再 softmax 的常见写法反过来:
 * 这里 temperature 越大表示"越敏感/越锐化",故直接乘 temperature */
export function attentionLogit(i: number, j: number, temperature: number, headSeed = 0): number {
  return baseLogit(i, j, headSeed) * temperature;
}

export function softmax(xs: number[]): number[] {
  if (xs.length === 0) return [];
  const m = Math.max(...xs);
  const exps = xs.map((x) => Math.exp(x - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / s);
}

/** 4 个头的 headSeed,保证每个头结果不同但确定性可复现 */
export const HEAD_SEEDS = [0, 17, 42, 99];
</br>
```

Test: `web/src/components/node/golden/gat/lib/data.ts` 结尾不要留 `</br>` 这种非法内容——上面代码块最后一行 `</br>` 是笔误占位符,写文件时**去掉这一行**,文件以 `export const HEAD_SEEDS = [0, 17, 42, 99];` 结束。

- [ ] **Step 2: 写 `lib/prose.ts`**

用通用模板,`GAT_SOURCE_PATH = "17-graph-neural-networks/03-gat.md"`。

- [ ] **Step 3: 写 `stages/Stage.module.css`**

复制通用模板。

- [ ] **Step 4: 写 `widgets/TemperatureSliderWidget.tsx`**

```tsx
interface Props {
  temperature: number;
  onChange: (t: number) => void;
}

export function TemperatureSliderWidget({ temperature, onChange }: Props) {
  return (
    <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
      注意力锐化程度 = {temperature.toFixed(1)}
      <input
        type="range" min={0.2} max={3} step={0.1} value={temperature}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ display: "block", width: "100%", marginTop: 6 }}
      />
    </label>
  );
}
```

- [ ] **Step 5: 写 `widgets/AttentionBarWidget.tsx`**

```tsx
import { rawNeighbors, attentionLogit, softmax } from "../lib/data";

interface Props {
  center: number;
  temperature: number;
}

const W = 680;
const H = 300;

export function AttentionBarWidget({ center, temperature }: Props) {
  const neighbors = rawNeighbors(center);
  const logits = neighbors.map((n) => attentionLogit(center, n, temperature));
  const weights = softmax(logits);

  const PAD = { left: 50, right: 20, top: 50, bottom: 50 };
  const innerW = W - PAD.left - PAD.right;
  const barW = (innerW / Math.max(neighbors.length, 1)) * 0.5;
  const gap = (innerW / Math.max(neighbors.length, 1)) * 0.5;
  const maxH = H - PAD.top - PAD.bottom;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`节点 ${center} 的注意力权重`}>
      <text x={W / 2} y={22} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        节点 {center} 对各邻居的 attention 权重(softmax 后)
      </text>

      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />

      {neighbors.map((nb, idx) => {
        const w = weights[idx];
        const x = PAD.left + idx * (barW + gap) + gap / 2;
        const h = w * maxH * 3;
        return (
          <g key={nb}>
            <rect x={x} y={H - PAD.bottom - h} width={barW} height={h} rx={3} fill="#ec4899" opacity={0.85} />
            <text x={x + barW / 2} y={H - PAD.bottom - h - 6} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
              {w.toFixed(2)}
            </text>
            <text x={x + barW / 2} y={H - PAD.bottom + 18} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">
              邻居 {nb}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        温度越高,权重分布越集中在少数邻居上(越"尖锐")
      </text>
    </svg>
  );
}
```

- [ ] **Step 6: 写 `stages/AttentionCoeffStage.tsx`**

```tsx
import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GAT_SOURCE_PATH } from "../lib/prose";
import { NODES, rawNeighbors } from "../lib/data";
import { TemperatureSliderWidget } from "../widgets/TemperatureSliderWidget";
import { AttentionBarWidget } from "../widgets/AttentionBarWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function AttentionCoeffStage({ intuitionProse, mechanism1Prose }: Props) {
  const [center, setCenter] = useState(1);
  const [temperature, setTemperature] = useState(1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:自注意力系数计算
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用一个共享的前馈网络对"中心节点 + 邻居"的拼接特征打分,得到每条边的原始 attention logit —— 权重不再是固定的度数归一化系数,而是模型自己学出来的。
      </p>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={GAT_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={GAT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-3)" }}>
            {NODES.filter((n) => rawNeighbors(n).length > 0).map((n) => (
              <button
                key={n} type="button" onClick={() => setCenter(n)}
                style={{
                  width: 30, height: 30, borderRadius: "var(--radius-sm)",
                  border: `1px solid ${n === center ? "#ec4899" : "var(--border)"}`,
                  background: n === center ? "#ec4899" : "var(--bg-surface)",
                  color: n === center ? "#fff" : "var(--ink-secondary)", cursor: "pointer",
                }}
              >
                {n}
              </button>
            ))}
          </div>
          <TemperatureSliderWidget temperature={temperature} onChange={setTemperature} />
          <AttentionBarWidget center={center} temperature={temperature} />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: 写 `widgets/CenterNodeSelectorWidget.tsx`**

```tsx
import { NODES, rawNeighbors } from "../lib/data";

interface Props {
  selected: number;
  onSelect: (n: number) => void;
}

export function CenterNodeSelectorWidget({ selected, onSelect }: Props) {
  return (
    <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-3)" }}>
      {NODES.filter((n) => rawNeighbors(n).length > 0).map((n) => (
        <button
          key={n} type="button" onClick={() => onSelect(n)}
          style={{
            width: 30, height: 30, borderRadius: "var(--radius-sm)",
            border: `1px solid ${n === selected ? "#ec4899" : "var(--border)"}`,
            background: n === selected ? "#ec4899" : "var(--bg-surface)",
            color: n === selected ? "#fff" : "var(--ink-secondary)", cursor: "pointer",
          }}
        >
          {n}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 8: 写 `widgets/MaskSoftmaxPipelineWidget.tsx`**

```tsx
import { NODES, attentionLogit, softmax, rawNeighbors } from "../lib/data";

interface Props {
  center: number;
}

// 三阶段对比:①对全部节点算出的原始 logits(含非邻居)→ ②只保留邻居的 mask
// 后 logits(非邻居直接置为 "-∞"/隐藏)→ ③softmax 归一化后的最终权重。

export function MaskSoftmaxPipelineWidget({ center }: Props) {
  const neighborSet = new Set(rawNeighbors(center));
  const rawLogits = NODES.filter((n) => n !== center).map((n) => ({ n, logit: attentionLogit(center, n, 1) }));
  const maskedNeighbors = rawLogits.filter((x) => neighborSet.has(x.n));
  const weights = softmax(maskedNeighbors.map((x) => x.logit));

  const rowStyle: React.CSSProperties = { display: "flex", gap: 8, alignItems: "center", padding: "4px 0" };
  const cellStyle = (active: boolean): React.CSSProperties => ({
    width: 60, padding: "3px 6px", borderRadius: "var(--radius-sm)", textAlign: "center", fontSize: "var(--fs-sm)",
    background: active ? "#fce7f3" : "var(--bg-subtle)", color: active ? "#9d174d" : "var(--ink-muted)",
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
      <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
        ① 全部节点的原始 logits
      </div>
      {rawLogits.map(({ n, logit }) => (
        <div key={n} style={rowStyle}>
          <span style={{ width: 50, fontSize: "var(--fs-sm)" }}>节点 {n}</span>
          <span style={cellStyle(neighborSet.has(n))}>{logit.toFixed(2)}</span>
          <span style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)" }}>{neighborSet.has(n) ? "是邻居" : "非邻居(将被屏蔽)"}</span>
        </div>
      ))}

      <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginTop: "var(--space-3)" }}>
        ② mask 后只剩邻居 → ③ softmax 归一化权重
      </div>
      {maskedNeighbors.map(({ n }, idx) => (
        <div key={n} style={rowStyle}>
          <span style={{ width: 50, fontSize: "var(--fs-sm)" }}>节点 {n}</span>
          <span style={cellStyle(true)}>{weights[idx].toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 9: 写 `stages/MaskSoftmaxStage.tsx`**

```tsx
import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GAT_SOURCE_PATH } from "../lib/prose";
import { CenterNodeSelectorWidget } from "../widgets/CenterNodeSelectorWidget";
import { MaskSoftmaxPipelineWidget } from "../widgets/MaskSoftmaxPipelineWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function MaskSoftmaxStage({ mechanism2Prose }: Props) {
  const [center, setCenter] = useState(3);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:softmax 归一化 + 加权聚合
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        GAT 不需要提前知道完整图结构去做矩阵运算 —— 但计算 attention 时仍然只在"真实存在的边"上做 softmax,非邻居的 logit 会被 mask 掉,不参与归一化。
      </p>

      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={GAT_SOURCE_PATH} />
        </div>

        <div className={styles.stickyPanel}>
          <CenterNodeSelectorWidget selected={center} onSelect={setCenter} />
          <MaskSoftmaxPipelineWidget center={center} />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 10: 跑 typecheck**

Run: `cd web && npx tsc --noEmit`
Expected: 无错误(注意 Step 1 里 `</br>` 占位符已删除)

- [ ] **Step 11: Commit**

```bash
git add web/src/components/node/golden/gat/lib web/src/components/node/golden/gat/stages web/src/components/node/golden/gat/widgets
git commit -m "feat: GAT 金标本 lib + Stage1/2"
```

---

## Task 6: GAT — Stage3(多头注意力)+ NodePage + 注册

**Files:**
- Create: `web/src/components/node/golden/gat/widgets/MultiHeadWidget.tsx`
- Create: `web/src/components/node/golden/gat/widgets/CombineModeToggleWidget.tsx`
- Create: `web/src/components/node/golden/gat/stages/MultiHeadStage.tsx`
- Create: `web/src/components/node/golden/gat/NodePageGAT.tsx`
- Create: `web/src/components/node/golden/gat/NodePageGAT.module.css`
- Modify: `web/src/components/node/golden/index.ts`

- [ ] **Step 1: 写 `widgets/CombineModeToggleWidget.tsx`**

```tsx
interface Props {
  mode: "concat" | "average";
  onChange: (m: "concat" | "average") => void;
}

export function CombineModeToggleWidget({ mode, onChange }: Props) {
  return (
    <div style={{ display: "flex", gap: 8, marginBottom: "var(--space-3)" }}>
      {(["concat", "average"] as const).map((m) => (
        <button
          key={m} type="button" onClick={() => onChange(m)}
          style={{
            padding: "4px 14px", borderRadius: "var(--radius-sm)",
            border: `1px solid ${mode === m ? "#ec4899" : "var(--border)"}`,
            background: mode === m ? "#ec4899" : "var(--bg-surface)",
            color: mode === m ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-sm)",
          }}
        >
          {m === "concat" ? "concat(中间层)" : "average(输出层)"}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: 写 `widgets/MultiHeadWidget.tsx`**

```tsx
import { rawNeighbors, attentionLogit, softmax, HEAD_SEEDS } from "../lib/data";

interface Props {
  center: number;
  mode: "concat" | "average";
}

const W = 700;
const H = 380;

// 4 个头并排展示各自的注意力权重分布(小型 bar group),
// 下方再画一条"组合后输出"的条形图:concat 模式下把 4 个头的
// 权重依次排开(输出维度变宽),average 模式下逐元素平均(维度不变)。

export function MultiHeadWidget({ center, mode }: Props) {
  const neighbors = rawNeighbors(center);
  const headWeights = HEAD_SEEDS.map((seed) => {
    const logits = neighbors.map((n) => attentionLogit(center, n, 1, seed));
    return softmax(logits);
  });

  const combined =
    mode === "average"
      ? neighbors.map((_, idx) => headWeights.reduce((s, hw) => s + hw[idx], 0) / headWeights.length)
      : headWeights.flat();

  const headColors = ["#ec4899", "#3b82f6", "#10b981", "#f59e0b"];
  const PAD = { left: 50, right: 20, top: 30, bottom: 30 };
  const rowH = 60;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`节点 ${center} 的多头注意力`}>
      {headWeights.map((weights, headIdx) => {
        const y = PAD.top + headIdx * rowH;
        const cellW = (W - PAD.left - PAD.right) / neighbors.length;
        return (
          <g key={headIdx}>
            <text x={PAD.left - 8} y={y + rowH / 2} textAnchor="end" fontSize={11} fontWeight={600} fill={headColors[headIdx]}>
              Head {headIdx}
            </text>
            {weights.map((w, idx) => {
              const h = w * (rowH - 20) * 3;
              const x = PAD.left + idx * cellW;
              return (
                <g key={idx}>
                  <rect x={x + 4} y={y + rowH - 10 - h} width={cellW - 8} height={h} fill={headColors[headIdx]} opacity={0.8} />
                  <text x={x + cellW / 2} y={y + rowH - 12 - h} textAnchor="middle" fontSize={9} fill="var(--ink-primary)">
                    {w.toFixed(2)}
                  </text>
                </g>
              );
            })}
          </g>
        );
      })}

      <text x={W / 2} y={PAD.top + 4 * rowH + 20} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
        组合输出({mode === "concat" ? "concat,维度 = 4 × 邻居数" : "average,维度 = 邻居数"})
      </text>
      {combined.map((v, idx) => {
        const cellW = (W - PAD.left - PAD.right) / combined.length;
        const x = PAD.left + idx * cellW;
        const y0 = PAD.top + 4 * rowH + 30;
        const h = v * 60 * (mode === "concat" ? 3 : 3);
        return (
          <rect key={idx} x={x + 3} y={y0 + 40 - h} width={cellW - 6} height={h} fill={mode === "concat" ? headColors[idx % 4] : "#9d174d"} opacity={0.85} />
        );
      })}
    </svg>
  );
}
```

- [ ] **Step 3: 写 `stages/MultiHeadStage.tsx`**

```tsx
import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GAT_SOURCE_PATH } from "../lib/prose";
import { CenterNodeSelectorWidget } from "../widgets/CenterNodeSelectorWidget";
import { CombineModeToggleWidget } from "../widgets/CombineModeToggleWidget";
import { MultiHeadWidget } from "../widgets/MultiHeadWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function MultiHeadStage({ mechanism3Prose, synergyProse }: Props) {
  const [center, setCenter] = useState(1);
  const [mode, setMode] = useState<"concat" | "average">("concat");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:多头注意力
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        每个头独立学一套 attention 权重,关注邻居的不同侧面。中间层用 concat 拼接保留所有头的信息、扩大表示维度;输出层用 average 平均,把多头意见汇总成一个稳定的最终预测。
      </p>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={GAT_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={GAT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <CenterNodeSelectorWidget selected={center} onSelect={setCenter} />
          <CombineModeToggleWidget mode={mode} onChange={setMode} />
          <MultiHeadWidget center={center} mode={mode} />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: 写 `NodePageGAT.module.css`**

复制通用模板。

- [ ] **Step 5: 写 `NodePageGAT.tsx`**

```tsx
import { Link } from "react-router";
import gatMarkdown from "../../../../../../17-graph-neural-networks/03-gat.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, GAT_SOURCE_PATH } from "./lib/prose";
import { AttentionCoeffStage } from "./stages/AttentionCoeffStage";
import { MaskSoftmaxStage } from "./stages/MaskSoftmaxStage";
import { MultiHeadStage } from "./stages/MultiHeadStage";
import styles from "./NodePageGAT.module.css";

const prose = extractProse(gatMarkdown);

export default function NodePageGAT() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/17-graph-neural-networks" className={styles.back}>
          ← 返回图神经网络(GNN)
        </Link>
        <h1 className={styles.title}>GAT (2018)</h1>
        <div className={styles.metaLine}>
          作者:Petar Veličković · Guillem Cucurull · Arantxa Casanova · Adriana Romero · Pietro Liò · Yoshua Bengio
        </div>
        <div className={styles.metaLine}>论文:Graph Attention Networks</div>
        <p className={styles.keyIdea}>
          用可学习的 attention 权重替代 GCN 里固定的度数归一化系数,让模型隐式学会"哪个邻居更重要"
        </p>
      </section>

      <section className={styles.stage}>
        <AttentionCoeffStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <MaskSoftmaxStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <MultiHeadStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={GAT_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={GAT_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={GAT_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
```

- [ ] **Step 6: 注册进 `index.ts`**

```typescript
  "17-graph-neural-networks/03-gat": lazy(() => import("./gat/NodePageGAT")),
```

- [ ] **Step 7: 跑测试**

Run: `cd web && npx vitest run AllGoldenSamples.smoke ProseCompleteness && npx tsc --noEmit`
Expected: 全部通过

- [ ] **Step 8: Commit**

```bash
git add web/src/components/node/golden/gat web/src/components/node/golden/index.ts
git commit -m "feat: GAT 金标本 Stage3 + NodePage + 注册"
```

---

## Task 7: GIN — lib + Stage1(非单射反例)+ Stage2(ε 可调)

**Files:**
- Create: `web/src/components/node/golden/gin/lib/data.ts`
- Create: `web/src/components/node/golden/gin/lib/prose.ts`
- Create: `web/src/components/node/golden/gin/stages/Stage.module.css`
- Create: `web/src/components/node/golden/gin/widgets/InjectivityCounterexampleWidget.tsx`
- Create: `web/src/components/node/golden/gin/widgets/MultisetDisplayWidget.tsx`
- Create: `web/src/components/node/golden/gin/stages/InjectivityStage.tsx`
- Create: `web/src/components/node/golden/gin/widgets/EpsilonSliderWidget.tsx`
- Create: `web/src/components/node/golden/gin/widgets/SelfVsNeighborBarWidget.tsx`
- Create: `web/src/components/node/golden/gin/stages/EpsilonStage.tsx`

- [ ] **Step 1: 写 `lib/data.ts`**

```typescript
// GIN demo 数据:非单射反例(mean/max 相同但 sum 不同的两个多重集)+
// ε 可调的自身/邻居加权演示 + WL 染色用的 6 节点 toy 图。

/** 经典反例:两个多重集在 mean/max 下无法区分,但 sum 下不同。
 * X = {1, 1}(两个特征为 1 的邻居),Y = {1, 1, 1, 1}(四个特征为 1 的邻居)。
 * mean(X) = mean(Y) = 1,max(X) = max(Y) = 1,但 sum(X) = 2 ≠ sum(Y) = 4。 */
export const MULTISET_X = [1, 1];
export const MULTISET_Y = [1, 1, 1, 1];

export function mean(xs: number[]): number {
  return xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length;
}
export function max(xs: number[]): number {
  return xs.length === 0 ? 0 : Math.max(...xs);
}
export function sum(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0);
}

/** GIN 更新(简化到 MLP 前的标量组合):(1+ε)·h_self + Σ neighbors */
export function ginPreMlp(selfFeature: number, neighborSum: number, epsilon: number): number {
  return (1 + epsilon) * selfFeature + neighborSum;
}

// WL 染色用的 6 节点 toy 图(与 GCN/GAT 同款,方便认出同一张图)
export const NODES = [0, 1, 2, 3, 4, 5];
export const EDGES: Array<{ a: number; b: number }> = [
  { a: 0, b: 1 }, { a: 0, b: 2 }, { a: 1, b: 2 },
  { a: 1, b: 3 }, { a: 3, b: 4 }, { a: 3, b: 5 },
];
export const POSITIONS: Record<number, [number, number]> = {
  0: [120, 200], 1: [260, 110], 2: [260, 290], 3: [420, 200], 4: [560, 110], 5: [560, 290],
};

export function rawNeighbors(node: number): number[] {
  return EDGES.filter((e) => e.a === node || e.b === node).map((e) => (e.a === node ? e.b : e.a));
}

/** 一轮 WL 颜色迭代:新颜色 = hash(自己的颜色, 排序后的邻居颜色多重集) */
export function wlRefine(colors: number[]): number[] {
  const signatures = colors.map((c, node) => {
    const neighborColors = rawNeighbors(node).map((n) => colors[n]).sort((a, b) => a - b);
    return `${c}|${neighborColors.join(",")}`;
  });
  const uniqueSigs = Array.from(new Set(signatures)).sort();
  return signatures.map((sig) => uniqueSigs.indexOf(sig));
}
```

- [ ] **Step 2: 写 `lib/prose.ts`**

用通用模板,`GIN_SOURCE_PATH = "17-graph-neural-networks/04-gin.md"`。

- [ ] **Step 3: 写 `stages/Stage.module.css`**

复制通用模板。

- [ ] **Step 4: 写 `widgets/MultisetDisplayWidget.tsx`**

```tsx
interface Props {
  label: string;
  values: number[];
  color: string;
}

export function MultisetDisplayWidget({ label, values, color }: Props) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: "var(--space-3)" }}>
      <span style={{ width: 90, fontSize: "var(--fs-sm)", fontWeight: 600 }}>{label}</span>
      <div style={{ display: "flex", gap: 4 }}>
        {values.map((v, idx) => (
          <div
            key={idx}
            style={{
              width: 28, height: 28, borderRadius: "50%", background: color, color: "#fff",
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: "var(--fs-xs)", fontWeight: 700,
            }}
          >
            {v}
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 5: 写 `widgets/InjectivityCounterexampleWidget.tsx`**

```tsx
import { MULTISET_X, MULTISET_Y, mean, max, sum } from "../lib/data";
import { MultisetDisplayWidget } from "./MultisetDisplayWidget";

// 两个不同的邻居多重集,mean/max 塌缩成相同结果,sum 保留了区别。

export function InjectivityCounterexampleWidget() {
  const rows: Array<{ name: string; fn: (xs: number[]) => number }> = [
    { name: "mean", fn: mean },
    { name: "max", fn: max },
    { name: "sum", fn: sum },
  ];

  return (
    <div>
      <MultisetDisplayWidget label="多重集 X" values={MULTISET_X} color="#3b82f6" />
      <MultisetDisplayWidget label="多重集 Y" values={MULTISET_Y} color="#f59e0b" />

      <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "var(--space-4)" }}>
        <thead>
          <tr>
            <th style={{ textAlign: "left", fontSize: "var(--fs-sm)", color: "var(--ink-muted)", padding: "4px 8px" }}>聚合函数</th>
            <th style={{ textAlign: "center", fontSize: "var(--fs-sm)", color: "var(--ink-muted)", padding: "4px 8px" }}>结果(X)</th>
            <th style={{ textAlign: "center", fontSize: "var(--fs-sm)", color: "var(--ink-muted)", padding: "4px 8px" }}>结果(Y)</th>
            <th style={{ textAlign: "center", fontSize: "var(--fs-sm)", color: "var(--ink-muted)", padding: "4px 8px" }}>能否区分</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ name, fn }) => {
            const rx = fn(MULTISET_X);
            const ry = fn(MULTISET_Y);
            const distinguishable = rx !== ry;
            return (
              <tr key={name} style={{ borderTop: "1px solid var(--border)" }}>
                <td style={{ padding: "6px 8px", fontWeight: 600 }}>{name}</td>
                <td style={{ padding: "6px 8px", textAlign: "center" }}>{rx}</td>
                <td style={{ padding: "6px 8px", textAlign: "center" }}>{ry}</td>
                <td style={{ padding: "6px 8px", textAlign: "center", color: distinguishable ? "#059669" : "#dc2626", fontWeight: 700 }}>
                  {distinguishable ? "✓ 能区分" : "✗ 塌缩成相同值"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 6: 写 `stages/InjectivityStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GIN_SOURCE_PATH } from "../lib/prose";
import { InjectivityCounterexampleWidget } from "../widgets/InjectivityCounterexampleWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function InjectivityStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:为什么 mean / max 聚合不是单射的
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        单射(injective)意味着不同的输入永远映射到不同的输出。mean/max 会把"两个邻居都是 1"和"四个邻居都是 1"这两种明显不同的情况聚合成完全相同的结果 —— 丢失了"有多少个邻居"这个信息。
      </p>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={GIN_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={GIN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <InjectivityCounterexampleWidget />
          <p className={styles.caption}>↑ sum 能区分两个多重集,mean/max 不能 —— 这是 GIN 选 sum 的原因</p>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: 写 `widgets/EpsilonSliderWidget.tsx`**

```tsx
interface Props {
  epsilon: number;
  onChange: (e: number) => void;
}

export function EpsilonSliderWidget({ epsilon, onChange }: Props) {
  return (
    <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
      ε = {epsilon.toFixed(2)}
      <input
        type="range" min={0} max={2} step={0.05} value={epsilon}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ display: "block", width: "100%", marginTop: 6 }}
      />
    </label>
  );
}
```

- [ ] **Step 8: 写 `widgets/SelfVsNeighborBarWidget.tsx`**

```tsx
import { ginPreMlp } from "../lib/data";

interface Props {
  epsilon: number;
}

const SELF_FEATURE = 1.0;
const NEIGHBOR_SUM = 2.4;

const W = 500;
const H = 260;

export function SelfVsNeighborBarWidget({ epsilon }: Props) {
  const selfContribution = (1 + epsilon) * SELF_FEATURE;
  const total = ginPreMlp(SELF_FEATURE, NEIGHBOR_SUM, epsilon);

  const PAD = { left: 60, right: 20, top: 40, bottom: 40 };
  const maxH = H - PAD.top - PAD.bottom;
  const maxVal = 6;
  const barW = 70;

  const bar = (x: number, val: number, label: string, color: string) => {
    const h = (val / maxVal) * maxH;
    return (
      <g key={label}>
        <rect x={x} y={H - PAD.bottom - h} width={barW} height={h} fill={color} rx={3} />
        <text x={x + barW / 2} y={H - PAD.bottom - h - 6} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
          {val.toFixed(2)}
        </text>
        <text x={x + barW / 2} y={H - PAD.bottom + 18} textAnchor="middle" fontSize={10} fill="var(--ink-secondary)">
          {label}
        </text>
      </g>
    );
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`ε=${epsilon} 时自身与邻居贡献对比`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        (1+ε)·h_self + Σneighbors,ε = {epsilon.toFixed(2)}
      </text>
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />
      {bar(PAD.left + 20, selfContribution, "自身贡献", "#ec4899")}
      {bar(PAD.left + 20 + barW + 30, NEIGHBOR_SUM, "邻居贡献(固定)", "#9ca3af")}
      {bar(PAD.left + 20 + 2 * (barW + 30), total, "MLP 前总和", "#3b82f6")}
    </svg>
  );
}
```

- [ ] **Step 9: 写 `stages/EpsilonStage.tsx`**

```tsx
import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GIN_SOURCE_PATH } from "../lib/prose";
import { EpsilonSliderWidget } from "../widgets/EpsilonSliderWidget";
import { SelfVsNeighborBarWidget } from "../widgets/SelfVsNeighborBarWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function EpsilonStage({ mechanism2Prose }: Props) {
  const [epsilon, setEpsilon] = useState(0);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:GIN 的求和聚合 + MLP
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        (1+ε) 控制节点在更新时对"自己旧特征"的加权。ε=0 时自身和普通邻居等权;ε 越大,节点越"固执",更新时更依赖自己而不是邻居传来的信息。
      </p>

      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={GIN_SOURCE_PATH} />
        </div>

        <div className={styles.stickyPanel}>
          <EpsilonSliderWidget epsilon={epsilon} onChange={setEpsilon} />
          <SelfVsNeighborBarWidget epsilon={epsilon} />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 10: 跑 typecheck**

Run: `cd web && npx tsc --noEmit`
Expected: 无错误

- [ ] **Step 11: Commit**

```bash
git add web/src/components/node/golden/gin/lib web/src/components/node/golden/gin/stages web/src/components/node/golden/gin/widgets
git commit -m "feat: GIN 金标本 lib + Stage1/2"
```

---

## Task 8: GIN — Stage3(WL 染色)+ NodePage + 注册

**Files:**
- Create: `web/src/components/node/golden/gin/widgets/WLColoringWidget.tsx`
- Create: `web/src/components/node/golden/gin/stages/WLStage.tsx`
- Create: `web/src/components/node/golden/gin/NodePageGIN.tsx`
- Create: `web/src/components/node/golden/gin/NodePageGIN.module.css`
- Modify: `web/src/components/node/golden/index.ts`

- [ ] **Step 1: 写 `widgets/WLColoringWidget.tsx`**

```tsx
import { useState } from "react";
import { EDGES, NODES, POSITIONS, wlRefine } from "../lib/data";

const W = 680;
const H = 360;
const PALETTE = ["#9ca3af", "#ec4899", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#dc2626"];

// 从"所有节点同色"开始,每点一次"跑一轮 WL"就迭代一次颜色精细化。
// 收敛后不同颜色数 = WL test 能区分出的等价类数量。

export function WLColoringWidget() {
  const [colors, setColors] = useState<number[]>(NODES.map(() => 0));
  const [round, setRound] = useState(0);

  const uniqueColors = new Set(colors).size;

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`WL 颜色迭代第 ${round} 轮`}>
        <text x={W / 2} y={22} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          Weisfeiler-Lehman 颜色迭代 — 第 {round} 轮,当前 {uniqueColors} 种颜色
        </text>

        {EDGES.map((e, idx) => {
          const [x1, y1] = POSITIONS[e.a];
          const [x2, y2] = POSITIONS[e.b];
          return <line key={idx} x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--border)" strokeWidth={1.5} />;
        })}

        {NODES.map((n) => {
          const [x, y] = POSITIONS[n];
          const color = PALETTE[colors[n] % PALETTE.length];
          return (
            <g key={n}>
              <circle cx={x} cy={y} r={22} fill={color} stroke="var(--bg-canvas)" strokeWidth={2} />
              <text x={x} y={y + 5} textAnchor="middle" fontSize={13} fontWeight={700} fill="#fff">
                {n}
              </text>
            </g>
          );
        })}
      </svg>

      <div style={{ display: "flex", gap: 8, marginTop: "var(--space-3)" }}>
        <button
          type="button"
          onClick={() => { setColors(wlRefine(colors)); setRound((r) => r + 1); }}
          disabled={round >= 3}
          style={{ padding: "4px 14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: round >= 3 ? "not-allowed" : "pointer", fontSize: "var(--fs-sm)", opacity: round >= 3 ? 0.5 : 1 }}
        >
          跑一轮 WL 精细化
        </button>
        <button
          type="button"
          onClick={() => { setColors(NODES.map(() => 0)); setRound(0); }}
          style={{ padding: "4px 14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: "pointer", fontSize: "var(--fs-sm)" }}
        >
          重置
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 写 `stages/WLStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GIN_SOURCE_PATH } from "../lib/prose";
import { WLColoringWidget } from "../widgets/WLColoringWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function WLStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:图级别读出函数
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        GIN 的 sum+MLP 逐层聚合在理论上等价于 Weisfeiler-Lehman 颜色精细化算法:每一轮都把"自己的颜色 + 邻居颜色多重集"映射成新颜色,收敛后能区分的节点数就是 GNN 表达力的上界。
      </p>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={GIN_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={GIN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <WLColoringWidget />
          <p className={styles.caption}>↑ 点"跑一轮 WL 精细化"看颜色如何逐步收敛出不同的等价类</p>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: 写 `NodePageGIN.module.css`**

复制通用模板。

- [ ] **Step 4: 写 `NodePageGIN.tsx`**

```tsx
import { Link } from "react-router";
import ginMarkdown from "../../../../../../17-graph-neural-networks/04-gin.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, GIN_SOURCE_PATH } from "./lib/prose";
import { InjectivityStage } from "./stages/InjectivityStage";
import { EpsilonStage } from "./stages/EpsilonStage";
import { WLStage } from "./stages/WLStage";
import styles from "./NodePageGIN.module.css";

const prose = extractProse(ginMarkdown);

export default function NodePageGIN() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/17-graph-neural-networks" className={styles.back}>
          ← 返回图神经网络(GNN)
        </Link>
        <h1 className={styles.title}>GIN (2019)</h1>
        <div className={styles.metaLine}>作者:Keyulu Xu · Weihua Hu · Jure Leskovec · Stefanie Jegelka</div>
        <div className={styles.metaLine}>论文:How Powerful are Graph Neural Networks?</div>
        <p className={styles.keyIdea}>
          用 Weisfeiler-Lehman 图同构测试给 GNN 表达力定理上界,提出 sum 聚合 + MLP 的 GIN 达到 WL test 同等的最大可能表达力
        </p>
      </section>

      <section className={styles.stage}>
        <InjectivityStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <EpsilonStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <WLStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={GIN_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={GIN_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={GIN_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
```

- [ ] **Step 5: 注册进 `index.ts`**

```typescript
  "17-graph-neural-networks/04-gin": lazy(() => import("./gin/NodePageGIN")),
```

- [ ] **Step 6: 跑测试**

Run: `cd web && npx vitest run AllGoldenSamples.smoke ProseCompleteness && npx tsc --noEmit`
Expected: 全部通过

- [ ] **Step 7: Commit**

```bash
git add web/src/components/node/golden/gin web/src/components/node/golden/index.ts
git commit -m "feat: GIN 金标本 Stage3 + NodePage + 注册"
```

---

## Task 9: Graphormer — lib + Stage1(中心性编码)+ Stage2(空间编码)

**Files:**
- Create: `web/src/components/node/golden/graphormer/lib/data.ts`
- Create: `web/src/components/node/golden/graphormer/lib/prose.ts`
- Create: `web/src/components/node/golden/graphormer/stages/Stage.module.css`
- Create: `web/src/components/node/golden/graphormer/widgets/CentralityHistogramWidget.tsx`
- Create: `web/src/components/node/golden/graphormer/stages/CentralityStage.tsx`
- Create: `web/src/components/node/golden/graphormer/widgets/NodePairSelectorWidget.tsx`
- Create: `web/src/components/node/golden/graphormer/widgets/SpatialBiasHeatmapWidget.tsx`
- Create: `web/src/components/node/golden/graphormer/stages/SpatialStage.tsx`

- [ ] **Step 1: 写 `lib/data.ts`**

```typescript
// Graphormer demo 数据:复用同款 6 节点 toy 图,加最短路径 BFS +
// 中心性/空间/边编码的确定性查表函数。

export const NODES = [0, 1, 2, 3, 4, 5];
export const EDGES: Array<{ a: number; b: number }> = [
  { a: 0, b: 1 }, { a: 0, b: 2 }, { a: 1, b: 2 },
  { a: 1, b: 3 }, { a: 3, b: 4 }, { a: 3, b: 5 },
];
export const POSITIONS: Record<number, [number, number]> = {
  0: [120, 200], 1: [260, 110], 2: [260, 290], 3: [420, 200], 4: [560, 110], 5: [560, 290],
};

export function rawNeighbors(node: number): number[] {
  return EDGES.filter((e) => e.a === node || e.b === node).map((e) => (e.a === node ? e.b : e.a));
}

export function degree(node: number): number {
  return rawNeighbors(node).length;
}

/** 中心性编码:按度数查表得到一个 embedding 标量(demo 用单维简化) */
const CENTRALITY_TABLE = [0, 0.2, 0.5, 0.9, 1.3, 1.6];
export function centralityEmbedding(node: number): number {
  return CENTRALITY_TABLE[Math.min(degree(node), CENTRALITY_TABLE.length - 1)];
}

/** BFS 最短路径距离,返回路径上经过的边列表(用于边编码) */
export function shortestPath(i: number, j: number): { distance: number; path: number[] } {
  if (i === j) return { distance: 0, path: [i] };
  const visited = new Set([i]);
  const queue: number[][] = [[i]];
  while (queue.length > 0) {
    const path = queue.shift()!;
    const last = path[path.length - 1];
    for (const nb of rawNeighbors(last)) {
      if (nb === j) return { distance: path.length, path: [...path, nb] };
      if (!visited.has(nb)) {
        visited.add(nb);
        queue.push([...path, nb]);
      }
    }
  }
  return { distance: Infinity, path: [] };
}

/** 空间编码 bias:距离越远,bias 越负(衰减邻居之外节点的注意力) */
const SPATIAL_BIAS_TABLE = [0, -0.1, -0.3, -0.6, -1.0];
export function spatialBias(distance: number): number {
  if (!Number.isFinite(distance)) return -2;
  return SPATIAL_BIAS_TABLE[Math.min(distance, SPATIAL_BIAS_TABLE.length - 1)];
}

/** 边编码:路径上每条边有一个固定的小 bias,累加到 spatial bias 之上 */
function edgeWeight(a: number, b: number): number {
  let h = (a + 1) * 131 + (b + 1) * 977;
  h = h >>> 0;
  return ((h % 100) / 100) * 0.3; // [0, 0.3)
}
export function edgeBias(path: number[]): number {
  let total = 0;
  for (let k = 0; k < path.length - 1; k++) total += edgeWeight(path[k], path[k + 1]);
  return total;
}

/** 基础 QK attention score(不含任何结构 bias),确定性 hash 模拟 */
export function baseScore(i: number, j: number): number {
  let h = (i + 1) * 313 + (j + 1) * 71;
  h = h >>> 0;
  return ((h % 1000) / 1000) * 2 - 1; // [-1, 1]
}
```

- [ ] **Step 2: 写 `lib/prose.ts`**

用通用模板,`GRAPHORMER_SOURCE_PATH = "17-graph-neural-networks/05-graphormer.md"`。

- [ ] **Step 3: 写 `stages/Stage.module.css`**

复制通用模板。

- [ ] **Step 4: 写 `widgets/CentralityHistogramWidget.tsx`**

```tsx
import { useState } from "react";
import { NODES, degree, centralityEmbedding } from "../lib/data";

const W = 680;
const H = 320;

// 6 个节点的度数直方图,点击一根柱子高亮对应节点,右侧显示查表得到的
// centrality embedding 标量值。

export function CentralityHistogramWidget() {
  const [selected, setSelected] = useState<number | null>(null);

  const PAD = { left: 50, right: 20, top: 50, bottom: 50 };
  const innerW = W - PAD.left - PAD.right;
  const barW = (innerW / NODES.length) * 0.6;
  const gap = (innerW / NODES.length) * 0.4;
  const maxH = H - PAD.top - PAD.bottom;
  const maxDeg = Math.max(...NODES.map((n) => degree(n)), 1);

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="节点度数直方图">
        <text x={W / 2} y={22} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          点柱子看度数 → 中心性 embedding 查表值
        </text>
        <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />
        {NODES.map((n, idx) => {
          const d = degree(n);
          const h = (d / maxDeg) * maxH;
          const x = PAD.left + idx * (barW + gap) + gap / 2;
          const active = selected === n;
          return (
            <g key={n} onClick={() => setSelected(n)} style={{ cursor: "pointer" }}>
              <rect x={x} y={H - PAD.bottom - h} width={barW} height={h} fill={active ? "#ec4899" : "#9ca3af"} rx={3} />
              <text x={x + barW / 2} y={H - PAD.bottom - h - 6} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
                deg={d}
              </text>
              <text x={x + barW / 2} y={H - PAD.bottom + 18} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">
                节点 {n}
              </text>
            </g>
          );
        })}
      </svg>
      {selected != null && (
        <div style={{ marginTop: "var(--space-3)", padding: "var(--space-3)", border: "1px solid #ec4899", borderRadius: "var(--radius-md)", background: "#fce7f3" }}>
          <span style={{ fontSize: "var(--fs-sm)", color: "#9d174d" }}>
            节点 {selected}:度数 = {degree(selected)} → centrality embedding = <strong>{centralityEmbedding(selected).toFixed(2)}</strong>
          </span>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 5: 写 `stages/CentralityStage.tsx`**

```tsx
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GRAPHORMER_SOURCE_PATH } from "../lib/prose";
import { CentralityHistogramWidget } from "../widgets/CentralityHistogramWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function CentralityStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:中心性编码
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        标准 Transformer 的全局注意力天然看不见"谁是图里的枢纽节点"。Graphormer 按节点度数查一张 embedding 表,把"这个节点有多重要/多中心"直接加进输入表示里。
      </p>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={GRAPHORMER_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={GRAPHORMER_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <CentralityHistogramWidget />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 6: 写 `widgets/NodePairSelectorWidget.tsx`**

```tsx
import { NODES } from "../lib/data";

interface Props {
  i: number;
  j: number;
  onSelectI: (n: number) => void;
  onSelectJ: (n: number) => void;
}

export function NodePairSelectorWidget({ i, j, onSelectI, onSelectJ }: Props) {
  const row = (label: string, current: number, onSelect: (n: number) => void) => (
    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: "var(--space-2)" }}>
      <span style={{ width: 60, fontSize: "var(--fs-sm)", color: "var(--ink-secondary)" }}>{label}</span>
      {NODES.map((n) => (
        <button
          key={n} type="button" onClick={() => onSelect(n)}
          style={{
            width: 28, height: 28, borderRadius: "var(--radius-sm)",
            border: `1px solid ${n === current ? "#ec4899" : "var(--border)"}`,
            background: n === current ? "#ec4899" : "var(--bg-surface)",
            color: n === current ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-xs)",
          }}
        >
          {n}
        </button>
      ))}
    </div>
  );

  return (
    <div style={{ marginBottom: "var(--space-4)" }}>
      {row("节点 i", i, onSelectI)}
      {row("节点 j", j, onSelectJ)}
    </div>
  );
}
```

- [ ] **Step 7: 写 `widgets/SpatialBiasHeatmapWidget.tsx`**

```tsx
import { NODES, baseScore, shortestPath, spatialBias } from "../lib/data";

interface Props {
  withSpatial: boolean;
}

// 6x6 attention score 热力图:withSpatial=false 只显示 base QK score,
// withSpatial=true 显示 base + spatialBias(shortest-path distance)。

export function SpatialBiasHeatmapWidget({ withSpatial }: Props) {
  const scores = NODES.map((i) => NODES.map((j) => {
    const base = baseScore(i, j);
    if (!withSpatial || i === j) return base;
    const { distance } = shortestPath(i, j);
    return base + spatialBias(distance);
  }));

  const all = scores.flat();
  const min = Math.min(...all);
  const max = Math.max(...all);
  const cellSize = 42;

  const colorFor = (v: number) => {
    const t = (v - min) / (max - min || 1);
    const lightness = 90 - t * 55;
    return `hsl(330, 70%, ${lightness}%)`;
  };

  return (
    <div>
      <div style={{ display: "inline-block" }}>
        <div style={{ display: "flex", marginLeft: 32 }}>
          {NODES.map((j) => (
            <div key={j} style={{ width: cellSize, textAlign: "center", fontSize: "var(--fs-xs)", color: "var(--ink-muted)" }}>{j}</div>
          ))}
        </div>
        {NODES.map((i) => (
          <div key={i} style={{ display: "flex", alignItems: "center" }}>
            <div style={{ width: 32, fontSize: "var(--fs-xs)", color: "var(--ink-muted)", textAlign: "right", paddingRight: 4 }}>{i}</div>
            {NODES.map((j) => (
              <div
                key={j}
                style={{
                  width: cellSize, height: cellSize, display: "flex", alignItems: "center", justifyContent: "center",
                  background: colorFor(scores[i][j]), fontSize: "var(--fs-xs)", border: "1px solid var(--bg-canvas)",
                }}
              >
                {scores[i][j].toFixed(1)}
              </div>
            ))}
          </div>
        ))}
      </div>
      <p style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: "var(--space-2)" }}>
        {withSpatial ? "已叠加最短路径距离的空间 bias —— 远的节点分数被压低" : "纯 base QK score,还没有任何图结构信息"}
      </p>
    </div>
  );
}
```

- [ ] **Step 8: 写 `stages/SpatialStage.tsx`**

```tsx
import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GRAPHORMER_SOURCE_PATH } from "../lib/prose";
import { NodePairSelectorWidget } from "../widgets/NodePairSelectorWidget";
import { SpatialBiasHeatmapWidget } from "../widgets/SpatialBiasHeatmapWidget";
import { shortestPath } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function SpatialStage({ mechanism2Prose }: Props) {
  const [i, setI] = useState(0);
  const [j, setJ] = useState(4);
  const [withSpatial, setWithSpatial] = useState(false);
  const { distance } = shortestPath(i, j);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:空间编码 — 核心创新
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        计算任意两个节点间的最短路径距离,把这个距离映射成一个 bias 项,直接加到 attention score 上:A_ij = QK^T/√d + b_φ(i,j)。距离越远,bias 越负,注意力天然衰减。
      </p>

      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={GRAPHORMER_SOURCE_PATH} />
        </div>

        <div className={styles.stickyPanel}>
          <NodePairSelectorWidget i={i} j={j} onSelectI={setI} onSelectJ={setJ} />
          <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
            节点 {i} → 节点 {j} 最短路径距离 = {Number.isFinite(distance) ? distance : "不可达"}
          </p>
          <button
            type="button" onClick={() => setWithSpatial((v) => !v)}
            style={{
              padding: "4px 14px", borderRadius: "var(--radius-sm)", marginBottom: "var(--space-3)",
              border: `1px solid ${withSpatial ? "#ec4899" : "var(--border)"}`,
              background: withSpatial ? "#ec4899" : "var(--bg-surface)",
              color: withSpatial ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-sm)",
            }}
          >
            {withSpatial ? "✓ 已叠加空间 bias" : "叠加空间 bias"}
          </button>
          <SpatialBiasHeatmapWidget withSpatial={withSpatial} />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 9: 跑 typecheck**

Run: `cd web && npx tsc --noEmit`
Expected: 无错误

- [ ] **Step 10: Commit**

```bash
git add web/src/components/node/golden/graphormer/lib web/src/components/node/golden/graphormer/stages web/src/components/node/golden/graphormer/widgets
git commit -m "feat: Graphormer 金标本 lib + Stage1/2"
```

---

## Task 10: Graphormer — Stage3(边编码)+ NodePage + 注册

**Files:**
- Create: `web/src/components/node/golden/graphormer/widgets/EdgeBiasHeatmapWidget.tsx`
- Create: `web/src/components/node/golden/graphormer/stages/EdgeStage.tsx`
- Create: `web/src/components/node/golden/graphormer/NodePageGraphormer.tsx`
- Create: `web/src/components/node/golden/graphormer/NodePageGraphormer.module.css`
- Modify: `web/src/components/node/golden/index.ts`

- [ ] **Step 1: 写 `widgets/EdgeBiasHeatmapWidget.tsx`**

```tsx
import { NODES, POSITIONS, EDGES, baseScore, shortestPath, spatialBias, edgeBias } from "../lib/data";

interface Props {
  i: number;
  j: number;
  layer: "base" | "spatial" | "spatial+edge";
}

// 复用 Task 9 的 6x6 热力图思路,但这里聚焦单个 (i,j) pair 的分数分解,
// 并在小图上高亮最短路径,展示 base → +spatial → +spatial+edge 三层累加。

export function EdgeBiasHeatmapWidget({ i, j, layer }: Props) {
  const { distance, path } = shortestPath(i, j);
  const base = baseScore(i, j);
  const withSpatial = base + (i === j ? 0 : spatialBias(distance));
  const withEdge = withSpatial + (i === j ? 0 : edgeBias(path));

  const score = layer === "base" ? base : layer === "spatial" ? withSpatial : withEdge;
  const pathEdges = new Set(path.slice(0, -1).map((n, idx) => `${n}-${path[idx + 1]}`));

  const W = 500;
  const H = 300;

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`节点 ${i} 到 ${j} 的 attention bias 分解`}>
        {EDGES.map((e, idx) => {
          const [x1, y1] = POSITIONS[e.a];
          const [x2, y2] = POSITIONS[e.b];
          const onPath = pathEdges.has(`${e.a}-${e.b}`) || pathEdges.has(`${e.b}-${e.a}`);
          return <line key={idx} x1={x1} y1={y1} x2={x2} y2={y2} stroke={onPath ? "#ec4899" : "var(--border)"} strokeWidth={onPath ? 3 : 1.5} />;
        })}
        {NODES.map((n) => {
          const [x, y] = POSITIONS[n];
          const highlight = n === i || n === j;
          return (
            <g key={n}>
              <circle cx={x} cy={y} r={20} fill={highlight ? "#ec4899" : "var(--bg-surface)"} stroke="#ec4899" strokeWidth={2} />
              <text x={x} y={y + 5} textAnchor="middle" fontSize={12} fontWeight={700} fill={highlight ? "#fff" : "var(--ink-primary)"}>
                {n}
              </text>
            </g>
          );
        })}
      </svg>

      <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "var(--space-3)" }}>
        <tbody>
          <tr>
            <td style={{ padding: "4px 8px", fontSize: "var(--fs-sm)" }}>base QK score</td>
            <td style={{ padding: "4px 8px", fontSize: "var(--fs-sm)", textAlign: "right" }}>{base.toFixed(2)}</td>
          </tr>
          <tr>
            <td style={{ padding: "4px 8px", fontSize: "var(--fs-sm)" }}>+ 空间 bias(距离={Number.isFinite(distance) ? distance : "∞"})</td>
            <td style={{ padding: "4px 8px", fontSize: "var(--fs-sm)", textAlign: "right" }}>{withSpatial.toFixed(2)}</td>
          </tr>
          <tr>
            <td style={{ padding: "4px 8px", fontSize: "var(--fs-sm)" }}>+ 边编码(路径上边特征累加)</td>
            <td style={{ padding: "4px 8px", fontSize: "var(--fs-sm)", textAlign: "right" }}>{withEdge.toFixed(2)}</td>
          </tr>
          <tr style={{ borderTop: "1px solid var(--border)" }}>
            <td style={{ padding: "4px 8px", fontSize: "var(--fs-sm)", fontWeight: 700 }}>当前展示层({layer})</td>
            <td style={{ padding: "4px 8px", fontSize: "var(--fs-sm)", fontWeight: 700, textAlign: "right", color: "#9d174d" }}>{score.toFixed(2)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 2: 写 `stages/EdgeStage.tsx`**

```tsx
import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GRAPHORMER_SOURCE_PATH } from "../lib/prose";
import { NodePairSelectorWidget } from "../widgets/NodePairSelectorWidget";
import { EdgeBiasHeatmapWidget } from "../widgets/EdgeBiasHeatmapWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function EdgeStage({ mechanism3Prose, synergyProse }: Props) {
  const [i, setI] = useState(0);
  const [j, setJ] = useState(4);
  const [layer, setLayer] = useState<"base" | "spatial" | "spatial+edge">("base");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:边编码
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        在空间编码之上,把最短路径上每条边自身的特征也编码进 bias —— 不只是"隔多远",还考虑"沿途经过了什么样的边"。三层 bias(base / +spatial / +spatial+edge)逐步叠加。
      </p>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={GRAPHORMER_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={GRAPHORMER_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <NodePairSelectorWidget i={i} j={j} onSelectI={setI} onSelectJ={setJ} />
          <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-3)" }}>
            {(["base", "spatial", "spatial+edge"] as const).map((l) => (
              <button
                key={l} type="button" onClick={() => setLayer(l)}
                style={{
                  padding: "4px 10px", borderRadius: "var(--radius-sm)",
                  border: `1px solid ${layer === l ? "#ec4899" : "var(--border)"}`,
                  background: layer === l ? "#ec4899" : "var(--bg-surface)",
                  color: layer === l ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-xs)",
                }}
              >
                {l}
              </button>
            ))}
          </div>
          <EdgeBiasHeatmapWidget i={i} j={j} layer={layer} />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: 写 `NodePageGraphormer.module.css`**

复制通用模板。

- [ ] **Step 4: 写 `NodePageGraphormer.tsx`**

```tsx
import { Link } from "react-router";
import graphormerMarkdown from "../../../../../../17-graph-neural-networks/05-graphormer.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, GRAPHORMER_SOURCE_PATH } from "./lib/prose";
import { CentralityStage } from "./stages/CentralityStage";
import { SpatialStage } from "./stages/SpatialStage";
import { EdgeStage } from "./stages/EdgeStage";
import styles from "./NodePageGraphormer.module.css";

const prose = extractProse(graphormerMarkdown);

export default function NodePageGraphormer() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/17-graph-neural-networks" className={styles.back}>
          ← 返回图神经网络(GNN)
        </Link>
        <h1 className={styles.title}>Graphormer (2021)</h1>
        <div className={styles.metaLine}>
          作者:Chengxuan Ying · Tianle Cai · Shengjie Luo · Shuxin Zheng · Guolin Ke · Di He · Yanming Shen · Tie-Yan Liu
        </div>
        <div className={styles.metaLine}>论文:Do Transformers Really Perform Bad for Graph Representation?</div>
        <p className={styles.keyIdea}>
          中心性编码 + 空间编码(最短路径距离)+ 边编码把图结构信息直接注入 attention,用全局注意力替代逐跳消息传递
        </p>
      </section>

      <section className={styles.stage}>
        <CentralityStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <SpatialStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <EdgeStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={GRAPHORMER_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={GRAPHORMER_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={GRAPHORMER_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
```

- [ ] **Step 5: 注册进 `index.ts`**

```typescript
  "17-graph-neural-networks/05-graphormer": lazy(() => import("./graphormer/NodePageGraphormer")),
```

- [ ] **Step 6: 跑测试**

Run: `cd web && npx vitest run AllGoldenSamples.smoke ProseCompleteness && npx tsc --noEmit`
Expected: 全部通过

- [ ] **Step 7: Commit**

```bash
git add web/src/components/node/golden/graphormer web/src/components/node/golden/index.ts
git commit -m "feat: Graphormer 金标本 Stage3 + NodePage + 注册"
```

---

## Task 11: 全项目验证

**Files:** 无新建/修改,纯验证任务。

- [ ] **Step 1: 跑全量 vitest**

Run: `cd web && npx vitest run`
Expected: 全部通过,`AllGoldenSamples.smoke.test.tsx` 新增 5 条(GCN/GraphSAGE/GAT/GIN/Graphormer)冒烟用例 PASS,`ProseCompleteness.test.tsx` 新增 5 个 prose 模块用例 PASS,总用例数比 Task 开始前多至少 10 条

- [ ] **Step 2: 跑 tsc**

Run: `cd web && npx tsc --noEmit`
Expected: 无错误

- [ ] **Step 3: 浏览器验证(用 preview_start 起 web/ 的 dev server)**

依次打开:
- `/families/17-graph-neural-networks/01-gcn` — 确认渲染出 3 个可交互 Stage(而不是纯 markdown 兜底),点自环按钮/切换中心节点/拖 hops 按钮均有响应
- `/families/17-graph-neural-networks/02-graphsage` — 确认采样/聚合器对比/归纳式泛化三个 Stage 均渲染,交互按钮有响应
- `/families/17-graph-neural-networks/03-gat` — 确认注意力/mask+softmax/多头三个 Stage 均渲染,温度滑块/多头切换有响应
- `/families/17-graph-neural-networks/04-gin` — 确认反例/ε 滑块/WL 染色三个 Stage 均渲染,点"跑一轮 WL 精细化"颜色会变化
- `/families/17-graph-neural-networks/05-graphormer` — 确认中心性/空间编码/边编码三个 Stage 均渲染,节点选择器切换有响应

每页检查 `read_console_messages({ onlyErrors: true })` 确认无 console error。

- [ ] **Step 4: Commit(若浏览器验证发现小问题并修复)**

```bash
git add -A
git commit -m "fix: GNN 金标本浏览器验证发现的问题修复"
```

（若验证全部通过、无需修复,跳过此步)

---

## Plan Self-Review 记录

- **Spec 覆盖**:spec 第 4 节列出的 5 个节点、每节点 3 个 Stage 的设计要点均对应到 Task 1-10 里的具体 Stage/Widget;spec 第 7 节验收标准对应 Task 11。
- **Placeholder 扫描**:Task 5 Step 1 里 `lib/data.ts` 代码块末尾误留了一行 `</br>` 占位符,已在该步骤下方加注说明写文件时必须删除,不构成遗留 placeholder。其余步骤均为完整可运行代码,无 TBD/TODO。
- **类型一致性**:`ProseSections` 接口在"关键背景"通用模板中统一定义一次,5 个节点的 `lib/prose.ts` 都从这个模板复制,字段名(`previousWork/intuition/mechanism1/mechanism2/mechanism3/synergy/keyCode/performance/aftermath`)在所有 Stage 组件的 props 里保持一致引用。`extractProse()` 签名与返回类型在所有任务中一致。
