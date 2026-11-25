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
