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
