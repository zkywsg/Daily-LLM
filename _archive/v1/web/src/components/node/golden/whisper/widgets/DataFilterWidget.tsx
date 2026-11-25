import { useState } from "react";
import { RAW_QUALITY_SCORES, filterLowQuality } from "../lib/data";

const W = 500;
const H = 260;
const BINS = [0, 0.2, 0.4, 0.6, 0.8, 1.0];

function histogram(scores: number[]): number[] {
  const counts = new Array(BINS.length - 1).fill(0);
  scores.forEach((s) => {
    for (let b = 0; b < BINS.length - 1; b++) {
      if (s >= BINS[b] && (s < BINS[b + 1] || (b === BINS.length - 2 && s <= BINS[b + 1]))) {
        counts[b]++;
        break;
      }
    }
  });
  return counts;
}

export function DataFilterWidget() {
  const [filtered, setFiltered] = useState(false);
  const scores = filtered ? filterLowQuality(RAW_QUALITY_SCORES) : RAW_QUALITY_SCORES;
  const counts = histogram(scores);
  const maxCount = Math.max(...counts, 1);

  const barW = (W - 80) / counts.length - 8;

  return (
    <div>
      <button
        type="button" onClick={() => setFiltered((v) => !v)} aria-pressed={filtered}
        style={{
          padding: "4px 14px", borderRadius: "var(--radius-sm)", marginBottom: "var(--space-3)",
          border: `1px solid ${filtered ? "#fb7185" : "var(--border)"}`,
          background: filtered ? "#fb7185" : "var(--bg-surface)",
          color: filtered ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-sm)",
        }}
      >
        {filtered ? "✓ 已过滤低质量样本" : "过滤低质量样本"}
      </button>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`数据质量分布直方图,${filtered ? "已过滤" : "未过滤"}`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          数据质量分布(共 {scores.length} 条样本)
        </text>
        <line x1={40} y1={H - 40} x2={W - 40} y2={H - 40} stroke="var(--border)" />
        {counts.map((c, i) => {
          const x = 50 + i * ((W - 80) / counts.length);
          const h = Math.min((c / maxCount) * (H - 100), H - 100);
          return (
            <g key={i}>
              <rect x={x} y={H - 40 - h} width={barW} height={h} fill="#fb7185" />
              <text x={x + barW / 2} y={H - 40 - h - 6} textAnchor="middle" fontSize={10} fill="var(--ink-primary)">{c}</text>
              <text x={x + barW / 2} y={H - 20} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">{BINS[i].toFixed(1)}-{BINS[i + 1].toFixed(1)}</text>
            </g>
          );
        })}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        未过滤时低质量区间(可能是机器生成的伪转写)样本不少;过滤后低质量区间样本明显减少,保留的都是质量分 ≥0.5 的样本。
      </p>
    </div>
  );
}
