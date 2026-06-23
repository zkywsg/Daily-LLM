import { useMemo } from "react";
import {
  demoQKV,
  scaledDotProduct,
} from "../lib/math";

interface Props {
  tokens: string[];
  dModel: number;
  scaled: boolean;
  /** 用 raw scores 还是 softmax 后的 weights —— 给 viewer 看"为什么要 softmax" */
  view: "weights" | "scaled" | "raw";
}

const SIZE = 360;
const PADDING_LEFT = 80;
const PADDING_TOP = 80;

// 用 hsl 色阶,值大颜色深(粉调,对齐 fce7f3 compute 色)
function cellFill(v: number, minV: number, maxV: number): string {
  if (maxV === minV) return "hsl(330, 80%, 90%)";
  const t = (v - minV) / (maxV - minV); // 0..1
  const light = 95 - t * 50; // 95%(浅) → 45%(深)
  return `hsl(330, 80%, ${light}%)`;
}

export function AttentionHeatmap({ tokens, dModel, scaled, view }: Props) {
  const { matrix, label } = useMemo(() => {
    const { Q, K, V } = demoQKV(tokens, dModel);
    const r = scaledDotProduct(Q, K, V, scaled);
    if (view === "weights")
      return { matrix: r.weights, label: "softmax(QKᵀ/√dk)" };
    if (view === "scaled")
      return {
        matrix: r.scaledScores,
        label: scaled ? "QKᵀ/√dk(scaled)" : "QKᵀ(unscaled)",
      };
    return { matrix: r.scores, label: "QKᵀ raw" };
  }, [tokens, dModel, scaled, view]);

  const n = tokens.length;
  const cellW = (SIZE - PADDING_LEFT) / n;
  const cellH = (SIZE - PADDING_TOP) / n;
  const flat = matrix.flat();
  const minV = Math.min(...flat);
  const maxV = Math.max(...flat);

  return (
    <svg
      viewBox={`0 0 ${SIZE + 20} ${SIZE + 30}`}
      style={{
        width: "100%",
        height: "auto",
        fontFamily: "var(--font-mono, ui-monospace)",
      }}
      role="img"
      aria-label={`Attention matrix — ${label}`}
    >
      <text
        x={SIZE / 2 + 10}
        y={18}
        textAnchor="middle"
        fontSize={13}
        fontWeight={600}
        fill="var(--ink-primary)"
      >
        {label}
      </text>

      {/* Key tokens 横排标签 */}
      {tokens.map((t, j) => (
        <text
          key={`k-${j}`}
          x={PADDING_LEFT + cellW * (j + 0.5)}
          y={PADDING_TOP - 8}
          textAnchor="middle"
          fontSize={11}
          fill="var(--ink-secondary)"
          transform={`rotate(-30 ${PADDING_LEFT + cellW * (j + 0.5)} ${
            PADDING_TOP - 8
          })`}
        >
          {t}
        </text>
      ))}
      <text
        x={PADDING_LEFT + ((SIZE - PADDING_LEFT) / 2)}
        y={36}
        textAnchor="middle"
        fontSize={11}
        fill="var(--ink-muted)"
        fontStyle="italic"
      >
        K (keys) →
      </text>

      {/* Query tokens 纵排标签 */}
      {tokens.map((t, i) => (
        <text
          key={`q-${i}`}
          x={PADDING_LEFT - 8}
          y={PADDING_TOP + cellH * (i + 0.5) + 4}
          textAnchor="end"
          fontSize={11}
          fill="var(--ink-secondary)"
        >
          {t}
        </text>
      ))}
      <text
        x={20}
        y={PADDING_TOP + (SIZE - PADDING_TOP) / 2}
        textAnchor="middle"
        fontSize={11}
        fill="var(--ink-muted)"
        fontStyle="italic"
        transform={`rotate(-90 20 ${PADDING_TOP + (SIZE - PADDING_TOP) / 2})`}
      >
        Q (queries) ↓
      </text>

      {/* 热力图 cells */}
      {matrix.map((row, i) =>
        row.map((v, j) => (
          <g key={`c-${i}-${j}`}>
            <rect
              x={PADDING_LEFT + j * cellW}
              y={PADDING_TOP + i * cellH}
              width={cellW}
              height={cellH}
              fill={cellFill(v, minV, maxV)}
              stroke="var(--bg-canvas)"
              strokeWidth={0.5}
            />
            <text
              x={PADDING_LEFT + (j + 0.5) * cellW}
              y={PADDING_TOP + (i + 0.5) * cellH + 3}
              textAnchor="middle"
              fontSize={9}
              fill="var(--ink-primary)"
              style={{ pointerEvents: "none" }}
            >
              {view === "weights" ? v.toFixed(2) : v.toFixed(1)}
            </text>
          </g>
        )),
      )}
    </svg>
  );
}
