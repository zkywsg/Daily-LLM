import { useMemo } from "react";
import { buildAttentionMask, type AttentionPattern } from "../lib/scaling";

interface Props {
  pattern: AttentionPattern;
  n?: number;
  stride?: number;
}

const W = 700;
const H = 360;
const MARGIN = { left: 60, top: 60, right: 60, bottom: 50 };

// 三种 attention pattern 的 causal mask 矩阵 —— 灰色 = 可 attend。
// dense 是全下三角(GPT-2/-3 主路);strided / fixed 是 sparse 变体(GPT-3 用 50% layer 走 sparse)。

const PATTERN_LABEL: Record<AttentionPattern, string> = {
  dense: "Dense (full causal)",
  strided: "Strided",
  fixed: "Fixed (Sparse Transformer)",
};

export function SparseAttentionPattern({ pattern, n = 32, stride = 4 }: Props) {
  const mask = useMemo(() => buildAttentionMask(pattern, n, stride), [pattern, n, stride]);

  const innerW = W - MARGIN.left - MARGIN.right;
  const innerH = H - MARGIN.top - MARGIN.bottom;
  const matSize = Math.min(innerW, innerH);
  const cell = matSize / n;

  let activeCount = 0;
  for (const row of mask) for (const v of row) if (v) activeCount++;
  const totalCausal = (n * (n + 1)) / 2;
  const ratio = (activeCount / totalCausal) * 100;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`Attention mask: ${pattern}`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--ink-primary)">
        {PATTERN_LABEL[pattern]} · 实际计算 {ratio.toFixed(0)}% causal slots
      </text>
      <text x={W / 2} y={38} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        粉色 = 可 attend(causal 下三角内) · 白色 = 被 mask 跳过
      </text>

      {/* 网格 */}
      {mask.map((row, i) =>
        row.map((v, j) => (
          <rect
            key={`${i}-${j}`}
            x={MARGIN.left + j * cell}
            y={MARGIN.top + i * cell}
            width={cell}
            height={cell}
            fill={v ? "#fce7f3" : "var(--bg-surface)"}
            stroke="var(--bg-canvas)"
            strokeWidth={0.3}
          />
        )),
      )}

      {/* 边框 */}
      <rect
        x={MARGIN.left}
        y={MARGIN.top}
        width={matSize}
        height={matSize}
        fill="none"
        stroke="var(--border)"
        strokeWidth={1.5}
      />

      {/* 轴标 */}
      <text x={MARGIN.left + matSize / 2} y={MARGIN.top - 8} textAnchor="middle" fontSize={11} fill="var(--ink-muted)">
        key →
      </text>
      <text
        x={MARGIN.left - 16}
        y={MARGIN.top + matSize / 2}
        textAnchor="middle"
        fontSize={11}
        fill="var(--ink-muted)"
        transform={`rotate(-90 ${MARGIN.left - 16} ${MARGIN.top + matSize / 2})`}
      >
        ↓ query
      </text>
    </svg>
  );
}
