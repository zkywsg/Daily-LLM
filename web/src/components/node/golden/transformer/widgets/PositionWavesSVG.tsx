import { useMemo } from "react";
import { positionalEncoding } from "../lib/math";

interface Props {
  nPos: number;
  dModel: number;
  /** 可选:只画指定维度的曲线;否则画前 8 个 */
  highlightDim?: number;
}

const W = 700;
const H = 280;
const PADDING = { left: 40, right: 20, top: 30, bottom: 30 };

export function PositionWavesSVG({ nPos, dModel, highlightDim }: Props) {
  const PE = useMemo(() => positionalEncoding(nPos, dModel), [nPos, dModel]);
  const innerW = W - PADDING.left - PADDING.right;
  const innerH = H - PADDING.top - PADDING.bottom;
  const xs = Array.from({ length: nPos }, (_, p) => p);
  const xScale = (p: number) =>
    PADDING.left + (p / Math.max(1, nPos - 1)) * innerW;
  const yScale = (v: number) =>
    PADDING.top + ((1 - v) / 2) * innerH; // v ∈ [-1, 1]

  const dimsToDraw = highlightDim != null
    ? [highlightDim]
    : Array.from({ length: Math.min(8, dModel) }, (_, i) => i);

  // 用 hue 给不同维度区分颜色,低维短波长 → 暖色;高维长波长 → 冷色
  const colorFor = (d: number) => {
    const t = d / Math.max(1, dModel - 1);
    return `hsl(${20 + t * 220}, 70%, ${highlightDim != null ? 50 : 55}%)`;
  };

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="位置编码 sin/cos 波形"
    >
      {/* y 轴 */}
      <line
        x1={PADDING.left}
        y1={PADDING.top}
        x2={PADDING.left}
        y2={H - PADDING.bottom}
        stroke="var(--border)"
      />
      <text
        x={PADDING.left - 6}
        y={yScale(1) + 4}
        textAnchor="end"
        fontSize={10}
        fill="var(--ink-muted)"
      >
        +1
      </text>
      <text
        x={PADDING.left - 6}
        y={yScale(0) + 4}
        textAnchor="end"
        fontSize={10}
        fill="var(--ink-muted)"
      >
        0
      </text>
      <text
        x={PADDING.left - 6}
        y={yScale(-1) + 4}
        textAnchor="end"
        fontSize={10}
        fill="var(--ink-muted)"
      >
        -1
      </text>
      <line
        x1={PADDING.left}
        y1={yScale(0)}
        x2={W - PADDING.right}
        y2={yScale(0)}
        stroke="var(--border)"
        strokeDasharray="2 4"
      />

      {/* x 轴 */}
      <line
        x1={PADDING.left}
        y1={H - PADDING.bottom}
        x2={W - PADDING.right}
        y2={H - PADDING.bottom}
        stroke="var(--border)"
      />
      <text
        x={W / 2}
        y={H - 8}
        textAnchor="middle"
        fontSize={11}
        fill="var(--ink-muted)"
      >
        position pos →
      </text>

      {/* 曲线 */}
      {dimsToDraw.map((d) => {
        const pts = xs
          .map((p) => `${xScale(p)},${yScale(PE[p][d])}`)
          .join(" ");
        return (
          <g key={d}>
            <polyline
              fill="none"
              stroke={colorFor(d)}
              strokeWidth={highlightDim != null ? 2.5 : 1.6}
              points={pts}
              opacity={highlightDim != null ? 1 : 0.8}
            />
            <text
              x={W - PADDING.right + 2}
              y={yScale(PE[nPos - 1][d]) + 3}
              fontSize={9}
              fill={colorFor(d)}
            >
              dim {d}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
