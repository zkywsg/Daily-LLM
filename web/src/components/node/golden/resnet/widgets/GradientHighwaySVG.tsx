import { GRADIENT_DECAY_PER_BLOCK } from "../lib/curves";

interface Props {
  stackDepth: number;
  width?: number;
  height?: number;
}

const PADDING_X = 30;
const BLOCK_W = 50;
const BLOCK_H = 30;
const BLOCK_Y = 160;

export function GradientHighwaySVG({ stackDepth, width = 560, height = 360 }: Props) {
  const innerW = width - 2 * PADDING_X;
  const visibleCount = Math.min(stackDepth, 8);
  const blockGap =
    visibleCount > 1 ? (innerW - visibleCount * BLOCK_W) / (visibleCount - 1) : 0;

  const blocks = Array.from({ length: visibleCount }, (_, i) => ({
    index: i,
    x: PADDING_X + i * (BLOCK_W + blockGap),
  }));

  const totalDecay = Math.pow(GRADIENT_DECAY_PER_BLOCK, stackDepth);
  const decayWidths = blocks.map((_, i) => {
    const remaining = Math.pow(GRADIENT_DECAY_PER_BLOCK, stackDepth - i);
    return Math.max(0.3, 2.5 * remaining);
  });

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ maxWidth: "100%", height: "auto", display: "block" }}
      role="img"
      aria-label={`${stackDepth} 块串联的梯度高速公路`}
    >
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

      {blocks.map((b) => (
        <g key={b.index}>
          <rect x={b.x} y={BLOCK_Y} width={BLOCK_W} height={BLOCK_H} rx={4}
            fill="#fce7f3" stroke="#db2777" strokeWidth={1.5} />
          <text x={b.x + BLOCK_W / 2} y={BLOCK_Y + 19} textAnchor="middle" fontSize={10} fill="#9d174d">
            B{b.index + 1}
          </text>
          <path
            d={`M ${b.x + 5} ${BLOCK_Y} C ${b.x + 5} ${BLOCK_Y - 20}, ${b.x + BLOCK_W - 5} ${BLOCK_Y - 20}, ${b.x + BLOCK_W - 5} ${BLOCK_Y}`}
            stroke="#2563eb"
            strokeWidth={1.5}
            fill="none"
          />
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

      <text
        x={width - PADDING_X}
        y={BLOCK_Y - 10}
        textAnchor="end"
        fontSize={12}
        fontWeight={600}
        fill="var(--ink-primary)"
      >
        ∂L/∂y
      </text>

      {blocks
        .slice()
        .reverse()
        .map((b) => {
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

      <line
        x1={width - PADDING_X}
        y1={BLOCK_Y + 90}
        x2={PADDING_X}
        y2={BLOCK_Y + 90}
        stroke="#2563eb"
        strokeWidth={2.5}
      />

      <text x={PADDING_X} y={BLOCK_Y + 55} fontSize={11} fill="#dc2626">
        主路：到达 input 时残留 {(totalDecay * 100).toFixed(1)}%
      </text>
      <text x={PADDING_X} y={BLOCK_Y + 110} fontSize={11} fill="#2563eb">
        shortcut：恒粗，无衰减
      </text>
    </svg>
  );
}
