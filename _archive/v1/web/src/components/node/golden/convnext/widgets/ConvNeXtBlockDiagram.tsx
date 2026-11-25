import { CONVNEXT_BLOCK_STEPS } from "../lib/data";

const W = 700;
const H = 460;

const FILL_BY_INDEX = [
  { bg: "#fce7f3", stroke: "#ec4899" }, // DWConv 7x7 — token mixer
  { bg: "#dbeafe", stroke: "#3b82f6" }, // LayerNorm
  { bg: "#fef3c7", stroke: "#f59e0b" }, // PWConv up
  { bg: "#ecfdf5", stroke: "#10b981" }, // GELU
  { bg: "#fef3c7", stroke: "#f59e0b" }, // PWConv down
  { bg: "#f3f4f6", stroke: "#9ca3af" }, // shortcut
];

export function ConvNeXtBlockDiagram() {
  const boxW = 260;
  const boxH = 46;
  const gapY = 20;
  const startY = 44;
  const cx = W / 2 - 70;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="ConvNeXt Block 内部结构,类比 Transformer FFN block"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        ConvNeXt Block ↔ Transformer Block
      </text>

      {CONVNEXT_BLOCK_STEPS.map((step, i) => {
        const y = startY + i * (boxH + gapY);
        const { bg, stroke } = FILL_BY_INDEX[i];
        return (
          <g key={`${step.label}-${i}`}>
            <rect x={cx - boxW / 2} y={y} width={boxW} height={boxH} fill={bg} stroke={stroke} strokeWidth={1.6} rx={6} />
            <text x={cx} y={y + 19} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--ink-primary)">
              {step.label}
            </text>
            <text x={cx} y={y + 34} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
              {step.detail}
            </text>

            {/* role annotation to the right, mapping to Transformer */}
            <text x={cx + boxW / 2 + 20} y={y + boxH / 2 + 4} fontSize={10} fill="var(--ink-secondary)">
              {step.role}
            </text>

            {i < CONVNEXT_BLOCK_STEPS.length - 1 && (
              <line
                x1={cx}
                y1={y + boxH}
                x2={cx}
                y2={y + boxH + gapY}
                stroke="var(--border)"
                strokeWidth={1.5}
                markerEnd="url(#arrow-convnext-block)"
              />
            )}
          </g>
        );
      })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={9.5} fill="var(--ink-muted)">
        DWConv 7×7 ≈ self-attention(局部 token mixer)· PWConv↑ + GELU + PWConv↓ ≈ Transformer FFN
      </text>

      <defs>
        <marker id="arrow-convnext-block" markerWidth={8} markerHeight={8} refX={6} refY={4} orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="var(--border)" />
        </marker>
      </defs>
    </svg>
  );
}
