import { RECEPTIVE_FIELD_GROWTH } from "../lib/data";

const W = 700;
const H = 220;

export function ReceptiveFieldGrowthDiagram() {
  const n = RECEPTIVE_FIELD_GROWTH.length;
  const colW = 100;
  const gap = 20;
  const startX = 50;
  const baseY = 190;
  const maxField = Math.max(...RECEPTIVE_FIELD_GROWTH.map((s) => s.fieldSize));
  const scale = 110 / maxField;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="LeNet-5 逐层感受野在原图上的等效范围增长"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        感受野随层数增长(相对原图的等效范围)
      </text>

      {RECEPTIVE_FIELD_GROWTH.map((step, i) => {
        const x = startX + i * (colW + gap);
        const size = step.fieldSize * scale;
        return (
          <g key={step.label}>
            <rect
              x={x + (colW - size) / 2}
              y={baseY - size}
              width={size}
              height={size}
              fill="#fef3c7"
              stroke="#f59e0b"
              strokeWidth={1.6}
              rx={3}
            />
            <text x={x + colW / 2} y={baseY + 18} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
              {step.label}
            </text>
            <text x={x + colW / 2} y={baseY - size - 8} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
              ≈{step.fieldSize}×{step.fieldSize}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        每经过一次卷积 / 池化,深层 neuron 在原图上能"看到"的范围就更大
      </text>
    </svg>
  );
}
