import { NORM_ACT_COUNTS } from "../lib/data";

const W = 700;
const H = 220;

export function NormActCountDiagram() {
  const rowH = 80;
  const startY = 40;
  const unitW = 26;
  const gap = 6;
  const startX = 220;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="ResNet Bottleneck 与 ConvNeXt Block 内 norm / 激活函数数量对比"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        每个 Block 里的 Norm / 激活函数数量
      </text>

      {NORM_ACT_COUNTS.map((row, i) => {
        const y = startY + i * rowH;
        return (
          <g key={row.name}>
            <text x={startX - 12} y={y + 18} textAnchor="end" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
              {row.name}
            </text>
            <text x={startX - 12} y={y + 32} textAnchor="end" fontSize={9} fill="var(--ink-muted)">
              {row.normType} + {row.actType}
            </text>

            {/* norm units */}
            {Array.from({ length: row.normCount }).map((_, j) => (
              <rect
                key={`norm-${j}`}
                x={startX + j * (unitW + gap)}
                y={y}
                width={unitW}
                height={26}
                fill="#dbeafe"
                stroke="#3b82f6"
                strokeWidth={1.3}
                rx={4}
              />
            ))}
            <text
              x={startX + row.normCount * (unitW + gap) + 6}
              y={y + 18}
              fontSize={9}
              fill="#3b82f6"
              fontWeight={700}
            >
              ×{row.normCount} norm
            </text>

            {/* act units */}
            {Array.from({ length: row.actCount }).map((_, j) => (
              <rect
                key={`act-${j}`}
                x={startX + j * (unitW + gap)}
                y={y + 34}
                width={unitW}
                height={26}
                fill="#ecfdf5"
                stroke="#10b981"
                strokeWidth={1.3}
                rx={4}
              />
            ))}
            <text
              x={startX + row.actCount * (unitW + gap) + 6}
              y={y + 52}
              fontSize={9}
              fill="#10b981"
              fontWeight={700}
            >
              ×{row.actCount} act
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={9.5} fill="var(--ink-muted)">
        ResNet Bottleneck 3 个 conv 就堆 3 组 BN+ReLU;ConvNeXt 照搬 Transformer,一个 block 只留 1 个 LN + 1 个 GELU
      </text>
    </svg>
  );
}
