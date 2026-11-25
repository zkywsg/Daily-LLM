import { PRECISION_COMPARE } from "../lib/data";

interface Props {
  mode: "fp32-all" | "bf16-all" | "selective";
}

const W = 700;
const H = 280;

// 三种精度方案对比:全 fp32(稳定但贵)/ 全 bf16(便宜但易发散)/
// selective precision(router 用 fp32,其余 bf16 —— Switch 的方案)。

export function PrecisionCompareDiagram({ mode }: Props) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`Precision scheme comparison, selected ${mode}`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Selective Precision — router 用 fp32,其余用 bf16
      </text>

      {PRECISION_COMPARE.map((row, i) => {
        const y = 50 + i * 72;
        const isCur = row.mode === mode;
        const stableColor = row.stable ? "#10b981" : "#dc2626";
        return (
          <g key={row.mode} opacity={isCur ? 1 : 0.55}>
            <rect
              x={30}
              y={y}
              width={W - 60}
              height={56}
              rx={6}
              fill={isCur ? "#fef3c7" : "var(--bg-surface)"}
              stroke={isCur ? "#f59e0b" : "var(--border)"}
              strokeWidth={isCur ? 2 : 1}
            />
            <text x={50} y={y + 22} fontSize={12} fontWeight={700} fill="var(--ink-primary)">
              {row.label}
            </text>
            <text x={50} y={y + 40} fontSize={10} fill="var(--ink-secondary)">
              {row.note}
            </text>
            <circle cx={W - 110} cy={y + 20} r={5} fill={stableColor} />
            <text x={W - 100} y={y + 24} fontSize={10} fontWeight={600} fill={stableColor}>
              {row.stable ? "稳定" : "易发散"}
            </text>
            <text x={W - 50} y={y + 24} fontSize={10} fontWeight={600} fill="var(--ink-muted)" textAnchor="end">
              显存 {row.memoryX.toFixed(2)}×
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        router 的 softmax / log 对精度敏感 · 只给这一小块用 fp32,几乎不多花显存却换来训练稳定
      </text>
    </svg>
  );
}
