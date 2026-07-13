import { KERNEL_COMPARISON } from "../lib/data";

const W = 700;
const H = 200;

export function KernelParamSavingsChart() {
  const PAD_L = 220;
  const PAD_T = 36;
  const barMaxW = 380;
  const rowH = 36;
  const maxCoeff = Math.max(...KERNEL_COMPARISON.map((r) => r.paramsCoeff));

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="等价感受野下,大卷积与堆叠小卷积的参数量对比"
    >
      <text x={W / 2} y={18} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        等价感受野下的参数量(单位:C²)
      </text>

      {KERNEL_COMPARISON.map((row, i) => {
        const y = PAD_T + i * rowH;
        const w = (row.paramsCoeff / maxCoeff) * barMaxW;
        const fill = row.isStacked ? "#3b82f6" : "#9ca3af";
        const bg = row.isStacked ? "#dbeafe" : "#f3f4f6";
        return (
          <g key={`${row.receptiveField}-${row.approach}-${i}`}>
            <text x={PAD_L - 10} y={y + 15} textAnchor="end" fontSize={10} fill="var(--ink-secondary)">
              {row.receptiveField} · {row.approach}
            </text>
            <rect x={PAD_L} y={y} width={barMaxW} height={20} fill={bg} rx={3} />
            <rect x={PAD_L} y={y} width={Math.max(w, 4)} height={20} fill={fill} rx={3} />
            <text x={PAD_L + Math.max(w, 4) + 8} y={y + 15} fontSize={11} fontWeight={700} fill="var(--ink-primary)">
              {row.paramsFormula}
              {row.savingsLabel ? ` (${row.savingsLabel})` : ""}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
