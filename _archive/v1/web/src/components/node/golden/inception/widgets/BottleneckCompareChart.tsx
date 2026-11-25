import { BOTTLENECK_COMPARE } from "../lib/data";

const W = 700;
const H = 200;

export function BottleneckCompareChart() {
  const PAD_L = 260;
  const PAD_T = 44;
  const barMaxW = 340;
  const rowH = 46;
  const maxParams = Math.max(...BOTTLENECK_COMPARE.map((r) => r.params));

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="1×1 卷积瓶颈前后的 5×5 分支参数量对比"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        1×1 瓶颈:先压再算 — 5×5 分支参数量对比
      </text>

      {BOTTLENECK_COMPARE.map((row, i) => {
        const y = PAD_T + i * rowH;
        const w = (row.params / maxParams) * barMaxW;
        const fill = row.highlight ? "#10b981" : "#9ca3af";
        const bg = row.highlight ? "#ecfdf5" : "#f3f4f6";
        return (
          <g key={`${row.label}-${i}`}>
            <text x={PAD_L - 10} y={y + 14} textAnchor="end" fontSize={10} fill="var(--ink-secondary)">
              {row.label}
            </text>
            <text x={PAD_L - 10} y={y + 27} textAnchor="end" fontSize={9} fill="var(--ink-muted)">
              {row.formula}
            </text>
            <rect x={PAD_L} y={y} width={barMaxW} height={24} fill={bg} rx={3} />
            <rect x={PAD_L} y={y} width={Math.max(w, 4)} height={24} fill={fill} rx={3} />
            <text x={PAD_L + Math.max(w, 4) + 8} y={y + 17} fontSize={11} fontWeight={700} fill="var(--ink-primary)">
              {row.params.toLocaleString("en-US")}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={11} fontWeight={700} fill="#10b981">
        1×1 先压通道 → 5×5 参数降到约 1/8
      </text>
    </svg>
  );
}
