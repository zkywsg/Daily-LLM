import { VGG16_PARAM_DISTRIBUTION, VGG16_TOTAL_PARAMS_M } from "../lib/data";

const W = 700;
const H = 210;

const COLORS = ["#ec4899", "#3b82f6", "#9ca3af"];
const BG_COLORS = ["#fce7f3", "#dbeafe", "#f3f4f6"];

// 138M 参数中 fc6 一层占 74%,fc7+fc8 占 15%,所有 13 层 conv 合计仅 11%
export function ParamDistributionChart() {
  const cx = 130;
  const cy = 105;
  const r = 80;

  let cumulativeAngle = -Math.PI / 2;
  const arcs = VGG16_PARAM_DISTRIBUTION.map((row, i) => {
    const angle = (row.share / 100) * Math.PI * 2;
    const startAngle = cumulativeAngle;
    const endAngle = cumulativeAngle + angle;
    cumulativeAngle = endAngle;
    const x1 = cx + r * Math.cos(startAngle);
    const y1 = cy + r * Math.sin(startAngle);
    const x2 = cx + r * Math.cos(endAngle);
    const y2 = cy + r * Math.sin(endAngle);
    const largeArc = angle > Math.PI ? 1 : 0;
    const path = `M${cx},${cy} L${x1},${y1} A${r},${r} 0 ${largeArc} 1 ${x2},${y2} Z`;
    return { ...row, path, color: COLORS[i % COLORS.length] };
  });

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="VGG-16 138M 参数的分布:fc6 占 74%"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        VGG-16 参数分布(共 {VGG16_TOTAL_PARAMS_M}M)
      </text>

      {arcs.map((a, i) => (
        <path key={`${a.label}-${i}`} d={a.path} fill={a.color} stroke="var(--bg-canvas)" strokeWidth={2} />
      ))}
      <circle cx={cx} cy={cy} r={40} fill="var(--bg-canvas)" />
      <text x={cx} y={cy - 4} textAnchor="middle" fontSize={16} fontWeight={700} fill="var(--ink-primary)">
        74%
      </text>
      <text x={cx} y={cy + 12} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
        在 fc6
      </text>

      {VGG16_PARAM_DISTRIBUTION.map((row, i) => {
        const legendY = 50 + i * 34;
        return (
          <g key={`legend-${row.label}-${i}`}>
            <rect x={280} y={legendY - 10} width={14} height={14} fill={BG_COLORS[i % BG_COLORS.length]} stroke={COLORS[i % COLORS.length]} strokeWidth={1.6} rx={2} />
            <text x={302} y={legendY - 6} fontSize={11} fontWeight={700} fill="var(--ink-primary)">
              {row.label}
            </text>
            <text x={302} y={legendY + 8} fontSize={10} fill="var(--ink-secondary)">
              ≈{row.params}M · {row.share}%
            </text>
          </g>
        );
      })}
    </svg>
  );
}
