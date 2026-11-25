import { CFG_SCALE_COMPARE } from "../lib/data";

const W = 700;
const H = 220;

export function CfgScaleCompareChart() {
  const PAD_L = 170;
  const PAD_R = 60;
  const PAD_T = 44;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 60;
  const maxScale = 12;
  const xOf = (v: number) => (v / maxScale) * plotW;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="SD 1.5 vs SD3 的 CFG scale 对比"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        CFG scale 对比 —— Flow Matching 训练的模型对条件控制更敏感
      </text>

      {CFG_SCALE_COMPARE.map((row, i) => {
        const y = PAD_T + i * rowH;
        const isFm = row.objective.startsWith("Flow Matching");
        const color = isFm ? "#3b82f6" : "#ec4899";
        const bg = isFm ? "#dbeafe" : "#fce7f3";
        const x1 = PAD_L + xOf(row.scaleMin);
        const x2 = PAD_L + xOf(row.scaleMax);
        return (
          <g key={row.model}>
            <text x={PAD_L - 8} y={y + 6} textAnchor="end" fontSize={10.5} fontWeight={700} fill={color}>
              {row.model}
            </text>
            <text x={PAD_L - 8} y={y + 20} textAnchor="end" fontSize={8.5} fill="var(--ink-muted)">
              {row.objective}
            </text>
            <line x1={x1} y1={y} x2={x2} y2={y} stroke={color} strokeWidth={10} strokeLinecap="round" opacity={0.85} />
            <rect x={x1 - 12} y={y - 12} width={24} height={24} fill={bg} stroke={color} strokeWidth={1.4} rx={4} />
            <text x={x1} y={y + 4} textAnchor="middle" fontSize={9} fontWeight={700} fill={color}>
              {row.scaleMin}
            </text>
            <rect x={x2 - 12} y={y - 12} width={24} height={24} fill={bg} stroke={color} strokeWidth={1.4} rx={4} />
            <text x={x2} y={y + 4} textAnchor="middle" fontSize={9} fontWeight={700} fill={color}>
              {row.scaleMax}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
