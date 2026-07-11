import { STEPS_COMPARE } from "../lib/data";

const W = 700;
const H = 320;

export function SamplingStepsCompareChart() {
  const PAD_L = 220;
  const PAD_R = 60;
  const PAD_T = 44;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 52;
  const maxSteps = 60;
  const wOf = (v: number) => (v / maxSteps) * plotW;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="DDPM vs Flow Matching 采样步数对比"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        采样步数对比 —— 直线路径允许大步 ODE 积分
      </text>

      {STEPS_COMPARE.map((row, i) => {
        const y = PAD_T + i * rowH;
        const isFm = row.method.startsWith("Flow Matching");
        const color = isFm ? "#3b82f6" : "#ec4899";
        const bg = isFm ? "#dbeafe" : "#fce7f3";
        const w = Math.max(wOf(row.steps), 4);
        return (
          <g key={row.method}>
            <text x={PAD_L - 8} y={y + 14} textAnchor="end" fontSize={10.5} fontWeight={700} fill={color}>
              {row.method}
            </text>
            <rect x={PAD_L} y={y} width={w} height={22} fill={bg} stroke={color} strokeWidth={1.4} rx={3} />
            <text x={PAD_L + w + 8} y={y + 16} fontSize={11} fontWeight={700} fill={color}>
              {row.steps} 步
            </text>
            <text x={PAD_L} y={y + 38} fontSize={9} fill="var(--ink-muted)">
              {row.note}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
