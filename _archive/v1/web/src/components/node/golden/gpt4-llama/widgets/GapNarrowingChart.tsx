import { GAP_NARROWING } from "../lib/data";

const W = 700;
const H = 200;

export function GapNarrowingChart() {
  const PAD_L = 220;
  const PAD_T = 40;
  const plotW = 380;
  const rowH = 60;
  const maxMonths = 20;
  const wOf = (m: number) => (m / maxMonths) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="闭源开源差距缩短时间线">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        闭源-开源能力差距 — 从 18 个月缩短到 6 个月
      </text>

      {GAP_NARROWING.map((row, i) => {
        const y = PAD_T + i * rowH;
        const color = i === 0 ? "#9ca3af" : "#10b981";
        const bg = i === 0 ? "#f3f4f6" : "#ecfdf5";
        return (
          <g key={row.era}>
            <text x={PAD_L - 10} y={y + 18} textAnchor="end" fontSize={10} fontWeight={700} fill={color}>{row.era}</text>
            <rect x={PAD_L} y={y} width={Math.max(wOf(row.months), 4)} height={24} fill={bg} stroke={color} strokeWidth={1.6} rx={4} />
            <text x={PAD_L + Math.max(wOf(row.months), 4) + 8} y={y + 17} fontSize={11} fontWeight={700} fill={color}>{row.months} 个月</text>
          </g>
        );
      })}
    </svg>
  );
}
