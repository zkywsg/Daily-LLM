import { BENCHMARKS } from "../lib/data";

const W = 700;
const H = 280;

// 论文 Table:IMDb sentiment reward + HH win rate
export function BenchmarkBars() {
  const PAD_L = 100;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 30;
  const plotW = (W - PAD_L - PAD_R) / 2 - 20;
  const plotH = H - PAD_T - PAD_B;
  const rowH = plotH / BENCHMARKS.length - 6;

  const xOf = (v: number, max: number, x0: number) => x0 + (v / max) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="PPO vs DPO benchmarks">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        DPO 论文核心 benchmark — IMDb 与 HH-RLHF 胜率
      </text>

      <text x={PAD_L} y={45} fontSize={11} fontWeight={700} fill="#374151">IMDb sentiment reward (↑)</text>
      <text x={PAD_L + plotW + 40} y={45} fontSize={11} fontWeight={700} fill="#374151">HH-RLHF 胜率 (↑)</text>

      {BENCHMARKS.map((b, i) => {
        const y = PAD_T + 10 + i * (rowH + 6);
        const color = b.method === "DPO" ? "#10b981" : "#9ca3af";
        const bg = b.method === "DPO" ? "#ecfdf5" : "#f3f4f6";
        const w1 = xOf(b.imdb, 1.0, PAD_L) - PAD_L;
        const w2 = xOf(b.hhWinRate, 100, PAD_L + plotW + 40) - (PAD_L + plotW + 40);
        return (
          <g key={i}>
            <text x={PAD_L - 8} y={y + rowH / 2 + 4} textAnchor="end" fontSize={11} fontWeight={600} fill="#374151">{b.method}</text>

            {/* IMDb */}
            <rect x={PAD_L} y={y} width={w1} height={rowH} fill={bg} stroke={color} strokeWidth={1.4} rx={2} />
            <text x={PAD_L + w1 + 4} y={y + rowH / 2 + 4} fontSize={10} fontWeight={700} fill={color}>{b.imdb.toFixed(2)}</text>

            {/* HH */}
            <rect x={PAD_L + plotW + 40} y={y} width={w2} height={rowH} fill={bg} stroke={color} strokeWidth={1.4} rx={2} />
            <text x={PAD_L + plotW + 40 + w2 + 4} y={y + rowH / 2 + 4} fontSize={10} fontWeight={700} fill={color}>{b.hhWinRate}%</text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        DPO 在两个 benchmark 上都比 PPO 略好 + 工程简化 1 个数量级
      </text>
    </svg>
  );
}
