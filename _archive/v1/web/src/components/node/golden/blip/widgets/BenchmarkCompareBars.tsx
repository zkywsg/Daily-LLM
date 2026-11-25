import { BENCHMARK_TABLE } from "../lib/data";

const W = 700;
const H = 300;

export function BenchmarkCompareBars() {
  const PAD_L = 130;
  const PAD_T = 50;
  const plotW = 420;
  const rowH = 55;
  const maxVal = 130;
  const wOf = (v: number) => (Math.max(v, 0) / maxVal) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Flamingo-80B vs BLIP-2 benchmark 对比">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        BLIP-2 用 1/6 参数达到甚至超过 Flamingo-80B
      </text>

      <g transform={`translate(${PAD_L}, 34)`}>
        <rect x={0} y={0} width={12} height={12} fill="#f3f4f6" stroke="#9ca3af" />
        <text x={18} y={10} fontSize={9} fill="#374151">Flamingo-80B</text>
        <rect x={120} y={0} width={12} height={12} fill="#ecfdf5" stroke="#10b981" />
        <text x={138} y={10} fontSize={9} fill="#374151">BLIP-2</text>
      </g>

      {BENCHMARK_TABLE.map((row, i) => {
        const y = PAD_T + i * rowH;
        const isNegative = row.delta < 0;
        return (
          <g key={row.task}>
            <text x={PAD_L - 10} y={y + 18} textAnchor="end" fontSize={10} fontWeight={700} fill="#374151">{row.task}</text>

            <rect x={PAD_L} y={y} width={Math.max(wOf(row.flamingo), 4)} height={16} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + Math.max(wOf(row.flamingo), 4) + 6} y={y + 13} fontSize={9} fill="#6b7280">{row.flamingo}</text>

            <rect x={PAD_L} y={y + 18} width={Math.max(wOf(row.blip2), 4)} height={16} fill={isNegative ? "#fce7f3" : "#ecfdf5"} stroke={isNegative ? "#ec4899" : "#10b981"} strokeWidth={1.4} rx={2} />
            <text x={PAD_L + Math.max(wOf(row.blip2), 4) + 6} y={y + 31} fontSize={9} fontWeight={700} fill={isNegative ? "#be185d" : "#065f46"}>
              {row.blip2}({row.delta > 0 ? "+" : ""}{row.delta})
            </text>
          </g>
        );
      })}
    </svg>
  );
}
