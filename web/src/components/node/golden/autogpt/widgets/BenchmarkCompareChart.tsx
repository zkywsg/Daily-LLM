import { BENCHMARK_TABLE } from "../lib/data";

const W = 700;
const H = 320;

export function BenchmarkCompareChart() {
  const PAD_L = 150;
  const PAD_T = 50;
  const plotW = 420;
  const rowH = 50;
  const maxVal = 92;
  const wOf = (v: number) => (v / maxVal) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="AutoGPT-style vs GPT-4+ReAct vs Human agent benchmark 对比">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        自动分解 + 反思让长任务表现全面超过单步 ReAct
      </text>

      <g transform={`translate(${PAD_L}, 36)`}>
        <rect x={0} y={0} width={10} height={10} fill="#f3f4f6" stroke="#9ca3af" />
        <text x={16} y={9} fontSize={8} fill="#374151">GPT-4 + ReAct</text>
        <rect x={110} y={0} width={10} height={10} fill="#ecfdf5" stroke="#10b981" />
        <text x={126} y={9} fontSize={8} fill="#374151">AutoGPT-style</text>
        <rect x={230} y={0} width={10} height={10} fill="none" stroke="#a855f7" strokeDasharray="2 2" />
        <text x={246} y={9} fontSize={8} fill="#374151">Human</text>
      </g>

      {BENCHMARK_TABLE.map((row, i) => {
        const y = PAD_T + i * rowH;
        return (
          <g key={row.benchmark}>
            <text x={PAD_L - 10} y={y + 18} textAnchor="end" fontSize={10} fontWeight={700} fill="#374151">{row.benchmark}</text>

            <rect x={PAD_L} y={y} width={Math.max(wOf(row.gpt4React), 4)} height={14} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + Math.max(wOf(row.gpt4React), 4) + 6} y={y + 11} fontSize={9} fill="#6b7280">{row.gpt4React}</text>

            <rect x={PAD_L} y={y + 16} width={Math.max(wOf(row.autogptStyle), 4)} height={14} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.4} rx={2} />
            <text x={PAD_L + Math.max(wOf(row.autogptStyle), 4) + 6} y={y + 27} fontSize={9} fontWeight={700} fill="#065f46">{row.autogptStyle}</text>

            {row.human !== null && (
              <line x1={PAD_L + wOf(row.human)} y1={y - 2} x2={PAD_L + wOf(row.human)} y2={y + 32} stroke="#a855f7" strokeWidth={1.6} strokeDasharray="3 2" />
            )}
          </g>
        );
      })}
    </svg>
  );
}
