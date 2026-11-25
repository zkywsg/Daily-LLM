import { BENCHMARK_COMPARE } from "../lib/data";

const W = 700;
const H = 340;

export function BenchmarkComparisonChart() {
  const PAD_L = 120;
  const PAD_T = 40;
  const rowH = 56;
  const plotW = 480;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="CoT 单次解码 vs Self-Consistency(N=40)在多个推理 benchmark 上的准确率对比"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        CoT(单次) vs Self-Consistency(N=40)— PaLM 540B
      </text>

      <g transform={`translate(${W - 180}, 30)`}>
        <rect x={0} y={0} width={10} height={10} fill="#f3f4f6" stroke="#9ca3af" />
        <text x={14} y={9} fontSize={9} fill="#374151">CoT 单次</text>
        <rect x={80} y={0} width={10} height={10} fill="#ecfdf5" stroke="#10b981" />
        <text x={94} y={9} fontSize={9} fill="#374151">SC(N=40)</text>
      </g>

      {BENCHMARK_COMPARE.map((row, i) => {
        const y = PAD_T + i * rowH;
        const maxVal = 100;
        const cotW = (row.cot / maxVal) * plotW;
        const scW = (row.selfConsistency / maxVal) * plotW;
        return (
          <g key={row.benchmark}>
            <text x={PAD_L - 12} y={y + 28} textAnchor="end" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
              {row.benchmark}
            </text>
            <rect x={PAD_L} y={y + 4} width={cotW} height={14} rx={3} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1} />
            <text x={PAD_L + cotW + 6} y={y + 15} fontSize={9} fill="#6b7280">{row.cot}</text>
            <rect x={PAD_L} y={y + 22} width={scW} height={14} rx={3} fill="#ecfdf5" stroke="#10b981" strokeWidth={1} />
            <text x={PAD_L + scW + 6} y={y + 33} fontSize={9} fontWeight={700} fill="#059669">
              {row.selfConsistency}(+{row.delta})
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
        GSM8K +17.9 分是 CoT 之后两年里最大的单点提升之一,且完全免费(无需重训)
      </text>
    </svg>
  );
}
