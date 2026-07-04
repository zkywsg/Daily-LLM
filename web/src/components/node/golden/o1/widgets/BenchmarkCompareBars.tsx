import { BENCHMARK_COMPARE } from "../lib/data";

const W = 700;
const H = 380;

export function BenchmarkCompareBars() {
  const PAD_L = 110;
  const PAD_R = 40;
  const PAD_T = 50;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 70;

  const maxVal = 100;
  const wOf = (v: number) => (v / maxVal) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="GPT-4o vs o1-preview vs o1 vs 人类专家 benchmark 对比">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Reasoning Benchmark — o1 首次在 GPQA 上超越人类专家
      </text>

      <g transform={`translate(${PAD_L}, 34)`}>
        <rect x={0} y={0} width={12} height={12} fill="#f3f4f6" stroke="#9ca3af" />
        <text x={18} y={10} fontSize={9} fill="#374151">GPT-4o</text>
        <rect x={80} y={0} width={12} height={12} fill="#dbeafe" stroke="#3b82f6" />
        <text x={98} y={10} fontSize={9} fill="#374151">o1-preview</text>
        <rect x={185} y={0} width={12} height={12} fill="#ecfdf5" stroke="#10b981" />
        <text x={203} y={10} fontSize={9} fill="#374151">o1</text>
        <rect x={235} y={0} width={12} height={12} fill="none" stroke="#a855f7" strokeDasharray="2 2" />
        <text x={253} y={10} fontSize={9} fill="#374151">人类专家</text>
      </g>

      {BENCHMARK_COMPARE.map((row, i) => {
        const y = PAD_T + i * rowH;
        return (
          <g key={row.benchmark}>
            <text x={PAD_L - 8} y={y + rowH / 2 - 8} textAnchor="end" fontSize={11} fontWeight={700} fill="#374151">{row.benchmark}</text>

            <rect x={PAD_L} y={y} width={wOf(row.gpt4o)} height={14} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1} rx={2} />
            <text x={PAD_L + wOf(row.gpt4o) + 6} y={y + 11} fontSize={8} fill="#6b7280">{row.gpt4o}</text>

            <rect x={PAD_L} y={y + 16} width={wOf(row.o1preview)} height={14} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + wOf(row.o1preview) + 6} y={y + 27} fontSize={8} fill="#1e40af">{row.o1preview}</text>

            <rect x={PAD_L} y={y + 32} width={wOf(row.o1)} height={14} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.4} rx={2} />
            <text x={PAD_L + wOf(row.o1) + 6} y={y + 43} fontSize={8} fontWeight={700} fill="#065f46">{row.o1}</text>

            <line x1={PAD_L + wOf(row.human)} y1={y - 2} x2={PAD_L + wOf(row.human)} y2={y + 48} stroke="#a855f7" strokeWidth={1.6} strokeDasharray="3 2" />
          </g>
        );
      })}
    </svg>
  );
}
