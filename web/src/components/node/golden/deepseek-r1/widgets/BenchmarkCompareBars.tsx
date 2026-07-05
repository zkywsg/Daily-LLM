import { BENCHMARK_COMPARE } from "../lib/data";

const W = 700;
const H = 340;

export function BenchmarkCompareBars() {
  const PAD_L = 110;
  const PAD_R = 40;
  const PAD_T = 50;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 70;

  const maxVal = 100;
  const wOf = (v: number) => (v / maxVal) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="DeepSeek-V3 vs R1-Zero vs R1 vs o1-mini vs o1 benchmark 对比">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        R1 与 o1 全面持平 — 开源第一次追上闭源 reasoning 旗舰
      </text>

      <g transform={`translate(${PAD_L}, 34)`}>
        <rect x={0} y={0} width={10} height={10} fill="#f3f4f6" stroke="#9ca3af" />
        <text x={16} y={9} fontSize={8} fill="#374151">V3</text>
        <rect x={45} y={0} width={10} height={10} fill="#fef3c7" stroke="#f59e0b" />
        <text x={61} y={9} fontSize={8} fill="#374151">R1-Zero</text>
        <rect x={125} y={0} width={10} height={10} fill="#ecfdf5" stroke="#10b981" />
        <text x={141} y={9} fontSize={8} fill="#374151">R1</text>
        <rect x={170} y={0} width={10} height={10} fill="#f3f4f6" stroke="#9ca3af" strokeDasharray="2 1" />
        <text x={186} y={9} fontSize={8} fill="#374151">o1-mini</text>
        <rect x={250} y={0} width={10} height={10} fill="#dbeafe" stroke="#3b82f6" />
        <text x={266} y={9} fontSize={8} fill="#374151">o1</text>
      </g>

      {BENCHMARK_COMPARE.map((row, i) => {
        const y = PAD_T + i * rowH;
        return (
          <g key={row.benchmark}>
            <text x={PAD_L - 8} y={y + rowH / 2 - 8} textAnchor="end" fontSize={11} fontWeight={700} fill="#374151">{row.benchmark}</text>

            <rect x={PAD_L} y={y} width={wOf(row.deepseekV3)} height={12} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1} rx={2} />
            <text x={PAD_L + wOf(row.deepseekV3) + 5} y={y + 10} fontSize={8} fill="#6b7280">{row.deepseekV3}</text>

            {row.r1zero !== null && (
              <>
                <rect x={PAD_L} y={y + 14} width={wOf(row.r1zero)} height={12} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1} rx={2} />
                <text x={PAD_L + wOf(row.r1zero) + 5} y={y + 24} fontSize={8} fill="#b45309">{row.r1zero}</text>
              </>
            )}

            <rect x={PAD_L} y={y + 28} width={wOf(row.r1)} height={12} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + wOf(row.r1) + 5} y={y + 38} fontSize={8} fontWeight={700} fill="#065f46">{row.r1}</text>

            <rect x={PAD_L} y={y + 42} width={wOf(row.o1mini)} height={10} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1} strokeDasharray="2 1" rx={2} />
            <text x={PAD_L + wOf(row.o1mini) + 5} y={y + 50} fontSize={7} fill="#6b7280">{row.o1mini}</text>

            <line x1={PAD_L + wOf(row.o1)} y1={y - 2} x2={PAD_L + wOf(row.o1)} y2={y + 56} stroke="#3b82f6" strokeWidth={1.6} strokeDasharray="3 2" />
            <text x={PAD_L + wOf(row.o1) + 4} y={y + 62} fontSize={8} fill="#1e40af">o1={row.o1}</text>
          </g>
        );
      })}
    </svg>
  );
}
