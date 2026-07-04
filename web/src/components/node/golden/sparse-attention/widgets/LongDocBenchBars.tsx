import { LONG_DOC_BENCH } from "../lib/data";

const W = 700;
const H = 260;

export function LongDocBenchBars() {
  const PAD_L = 110;
  const PAD_R = 40;
  const PAD_T = 50;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 44;

  const maxVal = 100;
  const wOf = (v: number) => (v / maxVal) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Longformer vs RoBERTa long document benchmarks">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Longformer(4K)vs RoBERTa(512)长文档任务
      </text>

      <g transform={`translate(${PAD_L}, 34)`}>
        <rect x={0} y={0} width={12} height={12} fill="#f3f4f6" stroke="#9ca3af" />
        <text x={18} y={10} fontSize={10} fill="#374151">RoBERTa-512</text>
        <rect x={130} y={0} width={12} height={12} fill="#fce7f3" stroke="#ec4899" />
        <text x={148} y={10} fontSize={10} fill="#374151">Longformer-4096</text>
      </g>

      {LONG_DOC_BENCH.map((row, i) => {
        const y = PAD_T + i * (rowH + 6);
        return (
          <g key={row.task}>
            <text x={PAD_L - 8} y={y + rowH / 2 + 4} textAnchor="end" fontSize={11} fontWeight={700} fill="#374151">{row.task}</text>

            <rect x={PAD_L} y={y} width={wOf(row.short)} height={16} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + wOf(row.short) + 6} y={y + 13} fontSize={9} fill="#6b7280">{row.short.toFixed(1)}</text>

            <rect x={PAD_L} y={y + 20} width={wOf(row.long)} height={16} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.4} rx={2} />
            <text x={PAD_L + wOf(row.long) + 6} y={y + 33} fontSize={9} fontWeight={700} fill="#ec4899">{row.long.toFixed(1)}</text>

            <text x={W - 8} y={y + rowH / 2 + 4} textAnchor="end" fontSize={10} fontWeight={700} fill="#065f46">
              +{(row.long - row.short).toFixed(1)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
