import { BENCHMARKS } from "../lib/data";

const W = 700;
const H = 380;

export function BenchmarkChart() {
  const PAD_L = 230;
  const PAD_R = 70;
  const PAD_T = 40;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 44;

  const maxVal = Math.max(...BENCHMARKS.map((b) => b.metric));
  const wOf = (v: number) => (v / maxVal) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Transformer-XL 性能数据对比">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        三个标准语言建模 benchmark 上的 perplexity / BPC 对比
      </text>

      {BENCHMARKS.map((row, i) => {
        const y = PAD_T + i * rowH;
        const color = row.highlight ? "#3b82f6" : "#9ca3af";
        const bg = row.highlight ? "#dbeafe" : "#f3f4f6";
        return (
          <g key={`${row.benchmark}-${row.model}`}>
            <text x={PAD_L - 8} y={y + 14} textAnchor="end" fontSize={10} fontWeight={700} fill={color}>
              {row.model}
            </text>
            <text x={PAD_L - 8} y={y + 26} textAnchor="end" fontSize={8} fill="var(--ink-muted)">
              {row.benchmark}
            </text>

            <rect x={PAD_L} y={y} width={Math.max(wOf(row.metric), 4)} height={20} fill={bg} stroke={color} strokeWidth={1.4} rx={3} />
            <text x={PAD_L + Math.max(wOf(row.metric), 4) + 6} y={y + 15} fontSize={10} fontWeight={700} fill={color}>
              {row.metric} {row.unit}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
