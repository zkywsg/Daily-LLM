import { BENCHMARK_TABLE } from "../lib/data";

const W = 700;
const H = 320;

export function SpeedupBars() {
  const PAD_L = 90;
  const PAD_R = 40;
  const PAD_T = 50;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 46;

  // log 尺度,最大约 25ms
  const maxLog = Math.log10(30);
  const wOf = (ms: number) => (Math.log10(Math.max(ms, 0.2)) / maxLog) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="朴素 attention vs FlashAttention 延迟对比(log 尺度)">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        延迟对比(A100 fp16,log 尺度)— N 越长优势越大
      </text>

      <g transform={`translate(${PAD_L}, 34)`}>
        <rect x={0} y={0} width={12} height={12} fill="#f3f4f6" stroke="#9ca3af" />
        <text x={18} y={10} fontSize={10} fill="#374151">朴素 PyTorch</text>
        <rect x={130} y={0} width={12} height={12} fill="#ecfdf5" stroke="#10b981" />
        <text x={148} y={10} fontSize={10} fill="#374151">FlashAttention</text>
      </g>

      {BENCHMARK_TABLE.map((row, i) => {
        const y = PAD_T + i * (rowH + 4);
        return (
          <g key={row.seqLen}>
            <text x={PAD_L - 8} y={y + rowH / 2 + 4} textAnchor="end" fontSize={11} fontWeight={700} fill="#374151">N={row.seqLen}</text>

            {row.naiveMs === null ? (
              <>
                <rect x={PAD_L} y={y} width={40} height={16} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.2} rx={2} />
                <text x={PAD_L + 48} y={y + 13} fontSize={9} fontWeight={700} fill="#be185d">OOM</text>
              </>
            ) : (
              <>
                <rect x={PAD_L} y={y} width={wOf(row.naiveMs)} height={16} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1.2} rx={2} />
                <text x={PAD_L + wOf(row.naiveMs) + 6} y={y + 13} fontSize={9} fill="#6b7280">{row.naiveMs}ms</text>
              </>
            )}

            <rect x={PAD_L} y={y + 18} width={wOf(row.flashMs)} height={16} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.4} rx={2} />
            <text x={PAD_L + wOf(row.flashMs) + 6} y={y + 31} fontSize={9} fontWeight={700} fill="#065f46">{row.flashMs}ms</text>
          </g>
        );
      })}
    </svg>
  );
}
