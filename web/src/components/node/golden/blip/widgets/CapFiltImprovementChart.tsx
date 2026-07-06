import { CAPFILT_IMPROVEMENT, DATA_SCALE } from "../lib/data";

const W = 700;
const H = 240;

export function CapFiltImprovementChart() {
  const PAD_L = 140;
  const PAD_T = 40;
  const plotW = 420;
  const rowH = 50;
  const maxVal = 140;
  const wOf = (v: number) => (v / maxVal) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="CapFilt 数据质量提升前后对比">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        数据从 {DATA_SCALE.beforeM}M 扩到 {DATA_SCALE.afterM}M — 质量同步提升
      </text>

      <g transform={`translate(${PAD_L}, 34)`}>
        <rect x={0} y={0} width={10} height={10} fill="#f3f4f6" stroke="#9ca3af" />
        <text x={16} y={9} fontSize={9} fill="#374151">CapFilt 前(14M noisy)</text>
        <rect x={160} y={0} width={10} height={10} fill="#ecfdf5" stroke="#10b981" />
        <text x={176} y={9} fontSize={9} fill="#374151">CapFilt 后(130M cleaner)</text>
      </g>

      {CAPFILT_IMPROVEMENT.map((row, i) => {
        const y = PAD_T + i * rowH;
        return (
          <g key={row.metric}>
            <text x={PAD_L - 10} y={y + 18} textAnchor="end" fontSize={11} fontWeight={700} fill="#374151">{row.metric}</text>

            <rect x={PAD_L} y={y} width={Math.max(wOf(row.before), 4)} height={16} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + Math.max(wOf(row.before), 4) + 6} y={y + 13} fontSize={9} fill="#6b7280">{row.before}</text>

            <rect x={PAD_L} y={y + 18} width={Math.max(wOf(row.after), 4)} height={16} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.4} rx={2} />
            <text x={PAD_L + Math.max(wOf(row.after), 4) + 6} y={y + 31} fontSize={9} fontWeight={700} fill="#065f46">
              {row.after}(+{(row.after - row.before).toFixed(1)})
            </text>
          </g>
        );
      })}
    </svg>
  );
}
