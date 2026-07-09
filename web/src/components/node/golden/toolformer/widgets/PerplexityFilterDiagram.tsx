import { FILTER_EXAMPLES } from "../lib/data";

const W = 700;
const H = 320;

export function PerplexityFilterDiagram() {
  const PAD_L = 40;
  const PAD_T = 50;
  const plotW = 480;
  const rowH = 60;
  const maxVal = 7;
  const wOf = (v: number) => (v / maxVal) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Perplexity 过滤:带/不带 API 调用的 loss 对比">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        只保留真正降低 loss 的调用,丢弃"无用装饰"
      </text>

      <g transform={`translate(${PAD_L}, 32)`}>
        <rect x={0} y={0} width={10} height={10} fill="#f3f4f6" stroke="#9ca3af" />
        <text x={16} y={9} fontSize={9} fill="#374151">L⁻(不带调用)</text>
        <rect x={110} y={0} width={10} height={10} fill="#dbeafe" stroke="#3b82f6" />
        <text x={126} y={9} fontSize={9} fill="#374151">L⁺(带调用)</text>
      </g>

      {FILTER_EXAMPLES.map((ex, i) => {
        const y = PAD_T + i * rowH;
        const color = ex.kept ? "#10b981" : "#ec4899";
        return (
          <g key={ex.call}>
            <text x={PAD_L} y={y - 4} fontSize={9} fontFamily="monospace" fill="var(--ink-secondary)">{ex.call}</text>

            <rect x={PAD_L} y={y} width={Math.max(wOf(ex.lossMinus), 4)} height={14} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + Math.max(wOf(ex.lossMinus), 4) + 6} y={y + 11} fontSize={9} fill="#6b7280">{ex.lossMinus}</text>

            <rect x={PAD_L} y={y + 16} width={Math.max(wOf(ex.lossPlus), 4)} height={14} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + Math.max(wOf(ex.lossPlus), 4) + 6} y={y + 27} fontSize={9} fill="#1e40af">{ex.lossPlus}</text>

            <text x={PAD_L + plotW + 30} y={y + 20} fontSize={11} fontWeight={700} fill={color}>
              {ex.kept ? "✓ 保留" : "✗ 丢弃"}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
