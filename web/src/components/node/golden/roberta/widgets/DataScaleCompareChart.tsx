import { SCALE_COMPARE } from "../lib/data";

const W = 700;
const H = 260;

export function DataScaleCompareChart() {
  const PAD_L = 100;
  const PAD_T = 50;
  const plotW = 450;
  const rowH = 70;

  const metrics: Array<{ key: keyof (typeof SCALE_COMPARE)[number]; label: string; unit: string; max: number }> = [
    { key: "dataGB", label: "训练数据", unit: "GB", max: 200 },
    { key: "tokensB", label: "训练 token", unit: "B", max: 2200 },
    { key: "batch", label: "batch size", unit: "序列", max: 9000 },
  ];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="BERT vs RoBERTa 数据与算力规模对比">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        BERT vs RoBERTa — 架构不变,规模差 10-32×
      </text>

      <g transform={`translate(${PAD_L}, 34)`}>
        <rect x={0} y={0} width={10} height={10} fill="#f3f4f6" stroke="#9ca3af" />
        <text x={16} y={9} fontSize={9} fill="#374151">BERT</text>
        <rect x={70} y={0} width={10} height={10} fill="#ecfdf5" stroke="#10b981" />
        <text x={86} y={9} fontSize={9} fill="#374151">RoBERTa</text>
      </g>

      {metrics.map((m, i) => {
        const y = PAD_T + i * rowH;
        const bertVal = SCALE_COMPARE[0][m.key] as number;
        const robertaVal = SCALE_COMPARE[1][m.key] as number;
        const wOf = (v: number) => (v / m.max) * plotW;
        return (
          <g key={m.key}>
            <text x={PAD_L - 10} y={y + rowH / 2 - 10} textAnchor="end" fontSize={11} fontWeight={700} fill="#374151">{m.label}</text>

            <rect x={PAD_L} y={y} width={Math.max(wOf(bertVal), 2)} height={16} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + Math.max(wOf(bertVal), 2) + 6} y={y + 13} fontSize={9} fill="#6b7280">{bertVal}{m.unit}</text>

            <rect x={PAD_L} y={y + 20} width={Math.max(wOf(robertaVal), 2)} height={16} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.4} rx={2} />
            <text x={PAD_L + Math.max(wOf(robertaVal), 2) + 6} y={y + 33} fontSize={9} fontWeight={700} fill="#065f46">{robertaVal}{m.unit}</text>
          </g>
        );
      })}
    </svg>
  );
}
