import { DISTILL_TABLE } from "../lib/data";

const W = 700;
const H = 260;

export function DistillTableChart() {
  const PAD_L = 150;
  const PAD_T = 40;
  const plotW = 480;
  const rowH = 40;
  const maxVal = 100;
  const wOf = (v: number) => (v / maxVal) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="R1 蒸馏模型 AIME/MATH-500 表现">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        R1-Distill — 32B 蒸馏模型接近 o1(单卡 A100 可跑)
      </text>

      <g transform={`translate(${PAD_L}, 30)`}>
        <rect x={0} y={0} width={10} height={10} fill="#dbeafe" stroke="#3b82f6" />
        <text x={16} y={9} fontSize={9} fill="#374151">AIME</text>
        <rect x={70} y={0} width={10} height={10} fill="#ecfdf5" stroke="#10b981" />
        <text x={86} y={9} fontSize={9} fill="#374151">MATH-500</text>
      </g>

      {DISTILL_TABLE.map((row, i) => {
        const y = PAD_T + i * rowH;
        return (
          <g key={row.model}>
            <text x={PAD_L - 10} y={y + 12} textAnchor="end" fontSize={10} fontWeight={700} fill="#374151">{row.model}</text>

            <rect x={PAD_L} y={y} width={wOf(row.aime)} height={12} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + wOf(row.aime) + 5} y={y + 10} fontSize={9} fontWeight={700} fill="#1e40af">{row.aime}</text>

            <rect x={PAD_L} y={y + 15} width={wOf(row.math500)} height={12} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + wOf(row.math500) + 5} y={y + 25} fontSize={9} fontWeight={700} fill="#065f46">{row.math500}</text>
          </g>
        );
      })}
    </svg>
  );
}
