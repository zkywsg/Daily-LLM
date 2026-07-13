import { SCALING_TREND } from "../lib/data";

const W = 640;
const H = 260;

export function ScalingTrendChart() {
  const PAD_L = 60;
  const PAD_R = 40;
  const PAD_T = 40;
  const PAD_B = 40;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const minVal = 66;
  const maxVal = 72;
  const yOf = (v: number) => PAD_T + plotH - ((v - minVal) / (maxVal - minVal)) * plotH;
  const xOf = (i: number) => PAD_L + (i / (SCALING_TREND.length - 1)) * plotW;

  const prefixPoints = SCALING_TREND.map((r, i) => `${xOf(i)},${yOf(r.prefixTuning)}`).join(" ");
  const fullFtPoints = SCALING_TREND.map((r, i) => `${xOf(i)},${yOf(r.fullFt)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="GPT-2 medium/large/XL 上 Prefix Tuning 与 Full FT 的差距">
      <text x={W / 2} y={18} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--ink-primary)">
        模型越大,Prefix Tuning 领先 Full FT 的优势保持
      </text>

      <polyline points={fullFtPoints} fill="none" stroke="#9ca3af" strokeWidth={2} />
      <polyline points={prefixPoints} fill="none" stroke="#ec4899" strokeWidth={2.4} />

      {SCALING_TREND.map((r, i) => (
        <g key={`${r.model}-${i}`}>
          <circle cx={xOf(i)} cy={yOf(r.fullFt)} r={4} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1.6} />
          <circle cx={xOf(i)} cy={yOf(r.prefixTuning)} r={4.5} fill="#fce7f3" stroke="#ec4899" strokeWidth={2} />
          <text x={xOf(i)} y={yOf(r.prefixTuning) - 12} textAnchor="middle" fontSize={10} fontWeight={700} fill="#be185d">
            {r.prefixTuning}
          </text>
          <text x={xOf(i)} y={yOf(r.fullFt) + 18} textAnchor="middle" fontSize={9} fill="#6b7280">
            {r.fullFt}
          </text>
          <text x={xOf(i)} y={H - PAD_B + 18} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
            {r.model}
          </text>
          <text x={xOf(i)} y={H - PAD_B + 30} textAnchor="middle" fontSize={8} fill="var(--ink-muted)">
            ({r.params})
          </text>
        </g>
      ))}

      <text x={PAD_L} y={H - 4} fontSize={9} fill="#be185d" fontStyle="italic">
        ↑ 粉色 = Prefix Tuning,灰色 = Full FT;差距(gap)在 medium→XL 稳定在 +1.9~+2.1 BLEU
      </text>
    </svg>
  );
}
