import { LLM_SCALE_ICL } from "../lib/data";

const W = 700;
const H = 300;

interface Props {
  highlightIdx: number;
}

export function LlmScaleIclChart({ highlightIdx }: Props) {
  const PAD_L = 130;
  const PAD_R = 60;
  const PAD_T = 60;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 60;

  const maxVal = 65;
  const wOf = (v: number) => (v / maxVal) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="LLM scale vs ICL emergence">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        LLM 规模 vs ICL 涌现 — 70B 是关键阈值
      </text>

      <g transform={`translate(${PAD_L}, 40)`}>
        <rect x={0} y={0} width={12} height={12} fill="#f3f4f6" stroke="#9ca3af" />
        <text x={18} y={10} fontSize={10} fill="#374151">0-shot</text>
        <rect x={90} y={0} width={12} height={12} fill="#fce7f3" stroke="#ec4899" />
        <text x={108} y={10} fontSize={10} fill="#374151">4-shot</text>
      </g>

      {LLM_SCALE_ICL.map((row, i) => {
        const y = PAD_T + i * (rowH + 8);
        const isFocus = i === highlightIdx || highlightIdx === -1;
        return (
          <g key={row.name} opacity={isFocus ? 1 : 0.35}>
            <text x={PAD_L - 8} y={y + 12} textAnchor="end" fontSize={11} fontWeight={700} fill="#374151">{row.name}</text>
            <text x={PAD_L - 8} y={y + 26} textAnchor="end" fontSize={9} fill="#9ca3af">{row.params}B params</text>

            <rect x={PAD_L} y={y} width={wOf(row.zeroShot)} height={20} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1.2} rx={2} />
            <text x={PAD_L + wOf(row.zeroShot) + 6} y={y + 15} fontSize={9} fill="#6b7280">{row.zeroShot}</text>

            <rect x={PAD_L} y={y + 26} width={wOf(row.fourShot)} height={20} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.4} rx={2} />
            <text x={PAD_L + wOf(row.fourShot) + 6} y={y + 41} fontSize={9} fontWeight={700} fill="#ec4899">{row.fourShot}</text>

            {row.hasICL && (
              <text x={PAD_L + wOf(row.fourShot) + 50} y={y + 20} fontSize={10} fontWeight={700} fill="#065f46">← ICL 涌现!</text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
