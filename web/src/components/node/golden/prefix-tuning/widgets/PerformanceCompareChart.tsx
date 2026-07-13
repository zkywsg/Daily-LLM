import { PERFORMANCE_COMPARE } from "../lib/data";

const W = 700;
const H = 260;

export function PerformanceCompareChart() {
  const PAD_L = 150;
  const PAD_R = 60;
  const PAD_T = 40;
  const plotW = W - PAD_L - PAD_R;
  const rowH = 50;

  const maxVal = 80;
  const wOf = (v: number) => (v / maxVal) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="E2E NLG BLEU:Full FT vs Adapter vs Prefix Tuning">
      <text x={W / 2} y={18} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--ink-primary)">
        E2E NLG BLEU:Full FT / Adapter / Prefix Tuning
      </text>

      {PERFORMANCE_COMPARE.map((row, i) => {
        const y = PAD_T + i * rowH;
        const barW = Math.max(wOf(row.e2eBleu), 6);
        return (
          <g key={`${row.method}-${i}`}>
            <text x={PAD_L - 10} y={y + 16} textAnchor="end" fontSize={11} fontWeight={700} fill={row.color}>
              {row.method}
            </text>
            <rect x={PAD_L} y={y} width={barW} height={24} fill={row.bg} stroke={row.color} strokeWidth={1.6} rx={4} />
            <text x={PAD_L + barW + 8} y={y + 17} fontSize={12} fontWeight={700} fill={row.color}>
              {row.e2eBleu}
            </text>
            <text x={PAD_L} y={y + 38} fontSize={9} fill="var(--ink-muted)">
              训练参数 {row.trainedParamsPct}% · WebNLG {row.webNlgBleu}
            </text>
          </g>
        );
      })}

      <text x={PAD_L} y={H - 6} fontSize={9} fill="var(--ink-muted)" fontStyle="italic">
        ↑ 同 0.1% 参数下 Prefix Tuning &gt; Adapter,甚至超过 Full FT(100% 参数)
      </text>
    </svg>
  );
}
