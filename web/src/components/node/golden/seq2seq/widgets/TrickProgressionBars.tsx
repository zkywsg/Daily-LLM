import { TRICK_PROGRESSION, SMT_BASELINE } from "../lib/data";

const W = 700;
const H = 300;

interface Props {
  highlightIdx: number;
}

export function TrickProgressionBars({ highlightIdx }: Props) {
  const PAD_L = 40;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 70;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const n = TRICK_PROGRESSION.length;
  const barW = plotW / n - 20;

  const yMin = 25, yMax = 38;
  const yOf = (v: number) => PAD_T + ((yMax - v) / (yMax - yMin)) * plotH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Sutskever trick progression BLEU">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Sutskever 三件工程 trick 累加 BLEU
      </text>

      {/* SMT baseline line */}
      <line x1={PAD_L} y1={yOf(SMT_BASELINE)} x2={W - PAD_R} y2={yOf(SMT_BASELINE)}
            stroke="#9ca3af" strokeWidth={1.5} strokeDasharray="4 4" />
      <text x={W - PAD_R} y={yOf(SMT_BASELINE) - 6} textAnchor="end" fontSize={10} fontWeight={600} fill="#6b7280">
        SMT baseline {SMT_BASELINE}
      </text>

      {TRICK_PROGRESSION.map((t, i) => {
        const x = PAD_L + 10 + i * (barW + 20);
        const isHigh = i === highlightIdx || highlightIdx === -1;
        const isPassSmt = t.bleu > SMT_BASELINE;
        const color = isPassSmt ? "#10b981" : "#ec4899";
        const bg = isPassSmt ? "#ecfdf5" : "#fce7f3";
        return (
          <g key={i} opacity={isHigh ? 1 : 0.4}>
            <rect x={x} y={yOf(t.bleu)} width={barW} height={(PAD_T + plotH) - yOf(t.bleu)}
                  fill={bg} stroke={color} strokeWidth={1.4} rx={2} />
            <text x={x + barW / 2} y={yOf(t.bleu) - 6} textAnchor="middle" fontSize={11} fontWeight={700} fill={color}>
              {t.bleu.toFixed(1)}
            </text>
            {t.delta > 0 && (
              <text x={x + barW / 2} y={yOf(t.bleu) + 16} textAnchor="middle" fontSize={9} fill="#374151">
                +{t.delta.toFixed(1)}
              </text>
            )}
            <text x={x + barW / 2} y={PAD_T + plotH + 16} textAnchor="middle" fontSize={9} fontWeight={500} fill="#374151">
              {t.label.split("(")[0]}
            </text>
            {t.label.includes("(") && (
              <text x={x + barW / 2} y={PAD_T + plotH + 30} textAnchor="middle" fontSize={8} fill="#9ca3af">
                {"(" + t.label.split("(")[1]}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
