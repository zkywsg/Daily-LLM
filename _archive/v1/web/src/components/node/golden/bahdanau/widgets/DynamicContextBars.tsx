import { ALIGNMENT_DEMO, buildAlignmentMatrix } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  tStep: number;
}

const M = buildAlignmentMatrix();

export function DynamicContextBars({ tStep }: Props) {
  const { src, tgt } = ALIGNMENT_DEMO;
  const alpha = M[tStep];

  const PAD_L = 30;
  const PAD_R = 30;
  const PAD_T = 80;
  const PAD_B = 60;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const barW = plotW / src.length - 6;

  const maxA = Math.max(...alpha);
  const yOf = (a: number) => PAD_T + (1 - a / Math.max(maxA, 0.01)) * plotH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Dynamic context vector per step">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        译第 {tStep + 1} 个目标词 = "{tgt[tStep]}" — α_{tStep + 1} 在源序列上的分布
      </text>
      <text x={W / 2} y={42} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        c_{tStep + 1} = Σ α_{`{${tStep + 1},i}`} · h_i · 这一步动态聚焦"{src[alpha.indexOf(Math.max(...alpha))]}"
      </text>

      {/* baseline */}
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {src.map((s, i) => {
        const x = PAD_L + 3 + i * (barW + 6);
        const a = alpha[i];
        const h = (PAD_T + plotH) - yOf(a);
        const isMax = a === maxA;
        return (
          <g key={i}>
            <rect x={x} y={yOf(a)} width={barW} height={h}
                  fill={isMax ? "#ec4899" : "#fce7f3"}
                  stroke={isMax ? "#831843" : "#ec4899"}
                  strokeWidth={isMax ? 1.6 : 1} rx={2} />
            {a > 0.05 && (
              <text x={x + barW / 2} y={yOf(a) - 4} textAnchor="middle"
                    fontSize={9} fontWeight={isMax ? 700 : 600}
                    fill={isMax ? "#831843" : "#6b7280"}>
                {a.toFixed(2)}
              </text>
            )}
            <text x={x + barW / 2} y={PAD_T + plotH + 14} textAnchor="middle"
                  fontSize={9} fontWeight={isMax ? 700 : 500}
                  fill={isMax ? "#831843" : "#374151"}>
              {s}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        α 通常很稀疏 — 80% 权重集中在 2-3 个源词上 · 这就是 attention 可解释性的起点
      </text>
    </svg>
  );
}
