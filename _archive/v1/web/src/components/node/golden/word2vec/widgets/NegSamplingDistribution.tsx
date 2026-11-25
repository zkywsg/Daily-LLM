import { negSamplingDist, WORD_FREQS } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  alpha: number;
}

// 显示在不同 alpha 下,采样分布如何调整 — uniform/unigram/^0.75 三种.
export function NegSamplingDistribution({ alpha }: Props) {
  const dist = negSamplingDist(alpha);
  const distSorted = [...dist].sort((a, b) => b.prob - a.prob).slice(0, 12);

  const baseY = 240;
  const maxBar = 180;
  const maxProb = Math.max(...distSorted.map((d) => d.prob));
  const barW = 44;
  const gap = 6;
  const startX = (W - (distSorted.length * (barW + gap) - gap)) / 2;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Negative sampling distribution by alpha">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        负采样分布 — P_n(w) ∝ count(w)^{alpha.toFixed(2)}
      </text>
      <text x={W / 2} y={42} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        α=1 偏向高频(单调 unigram);α=0 退化 uniform;α=0.75 是 Mikolov 实测最佳
      </text>

      {/* baseline */}
      <line x1={startX - 6} y1={baseY} x2={startX + distSorted.length * (barW + gap)} y2={baseY} stroke="#9ca3af" strokeWidth={1} />

      {distSorted.map((d, i) => {
        const h = (d.prob / maxProb) * maxBar;
        const x = startX + i * (barW + gap);
        const wf = WORD_FREQS.find((w) => w.word === d.word);
        const isStop = wf?.category === "stopword";
        const isRare = wf?.category === "rare";
        const fill = isStop ? "#fce7f3" : isRare ? "#dbeafe" : "#fef3c7";
        const stroke = isStop ? "#ec4899" : isRare ? "#3b82f6" : "#f59e0b";
        return (
          <g key={d.word}>
            <rect x={x} y={baseY - h} width={barW} height={h} fill={fill} stroke={stroke} strokeWidth={1.2} rx={2} />
            <text x={x + barW / 2} y={baseY - h - 4} textAnchor="middle" fontSize={9} fill="#6b7280">
              {(d.prob * 100).toFixed(1)}%
            </text>
            <text x={x + barW / 2} y={baseY + 14} textAnchor="middle" fontSize={10} fontWeight={500} fill="#374151">
              {d.word}
            </text>
          </g>
        );
      })}

      {/* legend */}
      <g transform={`translate(${W / 2 - 180}, ${H - 26})`}>
        <rect x={0} y={0} width={12} height={12} fill="#fce7f3" stroke="#ec4899" />
        <text x={18} y={10} fontSize={10} fill="#374151">stopword</text>
        <rect x={92} y={0} width={12} height={12} fill="#fef3c7" stroke="#f59e0b" />
        <text x={110} y={10} fontSize={10} fill="#374151">common</text>
        <rect x={184} y={0} width={12} height={12} fill="#dbeafe" stroke="#3b82f6" />
        <text x={202} y={10} fontSize={10} fill="#374151">rare</text>
      </g>
    </svg>
  );
}
