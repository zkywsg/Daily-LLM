import { SHEEP_MILK_PATHS } from "../lib/data";

const W = 700;
const H = 300;

export function MajorityVoteDiagram() {
  const counts: Record<string, number> = {};
  for (const p of SHEEP_MILK_PATHS) counts[p.answer] = (counts[p.answer] || 0) + 1;
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const maxCount = Math.max(...entries.map(([, c]) => c));
  const barW = 140;
  const gap = 60;
  const plotH = 160;
  const startX = W / 2 - (entries.length * barW + (entries.length - 1) * gap) / 2;

  const greedyAnswer = SHEEP_MILK_PATHS[0].answer;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="多数投票:统计每个答案出现次数,选出现最多的答案作为最终输出,对比单次贪婪解码"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Marginalize over r — 多数投票 tally
      </text>

      {entries.map(([answer, count], i) => {
        const x = startX + i * (barW + gap);
        const h = (count / maxCount) * plotH;
        const isWinner = count === maxCount;
        const color = isWinner ? "#10b981" : "#9ca3af";
        const fill = isWinner ? "#ecfdf5" : "#f3f4f6";
        const y = 220 - h;
        return (
          <g key={answer}>
            <rect x={x} y={y} width={barW} height={h} rx={6} fill={fill} stroke={color} strokeWidth={1.5} />
            <text x={x + barW / 2} y={y - 10} textAnchor="middle" fontSize={13} fontWeight={700} fill={color}>
              {count}/{SHEEP_MILK_PATHS.length} 票
            </text>
            <text x={x + barW / 2} y={244} textAnchor="middle" fontSize={14} fontWeight={700} fill="var(--ink-primary)">
              答案 {answer}
            </text>
            {isWinner && (
              <text x={x + barW / 2} y={262} textAnchor="middle" fontSize={10} fill="#059669">
                ✓ 多数投票胜出
              </text>
            )}
          </g>
        );
      })}

      <line x1={60} y1={220} x2={W - 60} y2={220} stroke="var(--border)" strokeWidth={1} />

      <rect x={W / 2 - 170} y={272} width={340} height={20} rx={4} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1} />
      <text x={W / 2} y={286} textAnchor="middle" fontSize={9} fill="#92400e">
        对比:单次贪婪解码只能拿到答案 {greedyAnswer}(碰运气,无法自我纠错)
      </text>
    </svg>
  );
}
