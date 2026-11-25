import { PREFERENCE_EXAMPLE, rmScore } from "../lib/data";

const W = 700;
const H = 360;

// 一个 prompt → 4 个 GPT 候选 → labeler ranks them A>B>C>D。
// 显示 candidate text + rank + RM 应该给的分数。

export function PreferenceRanking() {
  const ex = PREFERENCE_EXAMPLE;
  const N = ex.candidates.length;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Preference ranking example">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Labeler 偏好排序 — 1 个 prompt 对应 N 个 candidate
      </text>

      {/* Prompt */}
      <rect x={20} y={36} width={W - 40} height={36} rx={4} fill="#fef3c7" stroke="#f59e0b" />
      <text x={32} y={58} fontSize={11} fontWeight={600} fill="#92400e">Prompt:</text>
      <text x={100} y={58} fontSize={11} fill="var(--ink-primary)">{ex.prompt}</text>

      {/* candidates 排序 */}
      {ex.candidates
        .slice()
        .sort((a, b) => a.rank - b.rank)
        .map((c, idx) => {
          const y = 90 + idx * 60;
          const score = rmScore(c.rank, N);
          const isFirst = c.rank === 1;
          const isLast = c.rank === N;
          const stroke = isFirst ? "#10b981" : isLast ? "#dc2626" : "#9ca3af";
          const fill = isFirst ? "#ecfdf5" : isLast ? "#fef2f2" : "#f3f4f6";
          return (
            <g key={c.rank}>
              {/* rank 圆点 */}
              <circle cx={32} cy={y + 22} r={14} fill={fill} stroke={stroke} strokeWidth={1.8} />
              <text x={32} y={y + 26} textAnchor="middle" fontSize={12} fontWeight={700} fill={isFirst ? "#065f46" : isLast ? "#7f1d1d" : "#374151"}>
                #{c.rank}
              </text>
              {/* candidate 框 */}
              <rect x={56} y={y + 6} width={W - 220} height={32} rx={4} fill={fill} stroke={stroke} strokeWidth={1} />
              <text x={68} y={y + 26} fontSize={10} fontFamily="ui-monospace, monospace" fill="var(--ink-primary)">
                {c.text.length > 50 ? c.text.slice(0, 50) + "…" : c.text}
              </text>
              {/* RM 分数 */}
              <text x={W - 140} y={y + 26} fontSize={11} fontWeight={600} fill={stroke}>
                RM logit: {score.toFixed(2)}
              </text>
            </g>
          );
        })}

      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        RM 不学绝对分数,只学\"排序顺序\" — 训练目标:RM(#1) {">"} RM(#2) {">"} … {">"} RM(#N)
      </text>
    </svg>
  );
}
