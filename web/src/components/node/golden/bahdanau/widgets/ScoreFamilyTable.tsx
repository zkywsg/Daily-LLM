import { SCORE_TYPES } from "../lib/data";

const W = 700;
const H = 280;

// 4 种 score function 对比表(Bahdanau additive → Luong dot/general → Transformer scaled)
export function ScoreFamilyTable() {
  const PAD = 30;
  const rowH = 46;
  const startY = 60;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Attention score function family">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Attention Score 函数演化 — 从 Bahdanau 到 Transformer
      </text>

      {/* 表头 */}
      <text x={PAD + 20} y={50} fontSize={10} fontWeight={700} fill="#6b7280" style={{ textTransform: "uppercase" }}>方法</text>
      <text x={PAD + 180} y={50} fontSize={10} fontWeight={700} fill="#6b7280" style={{ textTransform: "uppercase" }}>打分公式</text>
      <text x={PAD + 420} y={50} fontSize={10} fontWeight={700} fill="#6b7280" style={{ textTransform: "uppercase" }}>开销</text>
      <text x={PAD + 510} y={50} fontSize={10} fontWeight={700} fill="#6b7280" style={{ textTransform: "uppercase" }}>备注</text>

      {SCORE_TYPES.map((s, i) => {
        const y = startY + i * rowH;
        return (
          <g key={i}>
            <rect x={PAD} y={y} width={W - PAD * 2} height={rowH - 4}
                  fill={s.color} fillOpacity={0.08} stroke={s.color} strokeOpacity={0.4} strokeWidth={1.2} rx={4} />
            <circle cx={PAD + 14} cy={y + rowH / 2} r={5} fill={s.color} />
            <text x={PAD + 26} y={y + rowH / 2 - 2} fontSize={11} fontWeight={700} fill={s.color}>{s.name}</text>
            <text x={PAD + 26} y={y + rowH / 2 + 12} fontSize={9} fill="#6b7280">{s.cost}</text>
            <text x={PAD + 180} y={y + rowH / 2 + 4} fontSize={11} fontFamily="ui-monospace, monospace" fill="#1f2937">{s.formula}</text>
            <text x={PAD + 420} y={y + rowH / 2 + 4} fontSize={10} fill="#374151">{s.cost.replace("MLP 一层", "慢").replace("1 次", "快")}</text>
            <text x={PAD + 510} y={y + rowH / 2 + 4} fontSize={10} fill="#374151">{s.note}</text>
          </g>
        );
      })}
    </svg>
  );
}
