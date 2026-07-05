import { TEACHER_LOGITS, hardTarget } from "../lib/data";

const W = 700;
const H = 260;

interface Props {
  mode: "soft" | "hard";
}

// 10 类玩具分布:soft = teacher 的完整概率分布(KL 散度目标),
// hard = teacher 的 argmax one-hot(cross-entropy 目标,DeiT 采用)。
export function HardVsSoftLabelDiagram({ mode }: Props) {
  const dist = mode === "soft" ? TEACHER_LOGITS : hardTarget(TEACHER_LOGITS);
  const PAD_L = 40;
  const PAD_R = 40;
  const barW = (W - PAD_L - PAD_R) / dist.length;
  const maxH = 150;
  const baseY = 200;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Hard vs soft label distillation target">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--ink-primary)">
        {mode === "soft" ? "Soft distillation — teacher 的完整概率分布(KL 散度)" : "Hard distillation — teacher 的 argmax one-hot(cross-entropy,DeiT 采用)"}
      </text>

      {dist.map((v, i) => {
        const x = PAD_L + i * barW;
        const h = v * maxH;
        const isMax = mode === "hard" ? v === 1 : i === TEACHER_LOGITS.indexOf(Math.max(...TEACHER_LOGITS));
        return (
          <g key={i}>
            <rect
              x={x + barW * 0.15}
              y={baseY - h}
              width={barW * 0.7}
              height={h}
              rx={3}
              fill={isMax ? "#ec4899" : "#fce7f3"}
              stroke={isMax ? "#db2777" : "#f9a8d4"}
              strokeWidth={1.2}
            />
            <text x={x + barW / 2} y={baseY + 16} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
              c{i}
            </text>
            {v > 0.05 && (
              <text x={x + barW / 2} y={baseY - h - 6} textAnchor="middle" fontSize={9} fontWeight={600} fill="#9d174d">
                {v.toFixed(2)}
              </text>
            )}
          </g>
        );
      })}

      <line x1={PAD_L} y1={baseY} x2={W - PAD_R} y2={baseY} stroke="var(--border)" />

      <text x={W / 2} y={240} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        {mode === "soft"
          ? "teacher 偶尔出错时,错误的概率质量也被传给 student"
          : "即使 teacher 出错,student 只学到 \"另一个具体类别\",不传染错误的模糊分布"}
      </text>
    </svg>
  );
}
