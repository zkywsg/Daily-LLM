import { QA_EXAMPLES } from "../lib/data";

interface Props {
  exampleIdx: number;
  visibleSteps: number;
}

const W = 700;
const H = 360;

// 逐步点亮:viewer 拖 step slider 看模型"想"的过程,一步一步显式写下来。
// 已显示的步骤亮粉色 + 实线,未显示的灰色 + 虚线,模拟"模型一步一步推下来"。
export function StepLighting({ exampleIdx, visibleSteps }: Props) {
  const ex = QA_EXAMPLES[exampleIdx];
  const total = ex.cotSteps.length;
  const v = Math.max(0, Math.min(total, visibleSteps));
  const showAnswer = v >= total;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="CoT step-by-step lighting demo">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        CoT 一步一步\"想\" — 拖 slider 看推理逐条点亮
      </text>

      {/* Question 区 */}
      <rect x={20} y={36} width={W - 40} height={48} rx={5} fill="#fef3c7" stroke="#f59e0b" />
      <text x={32} y={56} fontSize={10} fontWeight={600} fill="#92400e">Q</text>
      <text x={60} y={62} fontSize={11} fill="#1f2937">
        {ex.question.length > 80 ? ex.question.slice(0, 80) + "…" : ex.question}
      </text>

      {/* Reasoning chain */}
      {ex.cotSteps.map((step, i) => {
        const y = 110 + i * 60;
        const visible = i < v;
        return (
          <g key={i}>
            {/* 连接线(到下一步) */}
            {i < total - 1 && (
              <line
                x1={45}
                y1={y + 30}
                x2={45}
                y2={y + 60}
                stroke={i + 1 < v ? "#ec4899" : "#d1d5db"}
                strokeWidth={2}
                strokeDasharray={i + 1 < v ? "none" : "3 3"}
              />
            )}
            {/* 步骤圆点 */}
            <circle
              cx={45}
              cy={y + 20}
              r={14}
              fill={visible ? "#fce7f3" : "var(--bg-surface)"}
              stroke={visible ? "#ec4899" : "#d1d5db"}
              strokeWidth={visible ? 2 : 1}
            />
            <text x={45} y={y + 24} textAnchor="middle" fontSize={11} fontWeight={700} fill={visible ? "#831843" : "#9ca3af"}>
              {i + 1}
            </text>
            {/* 步骤文字 */}
            <rect
              x={70}
              y={y + 6}
              width={W - 90}
              height={28}
              rx={4}
              fill={visible ? "var(--bg-surface)" : "transparent"}
              stroke={visible ? "var(--border)" : "transparent"}
              strokeWidth={1}
            />
            <text
              x={82}
              y={y + 24}
              fontSize={11}
              fill={visible ? "var(--ink-primary)" : "#9ca3af"}
              fontFamily="ui-monospace, monospace"
            >
              {step.length > 64 ? step.slice(0, 64) + "…" : step}
            </text>
          </g>
        );
      })}

      {/* Final answer */}
      <rect
        x={20}
        y={H - 50}
        width={W - 40}
        height={36}
        rx={5}
        fill={showAnswer ? "#ecfdf5" : "var(--bg-surface)"}
        stroke={showAnswer ? "#10b981" : "var(--border)"}
        strokeWidth={1.5}
      />
      <text x={32} y={H - 28} fontSize={10} fontWeight={600} fill={showAnswer ? "#065f46" : "#9ca3af"}>
        A
      </text>
      <text x={60} y={H - 25} fontSize={13} fontWeight={700} fill={showAnswer ? "#065f46" : "#9ca3af"}>
        {showAnswer ? `${ex.cotAnswer}  ✓` : "(还没推完)"}
      </text>
    </svg>
  );
}
