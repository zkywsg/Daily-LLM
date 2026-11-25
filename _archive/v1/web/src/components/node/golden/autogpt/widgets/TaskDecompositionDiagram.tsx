import { GOAL_EXAMPLE, DECOMPOSED_TASKS } from "../lib/data";

const W = 700;

interface Props {
  revealedCount: number;
}

export function TaskDecompositionDiagram({ revealedCount }: Props) {
  const H = 100 + DECOMPOSED_TASKS.length * 34 + 40;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="高级目标自动分解为子任务队列">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        模糊高级目标 → LLM 自动拆成子任务队列
      </text>

      <foreignObject x={30} y={35} width={640} height={50}>
        <div style={{ fontSize: 10, color: "#92400e", background: "#fef3c7", border: "1px solid #f59e0b", borderRadius: 6, padding: "6px 10px", lineHeight: 1.4 }}>
          {GOAL_EXAMPLE}
        </div>
      </foreignObject>

      <line x1={350} y1={92} x2={350} y2={110} stroke="#9ca3af" strokeWidth={1.4} markerEnd="url(#arrow-decomp)" />

      {DECOMPOSED_TASKS.map((task, i) => {
        const y = 118 + i * 34;
        const isRevealed = i < revealedCount;
        return (
          <g key={task} opacity={isRevealed ? 1 : 0.25}>
            <rect x={30} y={y} width={640} height={26} fill={isRevealed ? "#ecfdf5" : "var(--bg-surface)"} stroke={isRevealed ? "#10b981" : "var(--border)"} strokeWidth={isRevealed ? 1.6 : 1} rx={4} />
            <text x={44} y={y + 17} fontSize={10} fontFamily="monospace" fill={isRevealed ? "#065f46" : "var(--ink-muted)"}>{i + 1}. {task}</text>
          </g>
        );
      })}
    </svg>
  );
}
