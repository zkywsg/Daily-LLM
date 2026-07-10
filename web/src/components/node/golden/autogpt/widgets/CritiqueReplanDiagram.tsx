import { CRITIQUE_EXAMPLE } from "../lib/data";

const W = 700;

interface Props {
  showAfter: boolean;
}

export function CritiqueReplanDiagram({ showAfter }: Props) {
  const list = showAfter ? CRITIQUE_EXAMPLE.after : CRITIQUE_EXAMPLE.before;
  const H = 90 + list.length * 30 + 60;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Self-Critique 反思与 task queue 重排">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {showAfter ? "反思后:LLM 插入了验证 + 引用步骤" : "反思前:原始 task queue"}
      </text>

      {list.map((task, i) => {
        const y = 45 + i * 30;
        const isNew = showAfter && !CRITIQUE_EXAMPLE.before.includes(task);
        return (
          <g key={task}>
            <rect x={30} y={y} width={640} height={24} fill={isNew ? "#fef3c7" : "#f3f4f6"} stroke={isNew ? "#f59e0b" : "#9ca3af"} strokeWidth={isNew ? 1.8 : 1} rx={4} />
            <text x={44} y={y + 16} fontSize={10} fontFamily="monospace" fill={isNew ? "#92400e" : "#374151"}>
              {task}{isNew ? "  ← 新增" : ""}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 20} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        {CRITIQUE_EXAMPLE.reason}
      </text>
    </svg>
  );
}
