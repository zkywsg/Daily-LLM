import { TEACHER_LOGITS, softmaxWithTemperature } from "../lib/data";

const W = 700;
const H = 360;

interface Props {
  temperature: number;
}

export function TeacherStudentDistillationDiagram({ temperature }: Props) {
  const logits = TEACHER_LOGITS.map((t) => t.logit);
  const probs = softmaxWithTemperature(logits, temperature);
  const barMaxW = 160;
  const barX = 380;
  const barTop = 90;
  const barGap = 46;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Teacher-Student 知识蒸馏示意,温度缩放软标签">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Teacher(12 层)软标签 → Student(6 层)模仿,温度 τ = {temperature.toFixed(1)}
      </text>

      {/* Teacher block */}
      <g transform="translate(30, 60)">
        <rect x={0} y={0} width={110} height={200} rx={8} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.6} />
        <text x={55} y={-10} textAnchor="middle" fontSize={11} fontWeight={700} fill="#3b82f6">Teacher</text>
        {Array.from({ length: 6 }, (_, i) => (
          <rect key={i} x={12} y={12 + i * 30} width={86} height={22} rx={4} fill="#bfdbfe" stroke="#3b82f6" strokeWidth={1} />
        ))}
        <text x={55} y={220} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">12 层(冻结)</text>
      </g>

      {/* Arrow */}
      <g stroke="#9ca3af" strokeWidth={1.6}>
        <line x1={150} y1={140} x2={200} y2={140} markerEnd="url(#arrow)" />
      </g>
      <text x={175} y={130} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">KL</text>

      {/* Student block */}
      <g transform="translate(210, 90)">
        <rect x={0} y={0} width={110} height={140} rx={8} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.6} />
        <text x={55} y={-10} textAnchor="middle" fontSize={11} fontWeight={700} fill="#ec4899">Student</text>
        {Array.from({ length: 3 }, (_, i) => (
          <rect key={i} x={12} y={12 + i * 40} width={86} height={30} rx={4} fill="#fbcfe8" stroke="#ec4899" strokeWidth={1} />
        ))}
        <text x={55} y={160} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">6 层(训练中)</text>
      </g>

      <line x1={W - 5} y1={5} x2={W - 5} y2={5} />
      <defs>
        <marker id="arrow" markerWidth={8} markerHeight={8} refX={6} refY={3} orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#9ca3af" />
        </marker>
      </defs>

      {/* Softmax bars */}
      <text x={barX} y={70} fontSize={11} fontWeight={700} fill="var(--ink-primary)">
        软化后的概率分布 p^(τ)
      </text>
      {TEACHER_LOGITS.map((t, i) => {
        const p = probs[i];
        const y = barTop + i * barGap;
        const w = Math.max(p * barMaxW, 2);
        return (
          <g key={t.label}>
            <text x={barX} y={y + 15} fontSize={10} fill="var(--ink-secondary)">{t.label}</text>
            <rect x={barX + 36} y={y} width={barMaxW} height={20} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1} rx={3} />
            <rect x={barX + 36} y={y} width={w} height={20} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.4} rx={3} />
            <text x={barX + 36 + barMaxW + 8} y={y + 15} fontSize={10} fontWeight={700} fill="var(--ink-primary)">
              {(p * 100).toFixed(1)}%
            </text>
          </g>
        );
      })}
      <text x={barX} y={barTop + 3 * barGap + 16} fontSize={9} fill="var(--ink-muted)">
        τ 越大分布越平滑,类别间相对关系(软信息)越明显
      </text>
    </svg>
  );
}
