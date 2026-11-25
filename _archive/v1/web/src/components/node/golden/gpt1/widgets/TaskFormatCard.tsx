import { TASK_FORMATS } from "../lib/data";

const W = 700;
const H = 300;

interface Props {
  taskIdx: number;
}

export function TaskFormatCard({ taskIdx }: Props) {
  const t = TASK_FORMATS[taskIdx];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="GPT-1 unified task format">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        统一任务接口 — {t.name}
      </text>

      {/* format */}
      <rect x={30} y={50} width={W - 60} height={60} rx={6} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.5} />
      <text x={42} y={70} fontSize={10} fontWeight={700} fill="#92400e" style={{ textTransform: "uppercase" }}>输入格式</text>
      <text x={42} y={92} fontSize={13} fontFamily="ui-monospace, monospace" fill="#1f2937">{t.format}</text>

      {/* example */}
      <rect x={30} y={130} width={W - 60} height={70} rx={6} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.5} />
      <text x={42} y={150} fontSize={10} fontWeight={700} fill="#1e40af" style={{ textTransform: "uppercase" }}>具体例子</text>
      <text x={42} y={172} fontSize={12} fontFamily="ui-monospace, monospace" fill="#1f2937">{t.example}</text>

      {/* forward count */}
      <rect x={30} y={220} width={W - 60} height={50} rx={6} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.5} />
      <text x={42} y={240} fontSize={10} fontWeight={700} fill="#065f46" style={{ textTransform: "uppercase" }}>Forward 次数</text>
      <text x={42} y={260} fontSize={13} fontWeight={700} fill="#065f46">{t.forwardCount}</text>

      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        模型本身没有任何任务特定模块,微调时只多一个 linear head
      </text>
    </svg>
  );
}
