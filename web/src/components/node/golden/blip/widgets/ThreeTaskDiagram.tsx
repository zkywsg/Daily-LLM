import { THREE_TASKS } from "../lib/data";

const W = 700;
const H = 280;

interface Props {
  highlightIdx: number;
}

export function ThreeTaskDiagram({ highlightIdx }: Props) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="BLIP 三任务联合预训练">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        共享 Vision Encoder + Text Encoder,三个任务头联合训练
      </text>

      <rect x={30} y={45} width={140} height={40} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.4} rx={6} />
      <text x={100} y={69} textAnchor="middle" fontSize={11} fontWeight={700} fill="#92400e">Vision Encoder(ViT)</text>

      <rect x={30} y={100} width={140} height={40} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.4} rx={6} />
      <text x={100} y={124} textAnchor="middle" fontSize={11} fontWeight={700} fill="#92400e">Text Encoder</text>

      {THREE_TASKS.map((task, i) => {
        const y = 45 + i * 75;
        const isFocus = i === highlightIdx || highlightIdx === -1;
        const color = i === 0 ? "#3b82f6" : i === 1 ? "#ec4899" : "#10b981";
        const bg = i === 0 ? "#dbeafe" : i === 1 ? "#fce7f3" : "#ecfdf5";
        return (
          <g key={task.name} opacity={isFocus ? 1 : 0.3}>
            <line x1={170} y1={65} x2={330} y2={y + 20} stroke={color} strokeWidth={1.4} />
            <line x1={170} y1={120} x2={330} y2={y + 20} stroke={color} strokeWidth={1.4} />
            <rect x={330} y={y} width={330} height={40} fill={bg} stroke={color} strokeWidth={1.6} rx={6} />
            <text x={345} y={y + 17} fontSize={11} fontWeight={700} fill={color}>{task.name}({task.full})</text>
            <text x={345} y={y + 32} fontSize={9} fill={color}>{task.desc.slice(0, 36)}...</text>
          </g>
        );
      })}
    </svg>
  );
}
