import { INIT_LAYER_INDICES, TEACHER_LAYERS, STUDENT_LAYERS } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  showMapping: boolean;
}

export function LayerInitializationDiagram({ showMapping }: Props) {
  const teacherX = 120;
  const studentX = 480;
  const layerH = 20;
  const gap = 6;
  const top = 60;

  const teacherLayerY = (idx: number) => top + idx * (layerH + gap);
  const studentLayerY = (idx: number) => top + idx * (layerH + gap) * 2 + layerH / 2;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="隔层初始化:BERT 12 层每隔一层取一层初始化 DistilBERT 6 层">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        隔层初始化 — 用 BERT 第 {INIT_LAYER_INDICES.map((i) => i + 1).join(", ")} 层初始化 DistilBERT
      </text>

      <text x={teacherX + 40} y={45} textAnchor="middle" fontSize={11} fontWeight={700} fill="#3b82f6">
        Teacher(BERT-base,{TEACHER_LAYERS} 层)
      </text>
      {Array.from({ length: TEACHER_LAYERS }, (_, i) => {
        const isSelected = INIT_LAYER_INDICES.includes(i);
        const y = teacherLayerY(i);
        return (
          <g key={i}>
            <rect
              x={teacherX}
              y={y}
              width={140}
              height={layerH}
              rx={4}
              fill={isSelected && showMapping ? "#fef3c7" : "#dbeafe"}
              stroke={isSelected && showMapping ? "#f59e0b" : "#3b82f6"}
              strokeWidth={isSelected && showMapping ? 2 : 1}
            />
            <text x={teacherX + 70} y={y + 14} textAnchor="middle" fontSize={9} fill="var(--ink-primary)">
              Layer {i + 1}
            </text>
          </g>
        );
      })}

      <text x={studentX + 40} y={45} textAnchor="middle" fontSize={11} fontWeight={700} fill="#ec4899">
        Student(DistilBERT,{STUDENT_LAYERS} 层)
      </text>
      {Array.from({ length: STUDENT_LAYERS }, (_, i) => {
        const y = studentLayerY(i);
        return (
          <g key={i}>
            <rect x={studentX} y={y} width={140} height={layerH} rx={4} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.4} />
            <text x={studentX + 70} y={y + 14} textAnchor="middle" fontSize={9} fill="var(--ink-primary)">
              Layer {i + 1}
            </text>
          </g>
        );
      })}

      {showMapping &&
        INIT_LAYER_INDICES.map((teacherIdx, studentIdx) => {
          const y1 = teacherLayerY(teacherIdx) + layerH / 2;
          const y2 = studentLayerY(studentIdx) + layerH / 2;
          return (
            <line
              key={teacherIdx}
              x1={teacherX + 140}
              y1={y1}
              x2={studentX}
              y2={y2}
              stroke="#f59e0b"
              strokeWidth={1.4}
              strokeDasharray="4 3"
            />
          );
        })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
        {showMapping
          ? "奇数层(1,3,5,7,9,11)权重直接复制,收敛速度提升 5-10×"
          : "随机初始化对比 — 无 teacher 权重继承,训练需要从零收敛"}
      </text>
    </svg>
  );
}
