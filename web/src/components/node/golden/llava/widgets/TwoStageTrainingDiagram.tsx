import { TWO_STAGE_TRAINING } from "../lib/data";

const W = 700;
const H = 240;

interface Props {
  activeStage: number;
}

export function TwoStageTrainingDiagram({ activeStage }: Props) {
  const boxW = 300;
  const gap = 30;
  const startX = (W - 2 * boxW - gap) / 2;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="LLaVA 两阶段训练流程">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        两阶段训练 — Stage 1 对齐 + Stage 2 指令跟随
      </text>

      {TWO_STAGE_TRAINING.map((s, i) => {
        const x = startX + i * (boxW + gap);
        const isActive = i === activeStage;
        return (
          <g key={s.stage}>
            <rect x={x} y={45} width={boxW} height={150} fill={isActive ? "#ecfdf5" : "var(--bg-surface)"} stroke={isActive ? "#10b981" : "var(--border)"} strokeWidth={isActive ? 2.2 : 1} rx={8} />
            <text x={x + boxW / 2} y={70} textAnchor="middle" fontSize={11} fontWeight={700} fill={isActive ? "#065f46" : "var(--ink-muted)"}>{s.stage}</text>

            <text x={x + 16} y={98} fontSize={9} fill={isActive ? "var(--ink-primary)" : "var(--ink-muted)"}>数据:{s.dataSize}</text>
            <text x={x + 16} y={120} fontSize={9} fill={isActive ? "var(--ink-primary)" : "var(--ink-muted)"}>可训练:{s.trainable}</text>
            <text x={x + 16} y={142} fontSize={9} fill={isActive ? "var(--ink-primary)" : "var(--ink-muted)"}>算力:{s.gpuTime}</text>

            {i === 0 && (
              <path d={`M ${x + boxW + 2} 120 L ${x + boxW + gap - 2} 120`} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#arrow-stage-llava)" />
            )}
          </g>
        );
      })}

      <text x={W / 2} y={220} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        lr 在两阶段差 100× — Stage 1 只训新加的 projection 层用大 lr,Stage 2 训整个 LLM 必须用小 lr 避免破坏预训练知识
      </text>

      <defs>
        <marker id="arrow-stage-llava" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#9ca3af" />
        </marker>
      </defs>
    </svg>
  );
}
