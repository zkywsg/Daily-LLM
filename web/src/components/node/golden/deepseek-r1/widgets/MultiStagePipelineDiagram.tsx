import { MULTI_STAGE_PIPELINE } from "../lib/data";

const W = 700;
const H = 200;

interface Props {
  activeStage: number;
}

export function MultiStagePipelineDiagram({ activeStage }: Props) {
  const boxW = 150;
  const gap = 16;
  const startX = 20;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="R1 四阶段训练流程">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        R1 四阶段训练 — Stage 2 出 reasoning 内核,Stage 3/4 叠加通用能力 + 对齐
      </text>

      {MULTI_STAGE_PIPELINE.map((stage, i) => {
        const x = startX + i * (boxW + gap);
        const isActive = i === activeStage;
        const isReasoningCore = i === 1;
        return (
          <g key={i}>
            <rect
              x={x}
              y={60}
              width={boxW}
              height={70}
              fill={isActive ? (isReasoningCore ? "#ecfdf5" : "#dbeafe") : "var(--bg-surface)"}
              stroke={isActive ? (isReasoningCore ? "#10b981" : "#3b82f6") : "var(--border)"}
              strokeWidth={isActive ? 2.4 : 1}
              rx={6}
            />
            <text x={x + boxW / 2} y={82} textAnchor="middle" fontSize={10} fontWeight={700} fill={isActive ? "var(--ink-primary)" : "var(--ink-muted)"}>
              Stage {i + 1}
            </text>
            <text x={x + boxW / 2} y={100} textAnchor="middle" fontSize={10} fill={isActive ? "var(--ink-primary)" : "var(--ink-muted)"}>
              {stage.short}
            </text>
            {isReasoningCore && (
              <text x={x + boxW / 2} y={118} textAnchor="middle" fontSize={8} fontWeight={700} fill="#065f46">reasoning 内核来源</text>
            )}
            {i < MULTI_STAGE_PIPELINE.length - 1 && (
              <path d={`M ${x + boxW + 2} 95 L ${x + boxW + gap - 2} 95`} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#arrow-stage)" />
            )}
          </g>
        );
      })}

      <defs>
        <marker id="arrow-stage" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#9ca3af" />
        </marker>
      </defs>
    </svg>
  );
}
