import { RL_TRAINING_STAGES } from "../lib/data";

const W = 700;

interface Props {
  stageIdx: number;
}

export function RlTrainingDemo({ stageIdx }: Props) {
  const stage = RL_TRAINING_STAGES[stageIdx];
  const lines = stage.text.split("\n");
  const H = 90 + lines.length * 20 + 40;
  const barMaxW = 300;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="RL 训练让模型逐渐学会长 thinking 演示">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {stage.step}
      </text>

      <rect x={30} y={40} width={W - 60} height={lines.length * 20 + 20} fill={stage.hasReflection ? "#ecfdf5" : "#f3f4f6"} stroke={stage.hasReflection ? "#10b981" : "#9ca3af"} strokeWidth={1.4} rx={6} />
      {lines.map((line, i) => (
        <text key={i} x={44} y={60 + i * 20} fontSize={11} fontFamily="monospace" fill={line.includes("Wait") || line.includes("验证") ? "#065f46" : "#374151"} fontWeight={line.includes("Wait") ? 700 : 400}>
          {line}
        </text>
      ))}

      <g transform={`translate(30, ${60 + lines.length * 20})`}>
        <text x={0} y={16} fontSize={10} fontWeight={700} fill="#6b7280">reward</text>
        <rect x={60} y={4} width={barMaxW} height={16} fill="#f3f4f6" stroke="#9ca3af" rx={3} />
        <rect x={60} y={4} width={stage.reward * barMaxW} height={16} fill={stage.reward > 0.7 ? "#10b981" : stage.reward > 0.2 ? "#3b82f6" : "#ec4899"} rx={3} />
        <text x={60 + barMaxW + 8} y={16} fontSize={11} fontWeight={700} fill="var(--ink-primary)">{stage.reward.toFixed(2)}</text>
      </g>
    </svg>
  );
}
