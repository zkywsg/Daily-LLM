import { simulateGrpoGroup } from "../lib/data";

const W = 700;
const H = 280;

interface Props {
  rewards: number[];
}

export function GrpoGroupDiagram({ rewards }: Props) {
  const samples = simulateGrpoGroup(rewards);
  const colW = (W - 100) / samples.length;
  const barBaseY = 130;
  const barMaxH = 50;
  const advBaseY = 220;
  const advMaxH = 40;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="GRPO 组内采样与归一化演示">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        同一 prompt 采 G 个 response — 组内均值/方差归一化,无需 value model
      </text>

      <text x={50} y={60} fontSize={10} fontWeight={700} fill="#6b7280">reward(rule-based,0/1)</text>
      {samples.map((s, i) => {
        const x = 50 + i * colW;
        const h = s.reward * barMaxH;
        const color = s.reward > 0.5 ? "#10b981" : "#ec4899";
        return (
          <g key={i}>
            <rect x={x} y={barBaseY - h} width={colW - 10} height={h || 2} fill={color} opacity={0.85} rx={2} />
            <text x={x + (colW - 10) / 2} y={barBaseY + 14} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">y{i + 1}</text>
            <text x={x + (colW - 10) / 2} y={barBaseY - h - 6} textAnchor="middle" fontSize={9} fontWeight={700} fill={color}>{s.reward}</text>
          </g>
        );
      })}

      <text x={50} y={175} fontSize={10} fontWeight={700} fill="#6b7280">
        mean={((samples.reduce((a, s) => a + s.reward, 0)) / samples.length).toFixed(2)}(组内均值作 baseline,无 value model)
      </text>

      <text x={50} y={200} fontSize={10} fontWeight={700} fill="#6b7280">advantage = (reward - mean) / std</text>
      {samples.map((s, i) => {
        const x = 50 + i * colW;
        const h = Math.min(Math.abs(s.advantage) * advMaxH, advMaxH);
        const color = s.advantage >= 0 ? "#3b82f6" : "#f59e0b";
        const y = s.advantage >= 0 ? advBaseY - h : advBaseY;
        return (
          <g key={i}>
            <rect x={x} y={y} width={colW - 10} height={h || 1} fill={color} opacity={0.85} rx={2} />
            <text x={x + (colW - 10) / 2} y={advBaseY + 16} textAnchor="middle" fontSize={9} fontWeight={700} fill={color}>
              {s.advantage.toFixed(2)}
            </text>
          </g>
        );
      })}
      <line x1={40} y1={advBaseY} x2={W - 30} y2={advBaseY} stroke="var(--border)" strokeWidth={1} />
    </svg>
  );
}
