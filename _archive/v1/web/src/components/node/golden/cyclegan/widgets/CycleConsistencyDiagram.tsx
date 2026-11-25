import { CYCLE_TRAINING_CURVE } from "../lib/data";

interface Props {
  step: number; // index into CYCLE_TRAINING_CURVE
}

const W = 700;
const H = 320;

export function CycleConsistencyDiagram({ step }: Props) {
  const point = CYCLE_TRAINING_CURVE[step];
  const gapPx = 10 + point.reconError * 90; // 重建误差越大,x 和 F(G(x)) 距离越大

  const xPos = { x: 60, y: 140 };
  const gxPos = { x: 300, y: 60 };
  const fgxPos = { x: 300 + gapPx, y: 220 };

  const color = point.contentPreserved ? "#10b981" : "#ec4899";

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="Cycle consistency：x -> G(x) -> F(G(x)) 循环与重建误差"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Cycle Consistency — x → G(x) → F(G(x)) ≈ x
      </text>

      {/* x box */}
      <rect x={xPos.x} y={xPos.y} width={90} height={60} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.6} rx={6} />
      <text x={xPos.x + 45} y={xPos.y + 34} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--ink-primary)">x(马)</text>

      {/* G(x) box */}
      <rect x={gxPos.x} y={gxPos.y} width={110} height={60} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.6} rx={6} />
      <text x={gxPos.x + 55} y={gxPos.y + 34} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--ink-primary)">G(x) 假斑马</text>

      {/* F(G(x)) box, position shifts with reconError */}
      <rect x={fgxPos.x} y={fgxPos.y} width={110} height={60} fill="#ecfdf5" stroke={color} strokeWidth={2} rx={6} />
      <text x={fgxPos.x + 55} y={fgxPos.y + 26} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">F(G(x))</text>
      <text x={fgxPos.x + 55} y={fgxPos.y + 42} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">重建马</text>

      {/* arrows */}
      <line x1={xPos.x + 90} y1={xPos.y + 20} x2={gxPos.x} y2={gxPos.y + 30} stroke="var(--border)" strokeWidth={1.6} markerEnd="url(#arrow-cyc)" />
      <text x={(xPos.x + 90 + gxPos.x) / 2} y={(xPos.y + gxPos.y) / 2 + 5} fontSize={10} fill="var(--ink-secondary)">G</text>

      <line x1={gxPos.x + 55} y1={gxPos.y + 60} x2={fgxPos.x + 55} y2={fgxPos.y} stroke="var(--border)" strokeWidth={1.6} markerEnd="url(#arrow-cyc)" />
      <text x={(gxPos.x + fgxPos.x) / 2 + 55} y={(gxPos.y + fgxPos.y) / 2 + 30} fontSize={10} fill="var(--ink-secondary)">F</text>

      {/* dashed line showing reconstruction gap back to x */}
      <line
        x1={fgxPos.x}
        y1={fgxPos.y + 30}
        x2={xPos.x + 90}
        y2={xPos.y + 40}
        stroke={color}
        strokeWidth={2}
        strokeDasharray="5,4"
      />
      <text x={(fgxPos.x + xPos.x + 90) / 2} y={(fgxPos.y + xPos.y) / 2 + 70} textAnchor="middle" fontSize={11} fontWeight={700} fill={color}>
        重建误差 ‖F(G(x))-x‖₁ = {point.reconError.toFixed(2)}
      </text>

      <text x={W / 2} y={H - 40} textAnchor="middle" fontSize={12} fontWeight={700} fill={color}>
        训练步数 {point.step} — {point.contentPreserved ? "内容已保留(cycle loss 生效)" : "尚未收敛(内容仍会丢失)"}
      </text>
      <text x={W / 2} y={H - 18} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        F(G(x)) 离 x 越近(虚线越短),说明 G 保留 x 的内容信息越好
      </text>

      <defs>
        <marker id="arrow-cyc" markerWidth={8} markerHeight={8} refX={6} refY={4} orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="var(--border)" />
        </marker>
      </defs>
    </svg>
  );
}
