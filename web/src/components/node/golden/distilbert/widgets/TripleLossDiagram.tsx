import { LOSS_WEIGHTS } from "../lib/data";

const W = 700;
const H = 340;

const COLORS = [
  { fill: "#fef3c7", stroke: "#f59e0b" },
  { fill: "#dbeafe", stroke: "#3b82f6" },
  { fill: "#ecfdf5", stroke: "#10b981" },
];

export function TripleLossDiagram() {
  const barX = 60;
  const barMaxW = 380;
  const rowH = 60;
  const top = 60;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="三损失联合训练:蒸馏损失+MLM损失+余弦损失按权重合并">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        三损失联合训练 — L = α·L_distill + β·L_MLM + γ·L_cos
      </text>

      {LOSS_WEIGHTS.map((lw, i) => {
        const y = top + i * rowH;
        const w = lw.weight * barMaxW * 1.6;
        const c = COLORS[i];
        return (
          <g key={lw.name}>
            <text x={barX} y={y + 14} fontSize={10} fontWeight={700} fill="var(--ink-primary)">
              {lw.symbol} = {lw.weight.toFixed(1)} · {lw.name}
            </text>
            <rect x={barX} y={y + 20} width={barMaxW} height={24} fill="#f3f4f6" stroke="#e5e7eb" strokeWidth={1} rx={4} />
            <rect x={barX} y={y + 20} width={w} height={24} fill={c.fill} stroke={c.stroke} strokeWidth={1.6} rx={4} />
            <text x={barX + 8} y={y + 37} fontSize={9} fill="var(--ink-secondary)">
              {lw.desc}
            </text>
          </g>
        );
      })}

      {/* summing arrows into total loss */}
      <g transform={`translate(${barX + barMaxW + 40}, ${top})`}>
        <line x1={0} y1={30} x2={40} y2={95} stroke="#f59e0b" strokeWidth={1.6} />
        <line x1={0} y1={90} x2={40} y2={95} stroke="#3b82f6" strokeWidth={1.6} />
        <line x1={0} y1={150} x2={40} y2={95} stroke="#10b981" strokeWidth={1.6} />
        <circle cx={50} cy={95} r={26} fill="#fdf2f8" stroke="#ec4899" strokeWidth={2} />
        <text x={50} y={92} textAnchor="middle" fontSize={11} fontWeight={700} fill="#ec4899">L</text>
        <text x={50} y={106} textAnchor="middle" fontSize={9} fill="#ec4899">total</text>
      </g>

      <text x={W / 2} y={H - 16} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
        蒸馏权重最大(α=0.5)— teacher 软标签是主信号,MLM 防止无脑继承 teacher 错误,cosine 对齐中间表征
      </text>
    </svg>
  );
}
