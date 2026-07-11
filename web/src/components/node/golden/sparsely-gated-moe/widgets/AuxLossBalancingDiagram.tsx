import { expertLoad } from "../lib/data";

interface Props {
  balanced: boolean;
}

const W = 700;
const H = 320;

// Expert 负载分布直方图(采样 16 个 expert 代表 2048 个):
// 不加 aux loss → 极不均匀(少数 expert 被反复选中,大多数几乎闲置)
// 加 aux loss(importance + load loss) → 接近均匀

const TARGET = 1.0;

export function AuxLossBalancingDiagram({ balanced }: Props) {
  const loads = expertLoad(balanced);
  const n = loads.length;
  const max = Math.max(...loads, 3.5);

  const PAD = { left: 60, right: 30, top: 54, bottom: 60 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const barW = (innerW / n) * 0.72;
  const gap = (innerW / n) * 0.28;
  const yScale = (v: number) => PAD.top + (1 - v / max) * innerH;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label={`Expert load distribution ${balanced ? "with" : "without"} auxiliary loss`}
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Expert 负载分布 {balanced ? "(加 Importance + Load Loss · 均匀)" : "(无 auxiliary loss · 塌缩)"}
      </text>
      <text x={W / 2} y={36} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        采样 16 个 expert(代表 2048 个),每 expert 处理的相对 token 量,1.0 = 理想均匀值
      </text>

      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />

      {[0, 1, 2, 3, 4].map((v) => (
        <g key={v}>
          <text x={PAD.left - 6} y={yScale(v) + 4} textAnchor="end" fontSize={10} fill="var(--ink-muted)">
            {v.toFixed(1)}
          </text>
          <line x1={PAD.left} x2={W - PAD.right} y1={yScale(v)} y2={yScale(v)} stroke="var(--border)" strokeDasharray="1 4" />
        </g>
      ))}

      <line x1={PAD.left} x2={W - PAD.right} y1={yScale(TARGET)} y2={yScale(TARGET)} stroke="#10b981" strokeWidth={1.5} strokeDasharray="4 3" />
      <text x={W - PAD.right - 4} y={yScale(TARGET) - 4} textAnchor="end" fontSize={10} fontStyle="italic" fill="#10b981">
        理想均匀 = 1.0
      </text>

      {loads.map((l, i) => {
        const x = PAD.left + i * (barW + gap) + gap / 2;
        const h = (l / max) * innerH;
        const tooHot = l > 2.5;
        const tooCold = l < 0.5;
        const color = tooHot ? "#dc2626" : tooCold ? "#9ca3af" : "#ec4899";
        return (
          <g key={i}>
            <rect x={x} y={H - PAD.bottom - h} width={barW} height={h} rx={2} fill={color} opacity={0.82} />
            <text x={x + barW / 2} y={H - PAD.bottom - h - 4} textAnchor="middle" fontSize={9} fontWeight={600} fill="var(--ink-primary)">
              {l.toFixed(1)}
            </text>
            <text x={x + barW / 2} y={H - PAD.bottom + 14} textAnchor="middle" fontSize={8} fontWeight={500} fill="var(--ink-secondary)">
              E{i}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill={balanced ? "#10b981" : "#dc2626"}>
        {balanced
          ? "✓ 所有 expert 都在被训练 · 137B 参数容量真的能用满"
          : "✗ 少数 expert 被反复选中,大多数永远学不到东西 · 参数堆上去也没用"}
      </text>
    </svg>
  );
}
