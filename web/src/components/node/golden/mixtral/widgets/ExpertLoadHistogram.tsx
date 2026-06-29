import { expertLoad, NUM_EXPERTS } from "../lib/data";

interface Props {
  balanced: boolean;
}

const W = 700;
const H = 320;

// 8 个 expert 在某个 batch 内处理的 token 数柱状图。
// 不加 aux loss → 极不均匀(某几个 expert 长期主导,某几个永远闲置)
// 加 aux loss  → 接近均匀

const TARGET = 1.0; // 均匀时每 expert ≈ 1 (相对单位)

export function ExpertLoadHistogram({ balanced }: Props) {
  const loads = expertLoad(balanced);
  const max = Math.max(...loads, 3.5);

  const PAD = { left: 60, right: 30, top: 50, bottom: 60 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const barW = (innerW / NUM_EXPERTS) * 0.7;
  const gap = (innerW / NUM_EXPERTS) * 0.3;
  const yScale = (v: number) => PAD.top + (1 - v / max) * innerH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`Expert load distribution ${balanced ? "balanced" : "imbalanced"}`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Expert 负载分布 {balanced ? "(加 aux loss · 均匀)" : "(无 aux loss · 不均匀)"}
      </text>
      <text x={W / 2} y={36} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        每 expert 处理的 token 数(相对单位,1.0 = 均匀理想值)
      </text>

      {/* y 轴 */}
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />

      {[0, 1, 2, 3].map((v) => (
        <g key={v}>
          <text x={PAD.left - 6} y={yScale(v) + 4} textAnchor="end" fontSize={10} fill="var(--ink-muted)">
            {v.toFixed(1)}
          </text>
          <line x1={PAD.left} x2={W - PAD.right} y1={yScale(v)} y2={yScale(v)} stroke="var(--border)" strokeDasharray="1 4" />
        </g>
      ))}

      {/* 1.0 理想线 */}
      <line x1={PAD.left} x2={W - PAD.right} y1={yScale(TARGET)} y2={yScale(TARGET)} stroke="#10b981" strokeWidth={1.5} strokeDasharray="4 3" />
      <text x={W - PAD.right - 4} y={yScale(TARGET) - 4} textAnchor="end" fontSize={10} fontStyle="italic" fill="#10b981">
        理想均匀 = 1.0
      </text>

      {/* 柱子 */}
      {loads.map((l, i) => {
        const x = PAD.left + i * (barW + gap) + gap / 2;
        const h = (l / max) * innerH;
        const tooHot = l > 2;
        const tooCold = l < 0.5;
        const color = tooHot ? "#dc2626" : tooCold ? "#9ca3af" : "#ec4899";
        return (
          <g key={i}>
            <rect x={x} y={H - PAD.bottom - h} width={barW} height={h} rx={3} fill={color} opacity={0.8} />
            <text x={x + barW / 2} y={H - PAD.bottom - h - 4} textAnchor="middle" fontSize={10} fontWeight={600} fill="var(--ink-primary)">
              {l.toFixed(1)}
            </text>
            <text x={x + barW / 2} y={H - PAD.bottom + 16} textAnchor="middle" fontSize={10} fontWeight={500} fill="var(--ink-secondary)">
              E{i}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill={balanced ? "#10b981" : "#dc2626"}>
        {balanced
          ? "✓ 所有 expert 都在工作 · 容量真的能用满"
          : "✗ E2/E3/E5/E7 几乎闲置 · 47B 参数实际只在用 \"热 expert\" 的部分"}
      </text>
    </svg>
  );
}
