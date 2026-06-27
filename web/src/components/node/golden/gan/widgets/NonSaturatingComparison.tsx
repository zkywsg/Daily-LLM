import { useMemo } from "react";
import { lossComparisonCurve } from "../lib/dist";

const W = 700;
const H = 280;
const PAD = { left: 56, right: 24, top: 36, bottom: 50 };

// 对比 G 的两种 loss:
//   original:    G 最小化 log(1 - D(G(z)))  → D(G(z)) 接近 0 时(G 很烂)梯度饱和 ≈ 0
//   non-saturating: G 最大化 log(D(G(z)))     → D(G(z)) 接近 0 时梯度大,救得回来
// 这是 Goodfellow 原论文里的 trick,实际所有 GAN 实现都用 non-saturating。

export function NonSaturatingComparison() {
  const { pseudoD, original, nonSat } = useMemo(() => lossComparisonCurve(), []);

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const xScale = (d: number) => PAD.left + d * innerW;
  const yMax = 5;
  const yMin = -5;
  const yScale = (v: number) => PAD.top + (1 - (v - yMin) / (yMax - yMin)) * innerH;

  const polyA = pseudoD.map((d, i) => `${xScale(d)},${yScale(original[i])}`).join(" ");
  const polyB = pseudoD.map((d, i) => `${xScale(d)},${yScale(nonSat[i])}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Original vs non-saturating G loss">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        G 损失对比:原版 vs Non-Saturating(横轴 = D(G(z)))
      </text>

      {/* 轴 */}
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} y1={yScale(0)} x2={W - PAD.right} y2={yScale(0)} stroke="var(--ink-muted)" strokeDasharray="2 3" />

      {/* x 轴标 */}
      {[0, 0.25, 0.5, 0.75, 1].map((d) => (
        <text key={d} x={xScale(d)} y={H - PAD.bottom + 18} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
          {d.toFixed(2)}
        </text>
      ))}
      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={11} fill="var(--ink-muted)">
        D(G(z))  =  D 给假图打分 →
      </text>

      {/* y 轴标 */}
      {[-4, -2, 0, 2, 4].map((v) => (
        <text key={v} x={PAD.left - 6} y={yScale(v) + 4} textAnchor="end" fontSize={10} fill="var(--ink-muted)">
          {v}
        </text>
      ))}

      {/* 曲线 */}
      <polyline fill="none" stroke="#9ca3af" strokeWidth={2.4} points={polyA} />
      <polyline fill="none" stroke="#ec4899" strokeWidth={2.4} points={polyB} />

      {/* 危险区(D≈0)标注 */}
      <rect x={xScale(0)} y={PAD.top} width={xScale(0.15) - PAD.left} height={innerH} fill="#fef3c7" opacity={0.5} />
      <text x={xScale(0.075)} y={PAD.top - 4} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#92400e">
        梯度饱和危险区
      </text>

      {/* 图例 */}
      <g transform={`translate(${PAD.left + 12}, ${PAD.top + 8})`}>
        <g>
          <line x1={0} x2={14} y1={0} y2={0} stroke="#9ca3af" strokeWidth={2.4} />
          <text x={18} y={4} fontSize={10} fill="var(--ink-secondary)">原版:log(1 − D)  → D→0 时梯度饱和</text>
        </g>
        <g transform="translate(0, 14)">
          <line x1={0} x2={14} y1={0} y2={0} stroke="#ec4899" strokeWidth={2.4} />
          <text x={18} y={4} fontSize={10} fill="var(--ink-secondary)">Non-Saturating:−log(D)  → D→0 时梯度爆炸,救得回来</text>
        </g>
      </g>
    </svg>
  );
}
