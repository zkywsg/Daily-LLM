import { useMemo } from "react";
import {
  alphaBarCumulative,
  alphaFromBeta,
  betaSchedule,
  type Schedule,
} from "../lib/math";

interface Props {
  T: number;
  schedule: Schedule;
}

const W = 700;
const H = 220;
const PAD = { left: 50, right: 16, top: 30, bottom: 30 };

// L_vlb 隐含的每步权重 ∝ β_t² / (α_t (1 − ᾱ_t))  (Ho 2020 Eq. 12 化简里这一项)
// L_simple 把它拍平到 1 —— 画两条曲线一比就懂为啥简化目标更好优化:
// 早期 t 权重大 / 晚期权重快接近 0,VLB 把所有梯度 budget 砸前面 →
// 后面去噪学不充分。L_simple 反而均匀,U-Net 各 timestep 都见过。
export function LossWeightCurve({ T, schedule }: Props) {
  const { wVlb, wSimple } = useMemo(() => {
    const beta = betaSchedule(T, schedule);
    const alpha = alphaFromBeta(beta);
    const aBar = alphaBarCumulative(alpha);
    const w: number[] = [];
    for (let t = 0; t < T; t++) {
      const b = beta[t];
      const a = alpha[t];
      const ab = aBar[t];
      const denom = Math.max(1e-6, a * (1 - ab));
      w.push((b * b) / denom);
    }
    // 归一到 [0, 1] 好看
    const max = Math.max(...w);
    const normW = w.map((v) => v / max);
    const simple = Array.from({ length: T }, () => 1 / T).map(() => 1 / (max / Math.max(...w))); // 1 after normalize
    void simple;
    return { wVlb: normW, wSimple: Array.from({ length: T }, () => 0.5) };
  }, [T, schedule]);

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const xScale = (i: number) => PAD.left + (i / Math.max(1, T - 1)) * innerW;
  const yScale = (v: number) => PAD.top + (1 - v) * innerH;
  const poly = (vals: number[]) => vals.map((v, i) => `${xScale(i)},${yScale(v)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Loss weight curves: L_vlb implicit vs L_simple flat">
      {/* 轴 */}
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - 16} y2={H - PAD.bottom} stroke="var(--border)" />

      {[0, 0.5, 1].map((v) => (
        <g key={v}>
          <text x={PAD.left - 6} y={yScale(v) + 4} textAnchor="end" fontSize={10} fill="var(--ink-muted)">
            {v.toFixed(1)}
          </text>
          <line x1={PAD.left} x2={W - 16} y1={yScale(v)} y2={yScale(v)} stroke="var(--border)" strokeDasharray="1 4" />
        </g>
      ))}

      {/* L_vlb 隐含权重(随 t 大幅变化) */}
      <polyline fill="none" stroke="#3b82f6" strokeWidth={2.2} points={poly(wVlb)} />
      {/* L_simple 拍平到 0.5(任意常数,关键是水平) */}
      <polyline fill="none" stroke="#10b981" strokeWidth={2.2} strokeDasharray="4 3" points={poly(wSimple)} />

      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={11} fill="var(--ink-muted)">
        timestep t →
      </text>
      <text x={PAD.left + 8} y={PAD.top - 10} fontSize={11} fontWeight={600} fill="var(--ink-primary)">
        每 timestep 在 loss 里的有效权重(归一化)
      </text>

      {/* 图例 */}
      <g transform={`translate(${W - 220}, ${PAD.top + 6})`}>
        <g>
          <line x1={0} x2={14} y1={0} y2={0} stroke="#3b82f6" strokeWidth={2.2} />
          <text x={18} y={4} fontSize={10} fill="var(--ink-secondary)">L_vlb 隐含 (∝β²/(α(1-ᾱ)))</text>
        </g>
        <g transform="translate(0, 16)">
          <line x1={0} x2={14} y1={0} y2={0} stroke="#10b981" strokeWidth={2.2} strokeDasharray="4 3" />
          <text x={18} y={4} fontSize={10} fill="var(--ink-secondary)">L_simple 拍平</text>
        </g>
      </g>
    </svg>
  );
}
