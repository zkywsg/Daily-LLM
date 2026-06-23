import { useMemo } from "react";
import {
  alphaBarCumulative,
  alphaFromBeta,
  betaSchedule,
  forwardSample,
  makeDemoSignal,
  reverseMean,
  seededGaussian,
  type Schedule,
} from "../lib/math";

interface Props {
  T: number;
  t: number;
  schedule: Schedule;
  /** 是否给 ε_θ 加噪声扰动模拟"网络预测不完美" */
  epsilonNoise: number;
  n?: number;
}

const W = 700;
const H = 220;
const PAD = { left: 36, right: 16, top: 30, bottom: 30 };

// 给定 x_t,假装我们有真 ε(在 demo 里我们确实知道,所以可以画出"完美 reverse"),
// 然后用 reverseMean 算 μ_{t-1}。viewer 拖 ε noise slider 模拟 ε_θ 不完美时
// reverse 也会偏离原信号 —— 这是为什么 DDPM 需要训练 ε_θ。
export function ReverseStrip({
  T,
  t,
  schedule,
  epsilonNoise,
  n = 64,
}: Props) {
  const { x0, xt, mu, epsNoisy } = useMemo(() => {
    const beta = betaSchedule(T, schedule);
    const alpha = alphaFromBeta(beta);
    const aBar = alphaBarCumulative(alpha);
    const x0 = makeDemoSignal(n);
    const eps = seededGaussian(n, 42);
    const tt = Math.max(1, Math.min(T - 1, t));
    const at = aBar[tt];
    const bt = beta[tt];
    const xt = forwardSample(x0, at, eps);
    // ε_θ 不完美:真 ε + δ·噪声
    const extra = seededGaussian(n, 99);
    const epsNoisy = eps.map((e, i) => e + epsilonNoise * extra[i]);
    const mu = reverseMean(xt, bt, at, epsNoisy);
    return { x0, xt, mu, epsNoisy };
  }, [T, t, schedule, epsilonNoise, n]);
  void epsNoisy;

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const cellW = innerW / n;
  const yScale = (v: number) => PAD.top + ((1 - (v + 3) / 6) * innerH);
  const polyFor = (vals: number[]) =>
    vals.map((v, i) => `${PAD.left + (i + 0.5) * cellW},${yScale(v)}`).join(" ");

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label={`Reverse μ_{t-1} estimate, ε noise = ${epsilonNoise.toFixed(2)}`}
    >
      {/* 0 线参考 */}
      <line
        x1={PAD.left}
        x2={W - PAD.right}
        y1={yScale(0)}
        y2={yScale(0)}
        stroke="var(--border)"
        strokeDasharray="2 4"
      />

      {/* baseline x_0 */}
      <polyline
        fill="none"
        stroke="#9ca3af"
        strokeWidth={1}
        strokeDasharray="3 3"
        points={polyFor(x0)}
      />
      {/* noisy input x_t */}
      <polyline
        fill="none"
        stroke="#fde68a"
        strokeWidth={1.2}
        points={polyFor(xt)}
        opacity={0.7}
      />
      {/* reverse mean μ_{t-1} */}
      <polyline fill="none" stroke="#10b981" strokeWidth={2.2} points={polyFor(mu)} />

      <text x={PAD.left} y={18} fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        reverse 一步:μ_{"ₜ₋₁"} = (1/√α_t)·(x_t − (β_t/√(1−ᾱ_t))·ε_θ)
      </text>
      <text x={W - PAD.right} y={18} textAnchor="end" fontSize={11} fill="var(--ink-muted)">
        ε noise δ = {epsilonNoise.toFixed(2)}
      </text>

      {/* 图例 */}
      <g transform={`translate(${PAD.left + 10}, ${H - PAD.bottom + 14})`}>
        <g>
          <line x1={0} x2={14} y1={0} y2={0} stroke="#fde68a" strokeWidth={1.2} />
          <text x={18} y={4} fontSize={10} fill="var(--ink-secondary)">x_t (输入)</text>
        </g>
        <g transform="translate(110, 0)">
          <line x1={0} x2={14} y1={0} y2={0} stroke="#10b981" strokeWidth={2.2} />
          <text x={18} y={4} fontSize={10} fill="var(--ink-secondary)">μ_{"ₜ₋₁"} (反向均值)</text>
        </g>
        <g transform="translate(260, 0)">
          <line x1={0} x2={14} y1={0} y2={0} stroke="#9ca3af" strokeWidth={1} strokeDasharray="3 3" />
          <text x={18} y={4} fontSize={10} fill="var(--ink-secondary)">x_0 (目标)</text>
        </g>
      </g>
    </svg>
  );
}
