import { useMemo } from "react";
import {
  alphaBarCumulative,
  alphaFromBeta,
  betaSchedule,
  forwardSample,
  makeDemoSignal,
  seededGaussian,
  type Schedule,
} from "../lib/math";

interface Props {
  T: number;
  t: number;
  schedule: Schedule;
  epsilonNoise: number;
  n?: number;
}

const W = 700;
const H = 200;
const PAD = { left: 36, right: 16, top: 26, bottom: 26 };

// 把"真噪声 ε vs 网络预测 ε_θ"两条曲线叠画。
// epsilonNoise=0 时两条重合(完美预测);加大 δ 看到 ε_θ 偏离真 ε。
// 这是 DDPM 训练的核心目标 —— L_simple = E[||ε - ε_θ||²]
export function EpsilonPredictionView({
  T,
  t,
  schedule,
  epsilonNoise,
  n = 64,
}: Props) {
  const { eps, epsTheta, mse } = useMemo(() => {
    const beta = betaSchedule(T, schedule);
    const alpha = alphaFromBeta(beta);
    const aBar = alphaBarCumulative(alpha);
    const x0 = makeDemoSignal(n);
    const tt = Math.max(0, Math.min(T - 1, t));
    const at = aBar[tt];
    const eps = seededGaussian(n, 42);
    // ε_θ(x_t, t) demo:用真 ε + 扰动
    const noise = seededGaussian(n, 99);
    const epsTheta = eps.map((e, i) => e + epsilonNoise * noise[i]);
    let s = 0;
    for (let i = 0; i < n; i++) s += (eps[i] - epsTheta[i]) ** 2;
    // forward only to validate at is used
    void forwardSample(x0, at, eps);
    return { eps, epsTheta, mse: s / n };
  }, [T, t, schedule, epsilonNoise, n]);

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const cellW = innerW / n;
  // ε 通常 ∈ ~[-3, 3]
  const yScale = (v: number) => PAD.top + (1 - (v + 3) / 6) * innerH;
  const poly = (vals: number[]) =>
    vals.map((v, i) => `${PAD.left + (i + 0.5) * cellW},${yScale(v)}`).join(" ");

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label={`Epsilon prediction view, MSE=${mse.toFixed(3)}`}
    >
      {/* 0 线 */}
      <line
        x1={PAD.left}
        x2={W - PAD.right}
        y1={yScale(0)}
        y2={yScale(0)}
        stroke="var(--border)"
        strokeDasharray="2 4"
      />

      <polyline fill="none" stroke="#ec4899" strokeWidth={1.8} points={poly(eps)} />
      <polyline
        fill="none"
        stroke="#3b82f6"
        strokeWidth={1.6}
        strokeDasharray="4 2"
        points={poly(epsTheta)}
      />

      <text x={PAD.left} y={16} fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        ε vs ε_θ (网络预测)
      </text>
      <text x={W - PAD.right} y={16} textAnchor="end" fontSize={11} fill="var(--ink-muted)">
        MSE = {mse.toFixed(3)}
      </text>

      <g transform={`translate(${PAD.left + 10}, ${H - PAD.bottom + 14})`}>
        <g>
          <line x1={0} x2={14} y1={0} y2={0} stroke="#ec4899" strokeWidth={1.8} />
          <text x={18} y={4} fontSize={10} fill="var(--ink-secondary)">真 ε (训练 target)</text>
        </g>
        <g transform="translate(160, 0)">
          <line x1={0} x2={14} y1={0} y2={0} stroke="#3b82f6" strokeWidth={1.6} strokeDasharray="4 2" />
          <text x={18} y={4} fontSize={10} fill="var(--ink-secondary)">ε_θ (网络预测)</text>
        </g>
      </g>
    </svg>
  );
}
