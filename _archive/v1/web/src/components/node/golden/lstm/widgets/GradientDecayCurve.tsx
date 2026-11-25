import { useMemo } from "react";
import { gradientFlow } from "../lib/lstm";

interface Props {
  forgetMean: number;
}

const W = 700;
const H = 300;
const PAD = { left: 60, right: 60, top: 36, bottom: 50 };

// 比较 vanilla RNN(每步乘 0.6)vs LSTM cell(每步乘 forgetMean≈0.95)的梯度衰减。
// log scale y 轴让 RNN 的指数衰减看得清楚。

export function GradientDecayCurve({ forgetMean }: Props) {
  const T = 50;
  const { rnn, lstm } = useMemo(() => gradientFlow(T, forgetMean), [forgetMean]);

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const xScale = (t: number) => PAD.left + (t / Math.max(1, T - 1)) * innerW;

  // log scale: [1e-10, 1]
  const logMin = -10;
  const logMax = 0;
  const yScale = (v: number) => {
    const l = Math.log10(Math.max(1e-12, v));
    return PAD.top + (1 - (l - logMin) / (logMax - logMin)) * innerH;
  };

  const rnnPts = rnn.map((v, i) => `${xScale(i)},${yScale(v)}`).join(" ");
  const lstmPts = lstm.map((v, i) => `${xScale(i)},${yScale(v)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`Gradient decay: RNN vs LSTM, forget mean ${forgetMean}`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        梯度沿时间反向衰减 — Vanilla RNN vs LSTM (log y)
      </text>

      {/* 轴 */}
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />

      {/* y 刻度 */}
      {[-10, -8, -6, -4, -2, 0].map((l) => (
        <g key={l}>
          <text x={PAD.left - 6} y={yScale(Math.pow(10, l)) + 4} textAnchor="end" fontSize={10} fill="var(--ink-muted)">
            1e{l}
          </text>
          <line x1={PAD.left} x2={W - PAD.right} y1={yScale(Math.pow(10, l))} y2={yScale(Math.pow(10, l))} stroke="var(--border)" strokeDasharray="1 4" />
        </g>
      ))}

      {/* x 刻度 */}
      {[0, 10, 20, 30, 40, 50].map((t) => (
        <text key={t} x={xScale(t)} y={H - PAD.bottom + 18} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
          {t}
        </text>
      ))}
      <text x={W / 2 - 30} y={H - 8} textAnchor="middle" fontSize={11} fill="var(--ink-muted)">
        反向 timestep →
      </text>

      {/* 危险线:1e-6 以下基本失效 */}
      <line x1={PAD.left} x2={W - PAD.right} y1={yScale(1e-6)} y2={yScale(1e-6)} stroke="#dc2626" strokeWidth={1.5} strokeDasharray="4 3" />
      <text x={W - PAD.right - 4} y={yScale(1e-6) - 4} textAnchor="end" fontSize={10} fontStyle="italic" fill="#dc2626">
        梯度消失 (~1e-6)
      </text>

      {/* RNN curve */}
      <polyline fill="none" stroke="#9ca3af" strokeWidth={2.2} points={rnnPts} />
      {/* LSTM curve */}
      <polyline fill="none" stroke="#ec4899" strokeWidth={2.4} points={lstmPts} />

      {/* 图例 */}
      <g transform={`translate(${W - PAD.right - 8}, ${PAD.top + 14})`}>
        <g transform="translate(-130, 0)">
          <line x1={0} x2={14} y1={0} y2={0} stroke="#ec4899" strokeWidth={2.4} />
          <text x={18} y={4} fontSize={10} fontWeight={600} fill="#ec4899">LSTM cell (f={forgetMean.toFixed(2)})</text>
        </g>
        <g transform="translate(-130, 16)">
          <line x1={0} x2={14} y1={0} y2={0} stroke="#9ca3af" strokeWidth={2.2} />
          <text x={18} y={4} fontSize={10} fontWeight={600} fill="#6b7280">Vanilla RNN (~0.6)</text>
        </g>
      </g>
    </svg>
  );
}
