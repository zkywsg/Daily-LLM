import { simulateTraining } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  beta: number;
  lr: number;
}

export function RewardTrajectories({ beta, lr }: Props) {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 60;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const steps = 100;
  const data = simulateTraining(beta, lr, steps);

  // y range:chosen/rejected reward(β · log-ratio)
  const yMin = -1.5;
  const yMax = 2.0;
  const xOf = (s: number) => PAD_L + (s / steps) * plotW;
  const yOf = (v: number) => PAD_T + ((yMax - v) / (yMax - yMin)) * plotH;

  const chosenPts = data.map((d) => `${xOf(d.step)},${yOf(d.chosenReward)}`).join(" ");
  const rejectedPts = data.map((d) => `${xOf(d.step)},${yOf(d.rejectedReward)}`).join(" ");

  const final = data[data.length - 1];
  const margin = final.chosenReward - final.rejectedReward;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="DPO reward trajectories">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        训练动力学 — β · log(π_θ / π_ref) 在 100 step 内的走势
      </text>

      {/* axes */}
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {/* y ticks */}
      {[-1, 0, 1, 2].map((y) => (
        <g key={y}>
          <line x1={PAD_L - 4} y1={yOf(y)} x2={PAD_L} y2={yOf(y)} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(y) + 4} textAnchor="end" fontSize={9} fill="#6b7280">{y >= 0 ? "+" : ""}{y.toFixed(1)}</text>
          <line x1={PAD_L} y1={yOf(y)} x2={W - PAD_R} y2={yOf(y)} stroke="#f3f4f6" strokeDasharray="2 3" />
        </g>
      ))}

      {/* zero baseline (π_ref) */}
      <line x1={PAD_L} y1={yOf(0)} x2={W - PAD_R} y2={yOf(0)} stroke="#9ca3af" strokeWidth={1.5} strokeDasharray="4 4" />
      <text x={W - PAD_R - 4} y={yOf(0) - 6} textAnchor="end" fontSize={10} fontWeight={600} fill="#6b7280">π_ref baseline (= 0)</text>

      {/* curves */}
      <polyline points={chosenPts} fill="none" stroke="#10b981" strokeWidth={2.5} />
      <polyline points={rejectedPts} fill="none" stroke="#ec4899" strokeWidth={2.5} />

      {/* end dots + labels */}
      <circle cx={xOf(steps)} cy={yOf(final.chosenReward)} r={5} fill="#10b981" stroke="#fff" strokeWidth={1.5} />
      <text x={xOf(steps) - 6} y={yOf(final.chosenReward) - 8} textAnchor="end" fontSize={10} fontWeight={700} fill="#10b981">chosen +{final.chosenReward.toFixed(2)}</text>
      <circle cx={xOf(steps)} cy={yOf(final.rejectedReward)} r={5} fill="#ec4899" stroke="#fff" strokeWidth={1.5} />
      <text x={xOf(steps) - 6} y={yOf(final.rejectedReward) + 16} textAnchor="end" fontSize={10} fontWeight={700} fill="#ec4899">rejected {final.rejectedReward.toFixed(2)}</text>

      <text x={W / 2} y={H - 28} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
        margin = chosen − rejected = {margin.toFixed(2)} · β={beta.toFixed(2)} · lr={lr.toExponential(1)}
      </text>
      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        chosen 推高 / rejected 压低 · margin 越大对齐越强 · 但偏离 π_ref 也越多
      </text>
    </svg>
  );
}
