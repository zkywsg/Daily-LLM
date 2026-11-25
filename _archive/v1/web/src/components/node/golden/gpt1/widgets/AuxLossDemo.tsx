const W = 700;
const H = 260;

interface Props {
  lambda: number;
}

// 展示 task loss + λ · LM loss 的曲线随 λ 变化的直觉:高 λ 更抗遗忘但学任务慢
export function AuxLossDemo({ lambda }: Props) {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 50;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const steps = 60;
  const xOf = (s: number) => PAD_L + (s / steps) * plotW;
  const yOf = (v: number) => PAD_T + (1 - v) * plotH;

  // task loss: 收敛速度受 lambda 影响(lambda 越大收敛越慢但更稳)
  function taskAcc(s: number) {
    const rate = 0.15 / (1 + lambda);
    return 1 - Math.exp(-rate * s);
  }
  // 通用语言能力保留度:lambda 越大保留越好
  function generalRetain(s: number) {
    const decay = 0.01 * (1 - lambda);
    return Math.max(0.3, 1 - decay * s);
  }

  const taskPts = Array.from({ length: steps + 1 }, (_, s) => `${xOf(s)},${yOf(taskAcc(s))}`).join(" ");
  const genPts = Array.from({ length: steps + 1 }, (_, s) => `${xOf(s)},${yOf(generalRetain(s))}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Auxiliary LM loss lambda demo">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        λ · L_LM auxiliary — λ = {lambda.toFixed(2)}
      </text>

      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {[0, 0.5, 1].map((v) => (
        <g key={v}>
          <line x1={PAD_L - 4} y1={yOf(v)} x2={PAD_L} y2={yOf(v)} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(v) + 4} textAnchor="end" fontSize={9} fill="#6b7280">{(v * 100).toFixed(0)}%</text>
        </g>
      ))}
      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={10} fill="#6b7280">微调 step</text>

      <polyline points={taskPts} fill="none" stroke="#ec4899" strokeWidth={2.4} />
      <polyline points={genPts} fill="none" stroke="#3b82f6" strokeWidth={2.4} />

      <g transform={`translate(${PAD_L + 14}, ${PAD_T + 10})`}>
        <line x1={0} y1={6} x2={20} y2={6} stroke="#ec4899" strokeWidth={2.5} />
        <text x={26} y={10} fontSize={10} fontWeight={600} fill="#ec4899">下游任务学习进度</text>
        <line x1={200} y1={6} x2={220} y2={6} stroke="#3b82f6" strokeWidth={2.5} />
        <text x={226} y={10} fontSize={10} fontWeight={600} fill="#3b82f6">通用语言能力保留</text>
      </g>

      <text x={W / 2} y={H - 26} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        λ 越大 → 任务学得慢但通用能力保留好;λ=0.5 是论文经验值
      </text>
    </svg>
  );
}
