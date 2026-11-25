const W = 700;
const H = 320;

interface Props {
  withDropout: boolean;
}

// 仿造 AlexNet 论文趋势:无 dropout 的 train/val gap 大,有 dropout 后 gap 收窄
export function TrainVsValGap({ withDropout }: Props) {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 60;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const epochs = 90;
  const xOf = (e: number) => PAD_L + (e / epochs) * plotW;
  const yOf = (err: number) => PAD_T + ((0.85 - err) / 0.85) * plotH;

  // Train curves: both quickly drop, but no-dropout drops faster
  function trainErr(e: number) {
    const base = withDropout ? 0.18 : 0.05; // floor
    return base + (0.80 - base) * Math.exp(-e * (withDropout ? 0.035 : 0.06));
  }
  // Val curves: dropout converges to ~0.37, no-dropout to ~0.43
  function valErr(e: number) {
    const floor = withDropout ? 0.375 : 0.43;
    return floor + (0.80 - floor) * Math.exp(-e * 0.04);
  }

  const trainPts = Array.from({ length: epochs + 1 }, (_, e) => `${xOf(e)},${yOf(trainErr(e))}`).join(" ");
  const valPts   = Array.from({ length: epochs + 1 }, (_, e) => `${xOf(e)},${yOf(valErr(e))}`).join(" ");

  const finalTrain = trainErr(epochs);
  const finalVal   = valErr(epochs);
  const gap = finalVal - finalTrain;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Train vs validation error with and without dropout">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        训练 vs 验证 Top-1 error — {withDropout ? "有 Dropout (p=0.5)" : "无 Dropout"}
      </text>

      {/* axes */}
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {/* y ticks */}
      {[0, 0.2, 0.4, 0.6, 0.8].map((y) => (
        <g key={y}>
          <line x1={PAD_L - 4} y1={yOf(y)} x2={PAD_L} y2={yOf(y)} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(y) + 4} textAnchor="end" fontSize={9} fill="#6b7280">{(y * 100).toFixed(0)}%</text>
          <line x1={PAD_L} y1={yOf(y)} x2={W - PAD_R} y2={yOf(y)} stroke="#f3f4f6" strokeDasharray="2 3" />
        </g>
      ))}
      {[0, 30, 60, 90].map((e) => (
        <g key={e}>
          <line x1={xOf(e)} y1={PAD_T + plotH} x2={xOf(e)} y2={PAD_T + plotH + 3} stroke="#9ca3af" />
          <text x={xOf(e)} y={PAD_T + plotH + 14} textAnchor="middle" fontSize={9} fill="#6b7280">{e}</text>
        </g>
      ))}

      <text x={W / 2} y={H - 28} textAnchor="middle" fontSize={10} fill="#6b7280">epoch</text>

      {/* curves */}
      <polyline points={trainPts} fill="none" stroke="#3b82f6" strokeWidth={2.2} />
      <polyline points={valPts} fill="none" stroke="#ec4899" strokeWidth={2.2} />

      {/* gap annotation */}
      <line x1={xOf(epochs) - 4} y1={yOf(finalTrain)} x2={xOf(epochs) - 4} y2={yOf(finalVal)} stroke="#1f2937" strokeWidth={1.5} markerEnd="url(#gap-arr)" markerStart="url(#gap-arr)" />
      <defs>
        <marker id="gap-arr" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#1f2937" />
        </marker>
      </defs>
      <text x={xOf(epochs) - 14} y={(yOf(finalTrain) + yOf(finalVal)) / 2 + 4} textAnchor="end" fontSize={11} fontWeight={700} fill="#1f2937">
        gap = {(gap * 100).toFixed(1)}%
      </text>

      {/* legend */}
      <g transform={`translate(${PAD_L + 20}, ${PAD_T + 12})`}>
        <line x1={0} y1={6} x2={20} y2={6} stroke="#3b82f6" strokeWidth={2.5} />
        <text x={26} y={10} fontSize={11} fontWeight={600} fill="#3b82f6">train</text>
        <line x1={80} y1={6} x2={100} y2={6} stroke="#ec4899" strokeWidth={2.5} />
        <text x={106} y={10} fontSize={11} fontWeight={600} fill="#ec4899">validation</text>
      </g>

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        {withDropout ? "gap ≈ 1-2%,泛化健康" : "gap ≈ 5%+,严重过拟合 — train 几乎到 0 但 val 还有 43%"}
      </text>
    </svg>
  );
}
