import { simulateTrajectories } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  resetLow: boolean; // true = 周期性低 reset(模拟句子边界重启)
}

export function TrajectoryCompare({ resetLow }: Props) {
  const pattern = resetLow ? [1, 1, 1, 1, 0.1, 1, 1, 1, 1, 0.1] : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
  const data = simulateTrajectories(pattern);

  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 60;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const xOf = (s: number) => PAD_L + (s / 19) * plotW;
  const yOf = (v: number) => PAD_T + (1 - (v + 1) / 2) * plotH;

  const lstmHPts = data.map((d) => `${xOf(d.step)},${yOf(d.lstmH)}`).join(" ");
  const lstmCPts = data.map((d) => `${xOf(d.step)},${yOf(d.lstmC)}`).join(" ");
  const gruHPts = data.map((d) => `${xOf(d.step)},${yOf(d.gruH)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="LSTM vs GRU hidden state trajectories">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        同序列上 LSTM(C, h)vs GRU(h)轨迹对比
      </text>

      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={yOf(0)} x2={W - PAD_R} y2={yOf(0)} stroke="#e5e7eb" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {[-1, -0.5, 0, 0.5, 1].map((v) => (
        <g key={v}>
          <line x1={PAD_L - 4} y1={yOf(v)} x2={PAD_L} y2={yOf(v)} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(v) + 4} textAnchor="end" fontSize={9} fill="#6b7280">{v.toFixed(1)}</text>
        </g>
      ))}
      <text x={W / 2} y={H - 16} textAnchor="middle" fontSize={10} fill="#6b7280">timestep</text>

      {/* reset markers */}
      {resetLow && data.map((d, i) => (
        pattern[i % pattern.length] < 0.5 ? (
          <line key={i} x1={xOf(d.step)} y1={PAD_T} x2={xOf(d.step)} y2={PAD_T + plotH}
                stroke="#f59e0b" strokeWidth={1.2} strokeDasharray="3 3" opacity={0.5} />
        ) : null
      ))}

      <polyline points={lstmCPts} fill="none" stroke="#ec4899" strokeWidth={2} strokeDasharray="4 2" />
      <polyline points={lstmHPts} fill="none" stroke="#3b82f6" strokeWidth={2} />
      <polyline points={gruHPts} fill="none" stroke="#10b981" strokeWidth={2.4} />

      <g transform={`translate(${PAD_L + 14}, ${PAD_T + 8})`}>
        <line x1={0} y1={6} x2={20} y2={6} stroke="#ec4899" strokeWidth={2} strokeDasharray="4 2" />
        <text x={26} y={10} fontSize={10} fontWeight={600} fill="#ec4899">LSTM · C_t</text>
        <line x1={140} y1={6} x2={160} y2={6} stroke="#3b82f6" strokeWidth={2} />
        <text x={166} y={10} fontSize={10} fontWeight={600} fill="#3b82f6">LSTM · h_t</text>
        <line x1={280} y1={6} x2={300} y2={6} stroke="#10b981" strokeWidth={2.4} />
        <text x={306} y={10} fontSize={10} fontWeight={600} fill="#10b981">GRU · h_t</text>
      </g>

      <text x={W / 2} y={H - 2} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        {resetLow ? "黄色虚线 = reset gate 低点(句子边界重启)" : "均匀 reset,GRU h 与 LSTM h/C 走势相近"}
      </text>
    </svg>
  );
}
