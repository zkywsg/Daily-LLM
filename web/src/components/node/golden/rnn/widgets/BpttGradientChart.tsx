import { simulateBPTT } from "../lib/data";

const W = 700;
const H = 320;

interface Props {
  spectralRadius: number;
}

export function BpttGradientChart({ spectralRadius }: Props) {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 60;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const steps = 20;
  const data = simulateBPTT(spectralRadius, steps);

  const xOf = (s: number) => PAD_L + (s / (steps - 1)) * plotW;

  // log scale for gradient
  const logMin = -8, logMax = 2;
  const yOf = (g: number) => {
    const logG = Math.log10(Math.max(Math.abs(g), 1e-10));
    return PAD_T + ((logMax - logG) / (logMax - logMin)) * plotH;
  };

  const pts = data.map((d) => `${xOf(d.step)},${yOf(d.gradToH1)}`).join(" ");

  const finalGrad = data[data.length - 1].gradToH1;
  const status = Math.abs(finalGrad) < 1e-4 ? "梯度消失" : Math.abs(finalGrad) > 100 ? "梯度爆炸" : "相对稳定";
  const statusColor = Math.abs(finalGrad) < 1e-4 ? "#ec4899" : Math.abs(finalGrad) > 100 ? "#f59e0b" : "#10b981";

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="BPTT gradient decay or explosion">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        BPTT 梯度回传 — 谱半径 = {spectralRadius.toFixed(2)}
      </text>

      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {[-8, -6, -4, -2, 0, 2].map((lg) => (
        <g key={lg}>
          <line x1={PAD_L - 4} y1={yOf(Math.pow(10, lg))} x2={PAD_L} y2={yOf(Math.pow(10, lg))} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(Math.pow(10, lg)) + 4} textAnchor="end" fontSize={9} fill="#6b7280">10^{lg}</text>
        </g>
      ))}
      {[0, 5, 10, 15, 19].map((s) => (
        <g key={s}>
          <line x1={xOf(s)} y1={PAD_T + plotH} x2={xOf(s)} y2={PAD_T + plotH + 3} stroke="#9ca3af" />
          <text x={xOf(s)} y={PAD_T + plotH + 14} textAnchor="middle" fontSize={9} fill="#6b7280">{s}</text>
        </g>
      ))}
      <text x={W / 2} y={H - 16} textAnchor="middle" fontSize={10} fill="#6b7280">回传步数(T-t)</text>
      <text x={20} y={PAD_T + plotH / 2} transform={`rotate(-90, 20, ${PAD_T + plotH / 2})`} textAnchor="middle" fontSize={10} fill="#6b7280">梯度幅度(log)</text>

      <polyline points={pts} fill="none" stroke={statusColor} strokeWidth={2.4} />

      <text x={W / 2} y={H - 30} textAnchor="middle" fontSize={12} fontWeight={700} fill={statusColor}>
        {status} · 20 步后梯度 ≈ {finalGrad.toExponential(2)}
      </text>
    </svg>
  );
}
