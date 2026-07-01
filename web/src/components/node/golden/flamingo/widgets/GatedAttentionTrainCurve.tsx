import { gateAlpha } from "../lib/data";

const W = 700;
const H = 300;

interface Props {
  currentStep: number;
}

export function GatedAttentionTrainCurve({ currentStep }: Props) {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 50;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const maxStep = 15000;
  const xOf = (s: number) => PAD_L + (s / maxStep) * plotW;
  const yOf = (v: number) => PAD_T + (1 - v) * plotH;

  const pts: string[] = [];
  for (let i = 0; i <= 100; i++) {
    const s = (i / 100) * maxStep;
    pts.push(`${xOf(s)},${yOf(gateAlpha(s))}`);
  }

  const curAlpha = gateAlpha(currentStep);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Gated cross-attention training dynamics">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Gated Cross-Attention 训练动态 — tanh(α) 从 0 逐步开启
      </text>

      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {[0, 0.5, 1].map((v) => (
        <g key={v}>
          <line x1={PAD_L - 4} y1={yOf(v)} x2={PAD_L} y2={yOf(v)} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(v) + 4} textAnchor="end" fontSize={9} fill="#6b7280">{v.toFixed(1)}</text>
        </g>
      ))}
      {[0, 5000, 10000, 15000].map((s) => (
        <g key={s}>
          <line x1={xOf(s)} y1={PAD_T + plotH} x2={xOf(s)} y2={PAD_T + plotH + 3} stroke="#9ca3af" />
          <text x={xOf(s)} y={PAD_T + plotH + 14} textAnchor="middle" fontSize={9} fill="#6b7280">{s / 1000}K</text>
        </g>
      ))}
      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={10} fill="#6b7280">training step</text>
      <text x={20} y={PAD_T + plotH / 2} transform={`rotate(-90, 20, ${PAD_T + plotH / 2})`} textAnchor="middle" fontSize={10} fill="#6b7280">tanh(α)</text>

      <polyline points={pts.join(" ")} fill="none" stroke="#f59e0b" strokeWidth={2.4} />

      <line x1={xOf(currentStep)} y1={PAD_T} x2={xOf(currentStep)} y2={PAD_T + plotH} stroke="#1f2937" strokeWidth={1.5} strokeDasharray="4 3" />
      <circle cx={xOf(currentStep)} cy={yOf(curAlpha)} r={6} fill="#f59e0b" stroke="#fff" strokeWidth={2} />

      <text x={W / 2} y={H - 24} textAnchor="middle" fontSize={11} fontWeight={700} fill="#92400e">
        step {currentStep} · tanh(α) = {curAlpha.toFixed(3)} · {curAlpha < 0.05 ? "≈ 原 LLM identity" : curAlpha < 0.5 ? "视觉信号逐步开启" : "视觉信号已稳定影响"}
      </text>
    </svg>
  );
}
