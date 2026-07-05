import { CONTEXT_GROWTH } from "../lib/data";

const W = 700;
const H = 240;

interface Props {
  visibleSteps: number;
}

export function ContextWindowGrowthChart({ visibleSteps }: Props) {
  const PAD_L = 60;
  const PAD_R = 40;
  const PAD_T = 40;
  const PAD_B = 40;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const maxLog = Math.log2(128);

  const xOf = (i: number) => PAD_L + (i / (CONTEXT_GROWTH.length - 1)) * plotW;
  const yOf = (k: number) => PAD_T + plotH - (Math.log2(k) / maxLog) * plotH;

  const points = CONTEXT_GROWTH.slice(0, visibleSteps);
  const path = points.map((p, i) => `${i === 0 ? "M" : "L"} ${xOf(i)} ${yOf(p.contextK)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="GPT-4 上下文窗口扩展时间线">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        8K → 32K → 128K — RoPE 外推让上下文窗口持续扩张
      </text>

      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="var(--border)" strokeWidth={1} />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={PAD_L + plotW} y2={PAD_T + plotH} stroke="var(--border)" strokeWidth={1} />

      <path d={path} fill="none" stroke="#10b981" strokeWidth={2.4} />
      {points.map((p, i) => (
        <g key={p.label}>
          <circle cx={xOf(i)} cy={yOf(p.contextK)} r={5} fill="#10b981" />
          <text x={xOf(i)} y={yOf(p.contextK) - 12} textAnchor="middle" fontSize={11} fontWeight={700} fill="#065f46">{p.contextK}K</text>
          <text x={xOf(i)} y={PAD_T + plotH + 20} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">{p.label}</text>
        </g>
      ))}
    </svg>
  );
}
