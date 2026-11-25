import { PE_TIMELINE } from "../lib/data";

const W = 700;
const H = 300;

const TYPE_COLOR: Record<string, string> = {
  learned: "#9ca3af",
  sinusoidal: "#3b82f6",
  "relative-bias": "#f59e0b",
  rope: "#ec4899",
  alibi: "#10b981",
};
const TYPE_LABEL: Record<string, string> = {
  learned: "learned absolute",
  sinusoidal: "sinusoidal",
  "relative-bias": "relative bias",
  rope: "RoPE",
  alibi: "ALiBi",
};

export function PeTimelineChart() {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 60;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const yMin = 2017, yMax = 2024;
  const xOf = (y: number) => PAD_L + ((y - yMin) / (yMax - yMin)) * plotW;

  // stack models by year to avoid overlap
  const yearCounts: Record<number, number> = {};

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Position encoding adoption timeline">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        位置编码方案演化 — RoPE 成为 2022+ 事实标准
      </text>

      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" strokeWidth={1.5} />
      {[2017, 2019, 2021, 2023].map((y) => (
        <g key={y}>
          <line x1={xOf(y)} y1={PAD_T + plotH} x2={xOf(y)} y2={PAD_T + plotH + 4} stroke="#9ca3af" />
          <text x={xOf(y)} y={PAD_T + plotH + 18} textAnchor="middle" fontSize={10} fill="#6b7280">{y}</text>
        </g>
      ))}

      {PE_TIMELINE.map((m, i) => {
        const x = xOf(m.year);
        const stackIdx = yearCounts[m.year] || 0;
        yearCounts[m.year] = stackIdx + 1;
        const y = PAD_T + plotH - 20 - stackIdx * 34;
        const color = TYPE_COLOR[m.peType];
        return (
          <g key={i}>
            <line x1={x} y1={y + 12} x2={x} y2={PAD_T + plotH} stroke={color} strokeWidth={1} opacity={0.3} />
            <circle cx={x} cy={y} r={5} fill={color} stroke="#fff" strokeWidth={1.5} />
            <text x={x + 8} y={y + 4} fontSize={10} fontWeight={600} fill={color}>{m.model}</text>
          </g>
        );
      })}

      <g transform={`translate(${PAD_L}, 40)`}>
        {Object.entries(TYPE_LABEL).map(([k, label], i) => (
          <g key={k} transform={`translate(${i * 120}, 0)`}>
            <circle cx={4} cy={0} r={4} fill={TYPE_COLOR[k]} />
            <text x={12} y={4} fontSize={9} fill="#374151">{label}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}
