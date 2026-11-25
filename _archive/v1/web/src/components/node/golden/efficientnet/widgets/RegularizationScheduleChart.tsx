import { REGULARIZATION_SCHEDULE } from "../lib/data";

const W = 700;
const H = 300;

export function RegularizationScheduleChart() {
  const PAD_L = 46;
  const PAD_R = 24;
  const PAD_T = 56;
  const PAD_B = 44;
  const plotW = W - PAD_L - PAD_R;
  const baseY = H - PAD_B;
  const maxVal = 0.55;
  const scaleY = (baseY - PAD_T) / maxVal;
  const yFor = (v: number) => baseY - v * scaleY;
  const colW = plotW / (REGULARIZATION_SCHEDULE.length - 1);
  const xFor = (i: number) => PAD_L + i * colW;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="EfficientNet B0-B7 的 Dropout 与 Stochastic Depth 正则强度随模型规模线性增长"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        正则强度随模型规模线性上涨 — Dropout / Stochastic Depth
      </text>
      <text x={W / 2} y={36} textAnchor="middle" fontSize={9.5} fill="#ec4899">
        ● Dropout(大点 = 源文档锚点 B0/B7)
        <tspan fill="#3b82f6" dx={16}>
          ● Stochastic Depth(锚点 B0/B4/B7,其余线性插值)
        </tspan>
      </text>

      {[0, 0.1, 0.2, 0.3, 0.4, 0.5].map((v) => (
        <g key={v}>
          <line x1={PAD_L} x2={W - PAD_R} y1={yFor(v)} y2={yFor(v)} stroke="var(--border)" strokeWidth={1} strokeDasharray="2,3" />
          <text x={PAD_L - 8} y={yFor(v) + 3} textAnchor="end" fontSize={9} fill="var(--ink-muted)">
            {v.toFixed(1)}
          </text>
        </g>
      ))}

      {REGULARIZATION_SCHEDULE.map((row, i) => (
        <text key={row.variant} x={xFor(i)} y={baseY + 16} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
          {row.variant}
        </text>
      ))}

      <path
        d={REGULARIZATION_SCHEDULE.map((r, i) => `${i === 0 ? "M" : "L"}${xFor(i)},${yFor(r.dropout)}`).join(" ")}
        fill="none"
        stroke="#ec4899"
        strokeWidth={2.5}
      />
      <path
        d={REGULARIZATION_SCHEDULE.map((r, i) => `${i === 0 ? "M" : "L"}${xFor(i)},${yFor(r.stochasticDepth)}`).join(" ")}
        fill="none"
        stroke="#3b82f6"
        strokeWidth={2.5}
      />

      {REGULARIZATION_SCHEDULE.map((row, i) => (
        <g key={`pts-${row.variant}`}>
          <circle cx={xFor(i)} cy={yFor(row.dropout)} r={row.anchor ? 4.5 : 2.5} fill="#ec4899" stroke={row.anchor ? "var(--bg-surface)" : "none"} strokeWidth={row.anchor ? 1.5 : 0} />
          <circle cx={xFor(i)} cy={yFor(row.stochasticDepth)} r={row.anchor ? 4.5 : 2.5} fill="#3b82f6" stroke={row.anchor ? "var(--bg-surface)" : "none"} strokeWidth={row.anchor ? 1.5 : 0} />
        </g>
      ))}

    </svg>
  );
}
