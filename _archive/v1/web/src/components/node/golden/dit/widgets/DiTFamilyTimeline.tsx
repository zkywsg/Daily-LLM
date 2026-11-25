import { DIT_FAMILY } from "../lib/data";

const W = 700;
const H = 280;

// 2022.12 → 2024.8 timeline 5 个变种
export function DiTFamilyTimeline() {
  const PAD_L = 30;
  const PAD_R = 30;
  const PAD_T = 60;
  const plotW = W - PAD_L - PAD_R;

  const minM = 2022 * 12 + 11;
  const maxM = 2024 * 12 + 9;
  const xOf = (year: number, month: number) => PAD_L + ((year * 12 + month - minM) / (maxM - minM)) * plotW;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="DiT family timeline">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        DiT Family — 2022.12 起 21 个月内改写整个生成式 AI 领域
      </text>

      {/* timeline */}
      <line x1={PAD_L} y1={140} x2={W - PAD_R} y2={140} stroke="#9ca3af" strokeWidth={2} />
      {[
        { year: 2023, label: "2023" },
        { year: 2024, label: "2024" },
      ].map((y, i) => (
        <g key={i}>
          <line x1={xOf(y.year, 1)} y1={134} x2={xOf(y.year, 1)} y2={146} stroke="#9ca3af" />
          <text x={xOf(y.year, 1)} y={160} textAnchor="middle" fontSize={10} fill="#6b7280">{y.label}</text>
        </g>
      ))}

      {DIT_FAMILY.map((v, i) => {
        const x = xOf(v.year, v.month);
        const isAbove = i % 2 === 0;
        const cy = isAbove ? 90 : 200;
        const lineY1 = isAbove ? 105 : 140;
        const lineY2 = isAbove ? 140 : 185;
        return (
          <g key={i}>
            <line x1={x} y1={lineY1} x2={x} y2={lineY2} stroke={v.color} strokeWidth={1.4} />
            <circle cx={x} cy={140} r={6} fill={v.color} stroke="#fff" strokeWidth={1.8} />

            <rect x={x - 75} y={cy - 18} width={150} height={36} rx={4} fill={v.color} fillOpacity={0.1} stroke={v.color} strokeWidth={1.2} />
            <text x={x} y={cy - 3} textAnchor="middle" fontSize={11} fontWeight={700} fill={v.color}>
              {v.name}{v.isVideo && " 📹"}
            </text>
            <text x={x} y={cy + 12} textAnchor="middle" fontSize={9} fill="#6b7280">{v.year}-{String(v.month).padStart(2, "0")} · {v.org}</text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 16} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        2024 年所有前沿生成模型(图像 / 视频 / 文图)几乎都是 DiT 系骨架
      </text>
    </svg>
  );
}
