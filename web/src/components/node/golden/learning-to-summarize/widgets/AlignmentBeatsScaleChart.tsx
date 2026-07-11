import { HUMAN_PREF_SCORES } from "../lib/data";

const W = 700;
const H = 360;

const GROUP_COLOR: Record<string, { fill: string; stroke: string }> = {
  baseline: { fill: "#f3f4f6", stroke: "#9ca3af" },
  sft: { fill: "#dbeafe", stroke: "#3b82f6" },
  human: { fill: "#fef3c7", stroke: "#f59e0b" },
  rlhf: { fill: "#ecfdf5", stroke: "#10b981" },
};

export function AlignmentBeatsScaleChart() {
  const plotBottom = 300;
  const plotTop = 60;
  const maxVal = 80;
  const barW = 74;
  const gap = 24;
  const startX = 46;

  const yOf = (v: number) => plotBottom - (v / maxVal) * (plotBottom - plotTop);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="人工评测:RLHF vs SFT vs 参考摘要 vs 人类摘要">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        人工评测(% 时间被偏好 vs 参考摘要)— Reddit TL;DR,论文 Figure 1
      </text>

      <line x1={startX - 10} y1={yOf(50)} x2={W - 20} y2={yOf(50)} stroke="var(--ink-muted)" strokeWidth={1} strokeDasharray="3,3" />
      <text x={W - 16} y={yOf(50) - 4} textAnchor="end" fontSize={8} fill="var(--ink-muted)">50% 基准线(=参考摘要本身)</text>

      {HUMAN_PREF_SCORES.map((row, i) => {
        const x = startX + i * (barW + gap);
        const y = yOf(row.winRate);
        const h = plotBottom - y;
        const c = GROUP_COLOR[row.group];
        return (
          <g key={row.label}>
            <rect x={x} y={y} width={barW} height={h} rx={4} fill={c.fill} stroke={c.stroke} strokeWidth={row.group === "rlhf" ? 2.2 : 1.2} />
            <text x={x + barW / 2} y={y - 8} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
              {row.winRate}%
            </text>
            <text x={x + barW / 2} y={plotBottom + 16} textAnchor="middle" fontSize={8} fill="var(--ink-secondary)">
              {row.label.length > 8 ? `${row.label.slice(0, 8)}` : row.label}
            </text>
            <text x={x + barW / 2} y={plotBottom + 28} textAnchor="middle" fontSize={8} fill="var(--ink-secondary)">
              {row.label.length > 8 ? row.label.slice(8) : ""}
            </text>
          </g>
        );
      })}

      <line x1={startX - 10} y1={plotBottom} x2={W - 20} y2={plotBottom} stroke="var(--border)" strokeWidth={1} />

      <text x={W / 2} y={344} textAnchor="middle" fontSize={9} fontStyle="italic" fill="var(--ink-secondary)">
        1.3B RLHF(62%)超过 6.7B SFT(41%)— 对齐胜过 5× 参数;6.7B RLHF(74%)甚至超过人类摘要(70%)
      </text>
    </svg>
  );
}
