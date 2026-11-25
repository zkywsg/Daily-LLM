import { GPT4_STATS } from "../lib/data";

const W = 700;
const H = 300;

export function DualTrackDiagram() {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="GPT-4 闭源前沿规格一览">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        GPT-4 — 闭源前沿的估计规格(技术报告不公开细节,来自行业推测)
      </text>

      {GPT4_STATS.map((row, i) => {
        const col = i % 2;
        const rowIdx = Math.floor(i / 2);
        const x = 30 + col * 340;
        const y = 50 + rowIdx * 60;
        return (
          <g key={row.label}>
            <rect x={x} y={y} width={320} height={46} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.2} rx={6} />
            <text x={x + 12} y={y + 20} fontSize={10} fontWeight={700} fill="#1e40af">{row.label}</text>
            <text x={x + 12} y={y + 37} fontSize={10} fill="#374151">{row.value}</text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        MoE(16 专家)+ 多模态原生预训练 + 可预测 scaling,三个特有技术让 GPT-4 成为 2023 年闭源前沿的标杆
      </text>
    </svg>
  );
}
