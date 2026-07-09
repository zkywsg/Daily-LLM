import { TOOLS } from "../lib/data";

const W = 700;
const H = 180;

export function ToolsGridDiagram() {
  const colW = 130;
  const gap = 8;
  const startX = (W - TOOLS.length * colW - (TOOLS.length - 1) * gap) / 2;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Toolformer 集成的 5 个工具">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        5 个工具,每个只用 ≤20 个手写例子 — 零人工标注
      </text>

      {TOOLS.map((t, i) => {
        const x = startX + i * (colW + gap);
        return (
          <g key={t.name}>
            <rect x={x} y={45} width={colW} height={90} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.4} rx={6} />
            <text x={x + colW / 2} y={70} textAnchor="middle" fontSize={10} fontWeight={700} fill="#92400e">{t.name}</text>
            <foreignObject x={x + 6} y={80} width={colW - 12} height={50}>
              <div style={{ fontSize: 8, color: "#92400e", textAlign: "center", lineHeight: 1.4 }}>{t.backend}</div>
            </foreignObject>
          </g>
        );
      })}
    </svg>
  );
}
