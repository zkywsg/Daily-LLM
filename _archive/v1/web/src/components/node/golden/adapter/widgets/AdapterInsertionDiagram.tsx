import { BOTTLENECK_DIM } from "../lib/data";

const W = 640;
const H = 460;

export function AdapterInsertionDiagram() {
  const blockX = 140;
  const blockW = 220;

  const rows: Array<{ y: number; h: number; label: string; kind: "frozen" | "adapter" | "norm" }> = [
    { y: 30, h: 38, label: "Multi-Head Attention", kind: "frozen" },
    { y: 78, h: 26, label: "+ skip", kind: "norm" },
    { y: 114, h: 30, label: "LayerNorm", kind: "norm" },
    { y: 154, h: 40, label: "Adapter₁", kind: "adapter" },
    { y: 204, h: 38, label: "Feed-Forward", kind: "frozen" },
    { y: 252, h: 26, label: "+ skip", kind: "norm" },
    { y: 288, h: 30, label: "LayerNorm", kind: "norm" },
    { y: 328, h: 40, label: "Adapter₂", kind: "adapter" },
  ];

  const colorOf = (kind: string) =>
    kind === "frozen"
      ? { fill: "#f3f4f6", stroke: "#9ca3af", text: "#6b7280" }
      : kind === "adapter"
        ? { fill: "#fce7f3", stroke: "#ec4899", text: "#be185d" }
        : { fill: "#ffffff", stroke: "#d1d5db", text: "#6b7280" };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Adapter 在 Transformer block 中的插入位置">
      <text x={W / 2} y={16} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--ink-primary)">
        Adapter 插入位置:Attention 后 + FFN 后(每层两次)
      </text>

      {rows.map((row, i) => {
        const c = colorOf(row.kind);
        return (
          <g key={`${row.label}-${i}`}>
            <rect
              x={blockX}
              y={row.y}
              width={blockW}
              height={row.h}
              fill={c.fill}
              stroke={c.stroke}
              strokeWidth={row.kind === "adapter" ? 2 : 1.2}
              rx={6}
            />
            <text x={blockX + blockW / 2} y={row.y + row.h / 2 + 4} textAnchor="middle" fontSize={11} fontWeight={row.kind === "adapter" ? 700 : 500} fill={c.text}>
              {row.label}
            </text>
            {row.kind === "adapter" && (
              <text x={blockX + blockW + 12} y={row.y + row.h / 2 + 4} fontSize={9} fill="#be185d">
                ← 新增
              </text>
            )}
          </g>
        );
      })}

      {/* connecting arrows */}
      {rows.slice(0, -1).map((row, i) => (
        <line
          key={`arrow-${row.label}-${i}`}
          x1={blockX + blockW / 2}
          y1={row.y + row.h}
          x2={blockX + blockW / 2}
          y2={rows[i + 1].y}
          stroke="#9ca3af"
          strokeWidth={1.4}
          markerEnd="url(#arrowhead-adapter)"
        />
      ))}

      <defs>
        <marker id="arrowhead-adapter" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#9ca3af" />
        </marker>
      </defs>

      {/* zoomed bottleneck detail */}
      <g transform="translate(20, 130)">
        <rect x={0} y={0} width={96} height={200} fill="none" stroke="#ec4899" strokeWidth={1} strokeDasharray="3 3" rx={8} />
        <text x={48} y={-8} textAnchor="middle" fontSize={9} fontWeight={700} fill="#be185d">
          Adapter 内部
        </text>
        <text x={48} y={20} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">d = {BOTTLENECK_DIM.d}</text>
        <rect x={16} y={30} width={64} height={22} fill="#fef3c7" stroke="#f59e0b" rx={4} />
        <text x={48} y={45} textAnchor="middle" fontSize={8} fontWeight={600} fill="#92400e">down</text>
        <line x1={48} y1={52} x2={48} y2={70} stroke="#9ca3af" strokeWidth={1.2} markerEnd="url(#arrowhead-adapter)" />
        <text x={48} y={84} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">r = {BOTTLENECK_DIM.r}</text>
        <rect x={16} y={92} width={64} height={22} fill="#dbeafe" stroke="#3b82f6" rx={4} />
        <text x={48} y={107} textAnchor="middle" fontSize={8} fontWeight={600} fill="#1d4ed8">ReLU</text>
        <line x1={48} y1={114} x2={48} y2={132} stroke="#9ca3af" strokeWidth={1.2} markerEnd="url(#arrowhead-adapter)" />
        <rect x={16} y={134} width={64} height={22} fill="#fef3c7" stroke="#f59e0b" rx={4} />
        <text x={48} y={149} textAnchor="middle" fontSize={8} fontWeight={600} fill="#92400e">up</text>
        <text x={48} y={172} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">d = {BOTTLENECK_DIM.d}</text>
        <text x={48} y={190} textAnchor="middle" fontSize={8} fill="#10b981" fontWeight={700}>+ residual</text>
      </g>
    </svg>
  );
}
