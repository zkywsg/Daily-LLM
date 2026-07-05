import { ROUTING_COMPARE } from "../lib/data";

interface Props {
  selectedK: number;
}

const W = 700;
const H = 300;

// Top-1 (Switch) vs Top-2 (Mixtral) vs Top-4 (Shazeer 2017) 的算力/通信对比柱状图。

export function RoutingCostCompare({ selectedK }: Props) {
  const PAD = { left: 60, right: 40, top: 50, bottom: 90 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const barW = (innerW / ROUTING_COMPARE.length) * 0.5;
  const slot = innerW / ROUTING_COMPARE.length;
  const maxCost = 4.5;
  const yScale = (v: number) => PAD.top + (1 - v / maxCost) * innerH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`Routing compute/communication cost, selected k=${selectedK}`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        路由算力 / 通信成本 — Top-K 越大越贵
      </text>

      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />

      {[0, 1, 2, 3, 4].map((v) => (
        <g key={v}>
          <text x={PAD.left - 6} y={yScale(v) + 4} textAnchor="end" fontSize={10} fill="var(--ink-muted)">
            {v}×
          </text>
          <line x1={PAD.left} x2={W - PAD.right} y1={yScale(v)} y2={yScale(v)} stroke="var(--border)" strokeDasharray="1 4" />
        </g>
      ))}

      {ROUTING_COMPARE.map((row, i) => {
        const x = PAD.left + i * slot + slot / 2;
        const isCur = row.k === selectedK;
        const barH = (row.computeCostX / maxCost) * innerH;
        return (
          <g key={row.k}>
            <rect
              x={x - barW / 2}
              y={H - PAD.bottom - barH}
              width={barW}
              height={barH}
              rx={3}
              fill={isCur ? "#f59e0b" : "#fef3c7"}
              stroke={isCur ? "#92400e" : "#f59e0b"}
              strokeWidth={isCur ? 2 : 1}
            />
            <text x={x} y={H - PAD.bottom - barH - 6} textAnchor="middle" fontSize={11} fontWeight={700} fill={isCur ? "#92400e" : "var(--ink-primary)"}>
              {row.computeCostX.toFixed(0)}×
            </text>
            <text x={x} y={H - PAD.bottom + 18} textAnchor="middle" fontSize={11} fontWeight={isCur ? 700 : 500} fill="var(--ink-primary)">
              top-{row.k}
            </text>
            <text x={x} y={H - PAD.bottom + 32} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
              {row.label}
            </text>
          </g>
        );
      })}

      <rect x={20} y={H - 58} width={W - 40} height={48} rx={4} fill="#fef3c7" stroke="#f59e0b" />
      <text x={32} y={H - 38} fontSize={11} fontWeight={700} fill="#92400e">
        top-{selectedK}:
      </text>
      <text x={90} y={H - 38} fontSize={11} fill="var(--ink-primary)">
        {ROUTING_COMPARE.find((r) => r.k === selectedK)?.note ?? "—"}
      </text>
    </svg>
  );
}
