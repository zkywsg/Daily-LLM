import { capacityOverflow, expertLoad, NUM_EXPERTS } from "../lib/data";

interface Props {
  capacityFactor: number;
}

const W = 700;
const H = 320;

// capacity factor 滑块:每 expert 的 token 容量上限。
// capacity 越低,越多 expert 撑爆 → token 被 drop(走 residual 跳过 MoE)。

export function CapacityFactorDiagram({ capacityFactor }: Props) {
  const loads = expertLoad(false);
  const { capacity, overflowExperts, overflowFrac } = capacityOverflow(capacityFactor);
  const max = Math.max(...loads, capacity, 3.5);

  const PAD = { left: 60, right: 30, top: 56, bottom: 60 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const barW = (innerW / NUM_EXPERTS) * 0.7;
  const gap = (innerW / NUM_EXPERTS) * 0.3;
  const yScale = (v: number) => PAD.top + (1 - v / max) * innerH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`Capacity factor ${capacityFactor}, overflow ${(overflowFrac * 100).toFixed(1)}%`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Capacity Factor = {capacityFactor.toFixed(2)} — 超出容量的 token 被 drop
      </text>
      <text x={W / 2} y={36} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        红色斜线部分 = 超出 capacity 的 token(走 residual 跳过该层 MoE)
      </text>

      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />

      <line x1={PAD.left} x2={W - PAD.right} y1={yScale(capacity)} y2={yScale(capacity)} stroke="#dc2626" strokeWidth={1.5} strokeDasharray="4 3" />
      <text x={W - PAD.right - 4} y={yScale(capacity) - 4} textAnchor="end" fontSize={10} fontStyle="italic" fill="#dc2626">
        capacity = {capacity.toFixed(2)}
      </text>

      <defs>
        <pattern id="overflow-hatch" width={6} height={6} patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
          <rect width={6} height={6} fill="#fecaca" />
          <line x1={0} y1={0} x2={0} y2={6} stroke="#dc2626" strokeWidth={2} />
        </pattern>
      </defs>

      {loads.map((l, i) => {
        const x = PAD.left + i * (barW + gap) + gap / 2;
        const kept = Math.min(l, capacity);
        const overflow = Math.max(0, l - capacity);
        const keptH = (kept / max) * innerH;
        const overflowH = (overflow / max) * innerH;
        const isOver = overflowExperts.includes(i);
        return (
          <g key={i}>
            <rect x={x} y={H - PAD.bottom - keptH} width={barW} height={keptH} rx={3} fill="#f59e0b" opacity={0.85} />
            {overflow > 0 && (
              <rect x={x} y={H - PAD.bottom - keptH - overflowH} width={barW} height={overflowH} rx={2} fill="url(#overflow-hatch)" />
            )}
            <text x={x + barW / 2} y={H - PAD.bottom - keptH - overflowH - 4} textAnchor="middle" fontSize={10} fontWeight={600} fill={isOver ? "#dc2626" : "var(--ink-primary)"}>
              {l.toFixed(1)}
            </text>
            <text x={x + barW / 2} y={H - PAD.bottom + 16} textAnchor="middle" fontSize={10} fontWeight={500} fill="var(--ink-secondary)">
              E{i}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill={overflowExperts.length > 0 ? "#dc2626" : "#10b981"}>
        {overflowExperts.length > 0
          ? `${overflowExperts.length} 个 expert 撑爆 · 约 ${(overflowFrac * 100).toFixed(0)}% 的 token 被 drop`
          : "✓ capacity 足够大 · 没有 token 被 drop"}
      </text>
    </svg>
  );
}
