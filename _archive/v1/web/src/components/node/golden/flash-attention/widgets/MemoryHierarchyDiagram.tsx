import { MEMORY_HIERARCHY } from "../lib/data";

const W = 700;
const H = 230;

export function MemoryHierarchyDiagram() {
  const maxBw = 19;
  const barMaxW = 380;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="SRAM vs HBM 内存层级对比">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        A100 内存层级 — SRAM 快 10× 但容量小 2000×
      </text>

      {MEMORY_HIERARCHY.map((tier, i) => {
        const y = 50 + i * 80;
        const barW = (tier.bandwidthTBs / maxBw) * barMaxW;
        const color = tier.tier === "SRAM" ? "#10b981" : "#3b82f6";
        const bg = tier.tier === "SRAM" ? "#ecfdf5" : "#dbeafe";
        return (
          <g key={tier.tier}>
            <text x={40} y={y + 14} fontSize={13} fontWeight={700} fill={color}>{tier.tier}</text>
            <text x={40} y={y + 30} fontSize={9} fill="var(--ink-muted)">容量 {tier.capacity}</text>

            <rect x={140} y={y} width={barMaxW} height={26} fill={bg} stroke={color} strokeWidth={1} rx={4} />
            <rect x={140} y={y} width={barW} height={26} fill={color} opacity={0.85} rx={4} />
            <text x={140 + barW + 8} y={y + 17} fontSize={11} fontWeight={700} fill={color}>
              {tier.bandwidthTBs} TB/s
            </text>

            <text x={140} y={y + 42} fontSize={9} fill="var(--ink-muted)">延迟 ~{tier.latencyNs} ns</text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        朴素 attention 把 N×N 矩阵反复写读 HBM,计算单元大部分时间在等内存,GPU 利用率仅 20-30%
      </text>
    </svg>
  );
}
