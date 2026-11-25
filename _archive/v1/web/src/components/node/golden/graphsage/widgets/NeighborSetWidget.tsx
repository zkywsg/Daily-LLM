import { EDGES, NODES, POSITIONS } from "../lib/data";

interface Props {
  center: number;
  sampled: number[];
}

const W = 700;
const H = 380;

// 高亮中心节点的全部邻居(灰色描边)vs 被采样到的子集(粉色实心)。

export function NeighborSetWidget({ center, sampled }: Props) {
  const sampledSet = new Set(sampled);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`节点 ${center} 的邻居采样`}>
      {EDGES.map((e, idx) => {
        const [x1, y1] = POSITIONS[e.a];
        const [x2, y2] = POSITIONS[e.b];
        const touchesCenter = e.a === center || e.b === center;
        const other = e.a === center ? e.b : e.a;
        const active = touchesCenter && sampledSet.has(other);
        return <line key={idx} x1={x1} y1={y1} x2={x2} y2={y2} stroke={active ? "#ec4899" : "var(--border)"} strokeWidth={active ? 2.5 : 1.5} />;
      })}

      {NODES.map((n) => {
        const [x, y] = POSITIONS[n];
        const isCenter = n === center;
        const isNeighbor = EDGES.some((e) => (e.a === center && e.b === n) || (e.b === center && e.a === n));
        const isSampled = sampledSet.has(n);
        const fill = isCenter ? "#ec4899" : isSampled ? "#fce7f3" : "var(--bg-surface)";
        const stroke = isCenter || isSampled ? "#ec4899" : isNeighbor ? "var(--ink-muted)" : "var(--border)";
        return (
          <g key={n}>
            <circle cx={x} cy={y} r={isCenter ? 24 : 20} fill={fill} stroke={stroke} strokeWidth={2} strokeDasharray={isNeighbor && !isSampled ? "3 2" : undefined} />
            <text x={x} y={y + 5} textAnchor="middle" fontSize={12} fontWeight={700} fill={isCenter ? "#fff" : "var(--ink-primary)"}>
              {n}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        虚线描边 = 未被采样到的邻居(本轮聚合完全不参与运算)
      </text>
    </svg>
  );
}
