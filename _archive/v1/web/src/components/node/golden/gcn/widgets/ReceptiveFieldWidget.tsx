import { EDGES, NODES, POSITIONS, reachableWithinHops } from "../lib/data";

interface Props {
  center: number;
  hops: number;
}

const W = 680;
const H = 360;

// 高亮中心节点在 L 层堆叠后感受野覆盖到的节点集合:
// L=1 只覆盖直接邻居,L=2 能扩展到邻居的邻居。

export function ReceptiveFieldWidget({ center, hops }: Props) {
  const reached = reachableWithinHops(center, hops);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`节点 ${center} 在 ${hops} 层 GCN 下的感受野`}>
      <text x={W / 2} y={24} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        节点 {center} 的感受野(L = {hops} 层)
      </text>

      {EDGES.map((e, idx) => {
        const [x1, y1] = POSITIONS[e.a];
        const [x2, y2] = POSITIONS[e.b];
        const active = reached.has(e.a) && reached.has(e.b);
        return <line key={idx} x1={x1} y1={y1} x2={x2} y2={y2} stroke={active ? "#ec4899" : "var(--border)"} strokeWidth={active ? 2.5 : 1.5} />;
      })}

      {NODES.map((n) => {
        const [x, y] = POSITIONS[n];
        const active = reached.has(n);
        const isCenter = n === center;
        return (
          <g key={n}>
            <circle cx={x} cy={y} r={isCenter ? 26 : 22} fill={active ? (isCenter ? "#ec4899" : "#fce7f3") : "var(--bg-surface)"} stroke={active ? "#ec4899" : "var(--border)"} strokeWidth={2} />
            <text x={x} y={y + 5} textAnchor="middle" fontSize={13} fontWeight={700} fill={isCenter ? "#fff" : "var(--ink-primary)"}>
              {n}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        粉色节点/边 = 这一层堆叠后节点 {center} 的特征里已经包含的信息来源
      </text>
    </svg>
  );
}
