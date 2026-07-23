import { NODES, POSITIONS, EDGES, baseScore, shortestPath, spatialBias, edgeBias } from "../lib/data";

interface Props {
  i: number;
  j: number;
  layer: "base" | "spatial" | "spatial+edge";
}

// 复用 Task 9 的 6x6 热力图思路,但这里聚焦单个 (i,j) pair 的分数分解,
// 并在小图上高亮最短路径,展示 base → +spatial → +spatial+edge 三层累加。

export function EdgeBiasHeatmapWidget({ i, j, layer }: Props) {
  const { distance, path } = shortestPath(i, j);
  const base = baseScore(i, j);
  const withSpatial = base + (i === j ? 0 : spatialBias(distance));
  const withEdge = withSpatial + (i === j ? 0 : edgeBias(path));

  const score = layer === "base" ? base : layer === "spatial" ? withSpatial : withEdge;
  const pathEdges = new Set(path.slice(0, -1).map((n, idx) => `${n}-${path[idx + 1]}`));

  const W = 500;
  const H = 300;

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`节点 ${i} 到 ${j} 的 attention bias 分解,当前展示层 ${layer}`}>
        {EDGES.map((e, idx) => {
          const [x1, y1] = POSITIONS[e.a];
          const [x2, y2] = POSITIONS[e.b];
          const onPath = pathEdges.has(`${e.a}-${e.b}`) || pathEdges.has(`${e.b}-${e.a}`);
          return <line key={idx} x1={x1} y1={y1} x2={x2} y2={y2} stroke={onPath ? "#ec4899" : "var(--border)"} strokeWidth={onPath ? 3 : 1.5} />;
        })}
        {NODES.map((n) => {
          const [x, y] = POSITIONS[n];
          const highlight = n === i || n === j;
          return (
            <g key={n}>
              <circle cx={x} cy={y} r={20} fill={highlight ? "#ec4899" : "var(--bg-surface)"} stroke="#ec4899" strokeWidth={2} />
              <text x={x} y={y + 5} textAnchor="middle" fontSize={12} fontWeight={700} fill={highlight ? "#fff" : "var(--ink-primary)"}>
                {n}
              </text>
            </g>
          );
        })}
      </svg>

      <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "var(--space-3)" }}>
        <tbody>
          <tr>
            <td style={{ padding: "4px 8px", fontSize: "var(--fs-sm)" }}>base QK score</td>
            <td style={{ padding: "4px 8px", fontSize: "var(--fs-sm)", textAlign: "right" }}>{base.toFixed(2)}</td>
          </tr>
          <tr>
            <td style={{ padding: "4px 8px", fontSize: "var(--fs-sm)" }}>+ 空间 bias(距离={Number.isFinite(distance) ? distance : "∞"})</td>
            <td style={{ padding: "4px 8px", fontSize: "var(--fs-sm)", textAlign: "right" }}>{withSpatial.toFixed(2)}</td>
          </tr>
          <tr>
            <td style={{ padding: "4px 8px", fontSize: "var(--fs-sm)" }}>+ 边编码(路径上边特征累加)</td>
            <td style={{ padding: "4px 8px", fontSize: "var(--fs-sm)", textAlign: "right" }}>{withEdge.toFixed(2)}</td>
          </tr>
          <tr style={{ borderTop: "1px solid var(--border)" }}>
            <td style={{ padding: "4px 8px", fontSize: "var(--fs-sm)", fontWeight: 700 }}>当前展示层({layer})</td>
            <td style={{ padding: "4px 8px", fontSize: "var(--fs-sm)", fontWeight: 700, textAlign: "right", color: "#9d174d" }}>{score.toFixed(2)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
