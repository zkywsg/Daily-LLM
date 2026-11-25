import { NODES, baseScore, shortestPath, spatialBias } from "../lib/data";

interface Props {
  withSpatial: boolean;
}

// 6x6 attention score 热力图:withSpatial=false 只显示 base QK score,
// withSpatial=true 显示 base + spatialBias(shortest-path distance)。

export function SpatialBiasHeatmapWidget({ withSpatial }: Props) {
  const scores = NODES.map((i) => NODES.map((j) => {
    const base = baseScore(i, j);
    if (!withSpatial || i === j) return base;
    const { distance } = shortestPath(i, j);
    return base + spatialBias(distance);
  }));

  const all = scores.flat();
  const min = Math.min(...all);
  const max = Math.max(...all);
  const cellSize = 42;

  const colorFor = (v: number) => {
    const t = (v - min) / (max - min || 1);
    const lightness = 90 - t * 55;
    return `hsl(330, 70%, ${lightness}%)`;
  };

  return (
    <div>
      <div
        style={{ display: "inline-block" }}
        role="img"
        aria-label={withSpatial ? "6x6 attention score 热力图,已叠加最短路径空间 bias" : "6x6 attention score 热力图,纯 base QK score"}
      >
        <div aria-hidden="true">
          <div style={{ display: "flex", marginLeft: 32 }}>
            {NODES.map((j) => (
              <div key={j} style={{ width: cellSize, textAlign: "center", fontSize: "var(--fs-xs)", color: "var(--ink-muted)" }}>{j}</div>
            ))}
          </div>
          {NODES.map((i) => (
            <div key={i} style={{ display: "flex", alignItems: "center" }}>
              <div style={{ width: 32, fontSize: "var(--fs-xs)", color: "var(--ink-muted)", textAlign: "right", paddingRight: 4 }}>{i}</div>
              {NODES.map((j) => (
                <div
                  key={j}
                  style={{
                    width: cellSize, height: cellSize, display: "flex", alignItems: "center", justifyContent: "center",
                    background: colorFor(scores[i][j]), fontSize: "var(--fs-xs)", border: "1px solid var(--bg-canvas)",
                  }}
                >
                  {scores[i][j].toFixed(1)}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
      <p style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: "var(--space-2)" }}>
        {withSpatial ? "已叠加最短路径距离的空间 bias —— 远的节点分数被压低" : "纯 base QK score,还没有任何图结构信息"}
      </p>
    </div>
  );
}
