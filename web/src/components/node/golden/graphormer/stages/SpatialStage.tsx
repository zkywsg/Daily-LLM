import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GRAPHORMER_SOURCE_PATH } from "../lib/prose";
import { NodePairSelectorWidget } from "../widgets/NodePairSelectorWidget";
import { SpatialBiasHeatmapWidget } from "../widgets/SpatialBiasHeatmapWidget";
import { shortestPath } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function SpatialStage({ mechanism2Prose }: Props) {
  const [i, setI] = useState(0);
  const [j, setJ] = useState(4);
  const [withSpatial, setWithSpatial] = useState(false);
  const { distance } = shortestPath(i, j);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:空间编码 — 核心创新
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        计算任意两个节点间的最短路径距离,把这个距离映射成一个 bias 项,直接加到 attention score 上:A_ij = QK^T/√d + b_φ(i,j)。距离越远,bias 越负,注意力天然衰减。
      </p>

      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={GRAPHORMER_SOURCE_PATH} />
        </div>

        <div className={styles.stickyPanel}>
          <NodePairSelectorWidget i={i} j={j} onSelectI={setI} onSelectJ={setJ} />
          <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
            节点 {i} → 节点 {j} 最短路径距离 = {Number.isFinite(distance) ? distance : "不可达"}
          </p>
          <button
            type="button" onClick={() => setWithSpatial((v) => !v)} aria-pressed={withSpatial}
            style={{
              padding: "4px 14px", borderRadius: "var(--radius-sm)", marginBottom: "var(--space-3)",
              border: `1px solid ${withSpatial ? "#ec4899" : "var(--border)"}`,
              background: withSpatial ? "#ec4899" : "var(--bg-surface)",
              color: withSpatial ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-sm)",
            }}
          >
            {withSpatial ? "✓ 已叠加空间 bias" : "叠加空间 bias"}
          </button>
          <SpatialBiasHeatmapWidget withSpatial={withSpatial} />
        </div>
      </div>
    </div>
  );
}
