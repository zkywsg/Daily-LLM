import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GCN_SOURCE_PATH } from "../lib/prose";
import { GraphSelfLoopWidget } from "../widgets/GraphSelfLoopWidget";
import { AdjacencyMatrixWidget } from "../widgets/AdjacencyMatrixWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function SelfLoopStage({ intuitionProse, mechanism1Prose }: Props) {
  const [withSelfLoop, setWithSelfLoop] = useState(false);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:重整化技巧 — Ã = A + I
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        给邻接矩阵加上单位矩阵,让每个节点在聚合时也把自己的特征算进去 —— 否则每一层传播都会把节点自己的信息完全丢掉,只剩邻居信息。
      </p>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={GCN_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={GCN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <button
            type="button"
            onClick={() => setWithSelfLoop((v) => !v)}
            style={{
              padding: "6px 16px", borderRadius: "var(--radius-sm)",
              border: `1px solid ${withSelfLoop ? "#ec4899" : "var(--border)"}`,
              background: withSelfLoop ? "#ec4899" : "var(--bg-surface)",
              color: withSelfLoop ? "#fff" : "var(--ink-secondary)",
              cursor: "pointer", marginBottom: "var(--space-4)",
            }}
          >
            {withSelfLoop ? "✓ 已加自环 Ã = A + I" : "点击加自环"}
          </button>
          <GraphSelfLoopWidget withSelfLoop={withSelfLoop} />
          <p className={styles.caption}>↑ 图结构与每个节点度数的变化</p>
          <div style={{ marginTop: "var(--space-6)" }}>
            <AdjacencyMatrixWidget withSelfLoop={withSelfLoop} />
          </div>
          <p className={styles.caption}>↑ 6×6 邻接矩阵,对角线(粉色)代表自环</p>
        </div>
      </div>
    </div>
  );
}
