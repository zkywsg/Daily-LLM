import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GRAPHORMER_SOURCE_PATH } from "../lib/prose";
import { NodePairSelectorWidget } from "../widgets/NodePairSelectorWidget";
import { EdgeBiasHeatmapWidget } from "../widgets/EdgeBiasHeatmapWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function EdgeStage({ mechanism3Prose, synergyProse }: Props) {
  const [i, setI] = useState(0);
  const [j, setJ] = useState(4);
  const [layer, setLayer] = useState<"base" | "spatial" | "spatial+edge">("base");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:边编码
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        在空间编码之上,把最短路径上每条边自身的特征也编码进 bias —— 不只是"隔多远",还考虑"沿途经过了什么样的边"。三层 bias(base / +spatial / +spatial+edge)逐步叠加。
      </p>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={GRAPHORMER_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={GRAPHORMER_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <NodePairSelectorWidget i={i} j={j} onSelectI={setI} onSelectJ={setJ} />
          <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-3)" }}>
            {(["base", "spatial", "spatial+edge"] as const).map((l) => (
              <button
                key={l} type="button" onClick={() => setLayer(l)} aria-pressed={layer === l}
                style={{
                  padding: "4px 10px", borderRadius: "var(--radius-sm)",
                  border: `1px solid ${layer === l ? "#ec4899" : "var(--border)"}`,
                  background: layer === l ? "#ec4899" : "var(--bg-surface)",
                  color: layer === l ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-xs)",
                }}
              >
                {l}
              </button>
            ))}
          </div>
          <EdgeBiasHeatmapWidget i={i} j={j} layer={layer} />
        </div>
      </div>
    </div>
  );
}
