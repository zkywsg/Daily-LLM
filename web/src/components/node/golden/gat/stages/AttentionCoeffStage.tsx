import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GAT_SOURCE_PATH } from "../lib/prose";
import { NODES, rawNeighbors } from "../lib/data";
import { TemperatureSliderWidget } from "../widgets/TemperatureSliderWidget";
import { AttentionBarWidget } from "../widgets/AttentionBarWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function AttentionCoeffStage({ intuitionProse, mechanism1Prose }: Props) {
  const [center, setCenter] = useState(1);
  const [temperature, setTemperature] = useState(1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:自注意力系数计算
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用一个共享的前馈网络对"中心节点 + 邻居"的拼接特征打分,得到每条边的原始 attention logit —— 权重不再是固定的度数归一化系数,而是模型自己学出来的。
      </p>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={GAT_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={GAT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-3)" }}>
            {NODES.filter((n) => rawNeighbors(n).length > 0).map((n) => (
              <button
                key={n} type="button" onClick={() => setCenter(n)} aria-pressed={n === center}
                style={{
                  width: 30, height: 30, borderRadius: "var(--radius-sm)",
                  border: `1px solid ${n === center ? "#ec4899" : "var(--border)"}`,
                  background: n === center ? "#ec4899" : "var(--bg-surface)",
                  color: n === center ? "#fff" : "var(--ink-secondary)", cursor: "pointer",
                }}
              >
                {n}
              </button>
            ))}
          </div>
          <TemperatureSliderWidget temperature={temperature} onChange={setTemperature} />
          <AttentionBarWidget center={center} temperature={temperature} />
        </div>
      </div>
    </div>
  );
}
