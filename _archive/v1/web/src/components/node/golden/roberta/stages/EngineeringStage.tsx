import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { ROBERTA_SOURCE_PATH } from "../lib/prose";
import { DynamicMaskingDiagram } from "../widgets/DynamicMaskingDiagram";
import { DataScaleCompareChart } from "../widgets/DataScaleCompareChart";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function EngineeringStage({ mechanism3Prose, synergyProse }: Props) {
  const [mode, setMode] = useState<"static" | "dynamic">("static");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:动态 Masking + 大 Batch + Byte-level BPE
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        BERT 的 masking 是静态的 — 预处理阶段每个样本 mask 一次,整个训练过程重复
        使用同一版本。RoBERTa 改成动态 masking,每次喂给模型时重新随机选 15%
        token,同一句子在不同 epoch 呈现不同的预测任务。配合大 batch(256→8K)
        和 10× 数据规模,三者共同构成训练 recipe 的工程优化部分。
      </p>

      <DynamicMaskingDiagram mode={mode} />
      <p className={styles.caption}>
        ↑ 切换看静态 vs 动态 masking 在 3 个 epoch 里 mask 位置的变化。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setMode("static")} style={btnStyle(mode === "static")}>静态(BERT)</button>
        <button type="button" onClick={() => setMode("dynamic")} style={btnStyle(mode === "dynamic")}>动态(RoBERTa)</button>
      </div>

      <div style={{ marginTop: "var(--space-8)" }}>
        <DataScaleCompareChart />
        <p className={styles.caption}>
          ↑ 数据 / token / batch 规模对比 — 架构完全不变,单纯规模差距就撑起 GLUE +5 分。
        </p>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={ROBERTA_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={ROBERTA_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              抽掉任何一件,涨幅打折
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>只有数据 + 训练步,NSP 还在</strong>:占 50% 算力学浅层信号,涨幅打 6 折</li>
              <li><strong>只有去 NSP,数据/步数没加</strong>:模型仍 under-trained,涨幅 &lt; 1 分</li>
              <li><strong>只有工程优化,数据/任务不变</strong>:byte-BPE + 动态 mask + 大 batch 共贡献约 1 分,撑不起 5 分涨幅</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
