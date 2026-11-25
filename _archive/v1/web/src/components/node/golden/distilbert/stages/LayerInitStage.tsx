import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DISTILBERT_SOURCE_PATH } from "../lib/prose";
import { LayerInitializationDiagram } from "../widgets/LayerInitializationDiagram";
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

export function LayerInitStage({ mechanism3Prose, synergyProse }: Props) {
  const [showMapping, setShowMapping] = useState(true);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:隔层初始化(收敛快 5-10×)
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用 BERT-base 的权重初始化 DistilBERT——把 12 层每隔一层取一层(第 1, 3, 5,
        7, 9, 11 层)作为 DistilBERT 6 层的初始权重。这一 trick 让蒸馏训练收敛快
        得多:从 random init 训要几天,从 BERT 权重初始化只要 1 天。
      </p>

      <LayerInitializationDiagram showMapping={showMapping} />
      <p className={styles.caption}>
        ↑ 切换看"隔层初始化"与"随机初始化"的差异——虚线表示 teacher 层权重直接复制到 student 对应层。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setShowMapping(true)} style={btnStyle(showMapping)}>隔层初始化</button>
        <button type="button" onClick={() => setShowMapping(false)} style={btnStyle(!showMapping)}>随机初始化</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={DISTILBERT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={DISTILBERT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              抽掉任何一件,DistilBERT 都不成立
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>只有三损失蒸馏</strong>:架构和 BERT 一样大,蒸馏后还是 110M 参数——没解决部署问题</li>
              <li><strong>只有层数减半</strong>:从零训 6 层 BERT,没 teacher 软标签信号——GLUE 保留只 85% 左右</li>
              <li><strong>只有隔层初始化</strong>:不蒸馏只用初始化的 6 层——等于 truncated BERT 继续训 MLM,失去蒸馏收益</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
