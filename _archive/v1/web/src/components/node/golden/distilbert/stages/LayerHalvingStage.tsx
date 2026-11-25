import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DISTILBERT_SOURCE_PATH } from "../lib/prose";
import { ARCH_COMPARE } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function LayerHalvingStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:层数减半(12 → 6,真省 FLOPs)
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        DistilBERT 的架构和 BERT-base 一致(d_model=768,h=12,d_ff=3072),但层数
        从 12 减到 6——这是和 ALBERT 参数共享的关键区别:ALBERT 省参数但 forward
        仍要走满 24 层,DistilBERT 层数真的减半,FLOPs 真的减半。
      </p>

      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "var(--fs-sm)" }}>
          <thead>
            <tr style={{ borderBottom: "2px solid var(--border)" }}>
              <th style={{ textAlign: "left", padding: "8px 12px", color: "var(--ink-muted)" }}>维度</th>
              <th style={{ textAlign: "left", padding: "8px 12px", color: "#3b82f6" }}>BERT-base</th>
              <th style={{ textAlign: "left", padding: "8px 12px", color: "#10b981" }}>DistilBERT</th>
            </tr>
          </thead>
          <tbody>
            {ARCH_COMPARE.map((row) => (
              <tr key={row.dim} style={{ borderBottom: "1px solid var(--border)" }}>
                <td style={{ padding: "8px 12px", fontWeight: 600 }}>{row.dim}</td>
                <td style={{ padding: "8px 12px", color: "var(--ink-secondary)" }}>{row.bert}</td>
                <td style={{ padding: "8px 12px", color: "var(--ink-primary)", fontWeight: 700 }}>{row.distilbert}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className={styles.caption}>
        ↑ 除层数外架构与 BERT-base 完全一致——d_model / heads / d_ff 都不变,只是层数减半带来近线性的推理加速。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={DISTILBERT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              为什么减层数而不是减 hidden size?
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              消融显示:减层数,推理时间近线性减小、效果略损;减 hidden size,推理
              时间近平方减小但效果损失更大(attention 表达力被压缩)。"6 层 768 维"
              是 latency 与 quality 的最佳平衡点,也是工业界最受欢迎的配置。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
