import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { VIT_SOURCE_PATH } from "../lib/prose";
import { VitArchFlow } from "../widgets/VitArchFlow";
import { ClsAttentionHeatmap } from "../widgets/ClsAttentionHeatmap";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
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

export function ClsTransformerStage({ mechanism2Prose }: Props) {
  const [patchSize, setPatchSize] = useState(2);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:CLS Token + 标准 Transformer Encoder
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        从 BERT 借来的 [CLS] 设计:在 patch token 序列最前面塞一个学得的
        分类占位 token。它跟所有 patch 做 self-attention,聚合全局信息;
        最后只取它的输出过 MLP head 出 logits。其余 encoder 完全是标准
        Transformer,没有任何 vision-specific 修改。
      </p>

      <VitArchFlow />
      <p className={styles.caption}>
        ↑ 整条 pipeline:N+1 个输入 token(CLS + patch) → 加位置嵌入 →
        N 层标准 encoder → 取 CLS 输出 → MLP head → 类别 logits。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={VIT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <ClsAttentionHeatmap patchSize={patchSize} />
          <p className={styles.caption}>
            CLS 对各 patch 的 attention 强度分布 —— 训练好的 ViT 自动把
            "中心 object" 区域的注意力拉高、"边角背景" 拉低,等价于学会了
            object localization(注意 ViT 没有任何位置先验)。
          </p>
          <div
            style={{
              marginTop: "var(--space-4)",
              padding: "var(--space-3)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-md)",
              background: "var(--bg-surface)",
            }}
          >
            <div
              style={{
                fontSize: "var(--fs-xs)",
                color: "var(--ink-muted)",
                marginBottom: 6,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
              }}
            >
              patch 粒度
            </div>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {[1, 2, 7].map((p) => (
                <button key={p} type="button" onClick={() => setPatchSize(p)} style={btnStyle(patchSize === p)}>
                  {p}×{p}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
