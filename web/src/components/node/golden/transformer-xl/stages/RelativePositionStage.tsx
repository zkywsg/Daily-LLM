import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { TRANSFORMER_XL_SOURCE_PATH } from "../lib/prose";
import { RelativePositionDiagram } from "../widgets/RelativePositionDiagram";
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

export function RelativePositionStage({ mechanism2Prose }: Props) {
  const [mode, setMode] = useState<"absolute" | "relative">("absolute");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Relative Position Encoding — 用相对距离替代绝对 PE
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Segment-level recurrence 立刻引入一个新问题:绝对 PE 跨段重复——第 1 段
        位置 5 的 PE 和第 2 段位置 5 的 PE 完全相同,attention 无法区分两个不同
        语境。Dai 把位置信号从"加在输入 embedding 上"改造成"加在 attention
        score 上",只依赖 query/key 的相对距离 i−j,跨段无歧义。
      </p>

      <RelativePositionDiagram mode={mode} />
      <p className={styles.caption}>
        ↑ 切换看绝对 PE 的跨段歧义问题,以及相对 PE 如何从结构上避免它。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setMode("absolute")} style={btnStyle(mode === "absolute")}>绝对 PE(有歧义)</button>
        <button type="button" onClick={() => setMode("relative")} style={btnStyle(mode === "relative")}>相对 PE(无歧义)</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={TRANSFORMER_XL_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              相对 PE attention score 四项展开
            </div>
            <pre style={{ fontSize: 10, lineHeight: 1.6, margin: 0, color: "var(--ink-primary)", overflow: "auto" }}>{`A_ij = Ex_i^T Wq^T Wk,E Ex_j     (内容-内容)
     + Ex_i^T Wq^T Wk,R R_{i-j}   (内容-位置)
     + u^T Wk,E Ex_j              (全局内容偏置)
     + v^T Wk,R R_{i-j}           (全局位置偏置)`}</pre>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              R_{"{i-j}"} 替代 U_j,u/v 是可学的全局偏置,W_k,E / W_k,R 内容和位置用独立 K 投影,互不干扰。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
