import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SPARSE_ATTN_SOURCE_PATH } from "../lib/prose";
import { GlobalTokenDiagram } from "../widgets/GlobalTokenDiagram";
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

export function GlobalTokenStage({ mechanism2Prose }: Props) {
  const [strategy, setStrategy] = useState<"task" | "fixed">("task");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Global Token — 少量"信息枢纽"看所有人
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        指定少量特殊位置(g=8-16 个)作为 global token:它们对所有其他位置做 dense
        attention,所有其他位置也 attend 到它们。复杂度 O(N×g)=O(N)。
        每篇文章总有几个关键概念应该能和全文每个位置直接双向交互 —
        没有它们,远端 token 传信息只能靠多层间接,效率极低。
      </p>

      <GlobalTokenDiagram strategy={strategy} />
      <p className={styles.caption}>
        ↑ 切换看两种 global token 选择策略:任务相关(Longformer 默认)vs 位置固定(BigBird)。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setStrategy("task")} style={btnStyle(strategy === "task")}>任务相关</button>
        <button type="button" onClick={() => setStrategy("fixed")} style={btnStyle(strategy === "fixed")}>位置固定</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={SPARSE_ATTN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              两种策略对比
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>任务相关(Longformer)</strong>:分类用 CLS,QA 用整个 question — 关键问题 token 直接看到全文</li>
              <li><strong>位置固定(BigBird)</strong>:每隔 k 个位置选一个 — 不需要任务先验,通用性更强</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
