import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { FLASH_ATTN_SOURCE_PATH } from "../lib/prose";
import { OnlineSoftmaxDiagram } from "../widgets/OnlineSoftmaxDiagram";
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

export function OnlineSoftmaxStage({ mechanism2Prose }: Props) {
  const [visibleBlocks, setVisibleBlocks] = useState(1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Online Softmax — 不落盘 N×N 的数学关键
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        普通 softmax 数学上必须看整行才能算 max 和 sum。如果按列 block 切,每次只看
        一段 K,根本不知道全行的 max/sum。FlashAttention 只维护 running max m_i 和
        running sum ℓ_i,每来一个新 block 用一次 rescale 把旧 partial output 更新到
        与新 max 一致 — 任意 block 顺序的最终结果与一次性 softmax 数值完全等价。
      </p>

      <OnlineSoftmaxDiagram visibleBlocks={visibleBlocks} />
      <p className={styles.caption}>
        ↑ 点按钮逐块"喂入"新数据,观察 running max(橙)和 running sum(粉,log 尺度)如何增量更新。
      </p>
      <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
        {[1, 2, 3].map((n) => (
          <button key={n} type="button" onClick={() => setVisibleBlocks(n)} style={btnStyle(visibleBlocks === n)}>
            看到 block 0-{n - 1}
          </button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={FLASH_ATTN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              为什么是 exact attention?
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              online softmax(Milakov 2018)保证任意 block 顺序增量计算的最终结果与
              一次性对整行做 softmax 数值完全等价 — 这是 FlashAttention 不是近似算法、
              而是 exact attention 的数学基础,也是它能被无痛接入所有现代 LLM 的根因。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
