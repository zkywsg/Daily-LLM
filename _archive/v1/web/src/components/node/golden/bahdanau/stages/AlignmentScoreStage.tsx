import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { BAHDANAU_SOURCE_PATH } from "../lib/prose";
import { AdditiveScoreDiagram } from "../widgets/AdditiveScoreDiagram";
import { AlignmentHeatmap } from "../widgets/AlignmentHeatmap";
import { ALIGNMENT_DEMO } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

type Highlight = "input" | "tanh" | "softmax" | "ctx" | null;

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function AlignmentScoreStage({ mechanism2Prose }: Props) {
  const [hl, setHl] = useState<Highlight>(null);
  const [tgtIdx, setTgtIdx] = useState<number | null>(null);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Alignment Score — s_{`{t-1}`} 与每个 h_i 算相关性
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        给定 (h_1..h_T) 之后,第 t 步该给哪些 h_i 多大权重?Bahdanau 用一个 MLP
        算 (s_{`{t-1}`}, h_i) 的兼容度 e_{`{t,i}`} = vᵀ tanh(W_s s + W_h h)。
        因为公式内部 W_s·s + W_h·h 是相加,所以叫 additive attention(名字由来)。
        softmax 归一化得 α_t — 一个可学的对齐分布。
      </p>

      <AdditiveScoreDiagram highlight={hl} />
      <p className={styles.caption}>
        ↑ 单步 t 的完整数据流:s_{`{t-1}`} + 3 个 h_i → MLP → 3 个 score → softmax → α_t → c_t。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setHl(null)} style={btnStyle(hl === null)}>全部</button>
        <button type="button" onClick={() => setHl("input")} style={btnStyle(hl === "input")}>① 输入</button>
        <button type="button" onClick={() => setHl("tanh")} style={btnStyle(hl === "tanh")}>② tanh MLP</button>
        <button type="button" onClick={() => setHl("softmax")} style={btnStyle(hl === "softmax")}>③ softmax</button>
        <button type="button" onClick={() => setHl("ctx")} style={btnStyle(hl === "ctx")}>④ c_t</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={BAHDANAU_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <AlignmentHeatmap highlightTgt={tgtIdx} />
          <p className={styles.caption}>
            ↑ 论文 Figure 3 同款对齐热图。注意中间 "European Economic Area" ↔
            "zone économique européenne" 反序对齐 — soft alignment 自动学到法语的形容词后置。
          </p>
          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              聚焦目标词(行)
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
              <button type="button" onClick={() => setTgtIdx(null)} style={btnStyle(tgtIdx === null)}>全部</button>
              {ALIGNMENT_DEMO.tgt.slice(0, 8).map((t, i) => (
                <button key={i} type="button" onClick={() => setTgtIdx(i)} style={btnStyle(tgtIdx === i)}>{t}</button>
              ))}
            </div>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 8, lineHeight: 1.4 }}>
              α 通常稀疏 — 80% 权重集中在 2-3 个源词。这是后来 Transformer
              <code> softmax(QKᵀ/√d) </code> 的直接祖先。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
