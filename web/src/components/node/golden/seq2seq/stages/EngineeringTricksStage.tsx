import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SEQ2SEQ_SOURCE_PATH } from "../lib/prose";
import { ReverseOrderCompare } from "../widgets/ReverseOrderCompare";
import { TrickProgressionBars } from "../widgets/TrickProgressionBars";
import { TRICK_PROGRESSION } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 10px",
  fontSize: "var(--fs-xs)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function EngineeringTricksStage({ mechanism3Prose, synergyProse }: Props) {
  const [reversed, setReversed] = useState(true);
  const [highlightIdx, setHighlightIdx] = useState(-1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:深层 LSTM + 倒序输入 + Beam Search — Sutskever 的三件工程胜利
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        基本架构 Cho 6 月已发,但 Sutskever 9 月才让 Seq2Seq 在 WMT'14 上超过 SMT。
        差距来自三个工程 trick:4 层深 LSTM(堆深)、倒序输入(缩短梯度路径)、
        beam search 解码(维护 top-k 候选)。三者叠加把 BLEU 从 28 推到 34.8,首次超过 SMT 33.3。
      </p>

      <ReverseOrderCompare reversed={reversed} />
      <p className={styles.caption}>
        ↑ 切换正序/倒序看 x_1 到 c 的距离变化。倒序后第一个词离 c 只有 1 步,梯度路径最短。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setReversed(false)} style={btnStyle(!reversed)}>正序</button>
        <button type="button" onClick={() => setReversed(true)} style={btnStyle(reversed)}>倒序(Sutskever trick)</button>
      </div>

      <TrickProgressionBars highlightIdx={highlightIdx} />
      <p className={styles.caption}>
        ↑ 三件工程 trick 累加效果。点按钮聚焦某一步,虚线是 SMT baseline 33.3。
      </p>
      <div style={{ display: "flex", gap: 4, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setHighlightIdx(-1)} style={btnStyle(highlightIdx === -1)}>全部</button>
        {TRICK_PROGRESSION.map((t, i) => (
          <button key={i} type="button" onClick={() => setHighlightIdx(i)} style={btnStyle(highlightIdx === i)}>
            {t.label.split("(")[0]}
          </button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={SEQ2SEQ_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={SEQ2SEQ_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              三个 trick 各自的命运
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>深层 LSTM</strong>:堆深思想延续至今(Transformer 也堆深)</li>
              <li><strong>倒序输入</strong>:被 Bahdanau attention 淘汰(每个目标词可直接看任意源词)</li>
              <li><strong>Beam search</strong>:沿用至今,今天 LLM 采样仍常配 beam / top-k / top-p</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
