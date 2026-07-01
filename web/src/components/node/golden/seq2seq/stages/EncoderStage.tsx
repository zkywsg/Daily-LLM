import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SEQ2SEQ_SOURCE_PATH } from "../lib/prose";
import { EncoderCompressPipeline } from "../widgets/EncoderCompressPipeline";
import { BleuVsLengthCurve } from "../widgets/BleuVsLengthCurve";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
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

export function EncoderStage({ intuitionProse, mechanism1Prose }: Props) {
  const [cDim, setCDim] = useState(500);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Encoder — 把任意长输入压成固定向量 c
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        翻译是变长输入到变长输出的任务,MLP/CNN/RNN 都天然做不了。
        Sutskever/Cho 2014 反问:能不能用 RNN 把任意长输入压成一个固定维度向量 c,
        decoder 从 c 起步生成任意长输出?Encoder 读完整个句子,最后一时刻隐状态就是 c —
        典型 500-1000 维,这一压缩是整套架构的关键也是局限。
      </p>

      <EncoderCompressPipeline cDim={cDim} />
      <p className={styles.caption}>
        ↑ Encoder 逐词处理,最后压成固定维度 c。切换维度大小看直觉 — 维度越小瓶颈越明显。
      </p>
      <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
        <button type="button" onClick={() => setCDim(300)} style={btnStyle(cDim === 300)}>300 维</button>
        <button type="button" onClick={() => setCDim(500)} style={btnStyle(cDim === 500)}>500 维</button>
        <button type="button" onClick={() => setCDim(1000)} style={btnStyle(cDim === 1000)}>1000 维</button>
      </div>

      <BleuVsLengthCurve />
      <p className={styles.caption}>
        ↑ Sutskever 论文实测:源句 &lt;20 词 BLEU≈35,70 词跌到 24 以下。
        这就是后来 Bahdanau attention 要解决的"信息瓶颈"。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={SEQ2SEQ_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={SEQ2SEQ_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              公式
            </div>
            <pre style={{ fontSize: 11, lineHeight: 1.6, margin: 0, color: "var(--ink-primary)" }}>{`h_t = LSTM(x_t, h_{t-1})

c = h_T   # 最后一时刻隐状态`}</pre>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              c 可以是最后一层 (h_T, C_T),也可以是所有层拼接 —
              后者信息更丰富但 decoder 接口更复杂。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
