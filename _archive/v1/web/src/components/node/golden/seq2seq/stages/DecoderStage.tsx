import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SEQ2SEQ_SOURCE_PATH } from "../lib/prose";
import { DecoderAutoregressive } from "../widgets/DecoderAutoregressive";
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

export function DecoderStage({ mechanism2Prose }: Props) {
  const [teacherForcing, setTeacherForcing] = useState(true);
  const [errorStep, setErrorStep] = useState(1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Decoder + 自回归生成 — 从 c 起步逐字预测
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Decoder 也是 RNN,初始状态是 c,每步根据上一时刻输出 y_{"{t-1}"} 生成下一个词,
        直到 &lt;EOS&gt;。训练时用 teacher forcing(输入用 ground truth)让训练稳定 —
        否则一步错就全错,梯度信号几乎学不到东西。代价是训练/推理分布不一致(exposure bias)。
      </p>

      <DecoderAutoregressive teacherForcing={teacherForcing} errorAtStep={teacherForcing ? -1 : errorStep} />
      <p className={styles.caption}>
        ↑ 切换 teacher forcing 看训练 vs 推理的差异。推理时若某步预测错误(粉色 "?"),
        后续所有输入都基于这个错误延续 — 误差累积。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setTeacherForcing(true)} style={btnStyle(teacherForcing)}>Teacher Forcing(训练)</button>
        <button type="button" onClick={() => setTeacherForcing(false)} style={btnStyle(!teacherForcing)}>自由生成(推理)</button>
      </div>
      {!teacherForcing && (
        <div style={{ display: "flex", gap: 4, marginTop: 8 }}>
          {[0, 1, 2].map((s) => (
            <button key={s} type="button" onClick={() => setErrorStep(s)} style={btnStyle(errorStep === s)}>
              第 {s + 1} 步出错
            </button>
          ))}
        </div>
      )}

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={SEQ2SEQ_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              公式
            </div>
            <pre style={{ fontSize: 11, lineHeight: 1.6, margin: 0, color: "var(--ink-primary)" }}>{`s_t = LSTM(y_{t-1}, s_{t-1}),  s_0 = c

p(y_t | y_{<t}, x) = softmax(W_o s_t)

L = -Σ_t log p(y_t | y_{<t}, x)`}</pre>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              这种"逐字预测 + 上一步输出作输入"的自回归方式后来成为
              所有语言模型(GPT/BERT decoder/Transformer)的标准生成方式。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
