import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { BERT_SOURCE_PATH } from "../lib/prose";
import { DEMO_SENTENCES } from "../lib/data";
import { MLMInteractive } from "../widgets/MLMInteractive";
import { MaskingStrategyBar } from "../widgets/MaskingStrategyBar";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
  nspProse: string;
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

export function MLMStage({ mechanism2Prose, nspProse }: Props) {
  const [sentenceIdx, setSentenceIdx] = useState(0);
  const [maskedIdx, setMaskedIdx] = useState<number | null>(2);

  const sentence = DEMO_SENTENCES[sentenceIdx];

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Masked Language Modeling (MLM)
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        随机把 15% 输入 token 替换成 [MASK],让模型从双向上下文反推被 mask 的词。
        这是 BERT 唯一的预训练自监督信号 —— 没有人工标注,只靠"填空"学语义。
      </p>

      <MLMInteractive
        sentence={sentence}
        maskedIdx={maskedIdx}
        onMaskChange={setMaskedIdx}
      />
      <p className={styles.caption}>
        点 token 把它换成 [MASK],右下方出现 top-5 候选 + 概率条。BERT 在
        预训练时就反复做这个任务,逼模型学会从前后文反推被遮的词。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={BERT_SOURCE_PATH} />
          </div>

          {nspProse && (
            <>
              <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>
                Next Sentence Prediction (NSP)
              </h3>
              <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
                <MarkdownRenderer markdown={nspProse} sourcePath={BERT_SOURCE_PATH} />
              </div>
            </>
          )}
        </div>

        <div className={styles.stickyPanel}>
          <MaskingStrategyBar />
          <p className={styles.caption}>
            注意被选中的 15% 内部还有三段切分。这一招让 BERT 解决了"训练
            时见 [MASK]、推理时见不到"的分布差异。
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
              示例句子
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {DEMO_SENTENCES.map((s, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => {
                    setSentenceIdx(i);
                    setMaskedIdx(null);
                  }}
                  style={btnStyle(i === sentenceIdx)}
                >
                  {s.tokens.join(" ")}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
