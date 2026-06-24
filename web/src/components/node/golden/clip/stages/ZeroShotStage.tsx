import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { CLIP_SOURCE_PATH } from "../lib/prose";
import { ZERO_SHOT_EXAMPLES } from "../lib/data";
import { ZeroShotClassifier } from "../widgets/ZeroShotClassifier";
import { PromptEnsembleBar } from "../widgets/PromptEnsembleBar";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "6px 14px",
  fontSize: "var(--fs-md)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function ZeroShotStage({ mechanism3Prose, synergyProse }: Props) {
  const [exampleIdx, setExampleIdx] = useState(0);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Zero-Shot Classifier — 把"类别"变成"a photo of a &#123;class&#125;"
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        训练好的 CLIP 不需要 fine-tune 就能做任意分类任务 —— 把每个候选类别
        套进 prompt template,过 text encoder 得 N 个文本 emb,跟 image emb
        算 cos sim 取 argmax。换分类任务只需改 prompt,不动模型。
      </p>

      <ZeroShotClassifier exampleIdx={exampleIdx} />
      <p className={styles.caption}>
        ↑ 选不同 emoji 看 CLIP 的 zero-shot 推理过程。最高 cos sim 的 prompt
        就是预测类别(绿色高亮)。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={CLIP_SOURCE_PATH} />
          </div>

          <h3
            style={{
              fontSize: "var(--fs-xl)",
              marginTop: "var(--space-8)",
              marginBottom: "var(--space-4)",
            }}
          >
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={CLIP_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <PromptEnsembleBar />
          <p className={styles.caption}>
            同一模型只换 prompt 就能差 5-10 个点 —— CLIP 论文里专门写了
            §3.1.4 prompt engineering。后来 LLM 时代"prompt 工程"的源头就是这里。
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
              选输入 image
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              {ZERO_SHOT_EXAMPLES.map((ex, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => setExampleIdx(i)}
                  style={{ ...btnStyle(i === exampleIdx), fontSize: 24, padding: "4px 14px" }}
                  aria-label={`Example ${ex.trueLabel}`}
                >
                  {ex.emoji}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
