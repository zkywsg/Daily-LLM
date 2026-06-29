import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { WORD2VEC_SOURCE_PATH } from "../lib/prose";
import { CbowSkipgramCompare } from "../widgets/CbowSkipgramCompare";
import { ContextWindowSlider } from "../widgets/ContextWindowSlider";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

const SENTENCE = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"];

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function CbowSkipgramStage({ intuitionProse, mechanism1Prose }: Props) {
  const [windowC, setWindowC] = useState(2);
  const [centerIdx, setCenterIdx] = useState(3);
  const [side, setSide] = useState<"cbow" | "skipgram" | "both">("both");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:CBOW 与 Skip-gram — 两个对偶的预测任务
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        把 "用上下文预测中心词" 和 "用中心词预测上下文" 当成训练目标 ——
        其实是同一件事的两种切法。两个任务都不在乎预测准不准,在乎的是
        副产品:训完后 input embedding 矩阵 W 就是词向量。
      </p>

      <CbowSkipgramCompare highlightSide={side} />
      <p className={styles.caption}>
        ↑ 同一个 4 词窗口下两种任务的对偶。CBOW 把 4 个 context 词
        sum/avg 成一个向量 → softmax 预测中心词 "fox";Skip-gram 反过来,
        用 "fox" 单独预测 4 个 context 词。点下方按钮聚焦其中一边。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={WORD2VEC_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={WORD2VEC_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <ContextWindowSlider window={windowC} centerIndex={centerIdx} tokens={SENTENCE} />
          <p className={styles.caption}>
            ↑ 拖动两个 slider 看窗口和中心位置怎么变。每个 (center, ctx) pair
            就是一条训练样本 — Skip-gram 每个 center 产 2c 条,
            CBOW 把 2c 个 ctx 折成 1 条。
          </p>

          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: 4 }}>
              <span>窗口半径 c</span><strong>{windowC}</strong>
            </label>
            <input type="range" min={1} max={4} step={1} value={windowC} onChange={(e) => setWindowC(parseInt(e.target.value))} style={{ width: "100%" }} />

            <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: "8px 0 4px" }}>
              <span>中心词位置</span><strong>{SENTENCE[centerIdx]}</strong>
            </label>
            <input type="range" min={0} max={SENTENCE.length - 1} step={1} value={centerIdx} onChange={(e) => setCenterIdx(parseInt(e.target.value))} style={{ width: "100%" }} />

            <div style={{ marginTop: 10, display: "flex", gap: 6 }}>
              <button type="button" onClick={() => setSide("cbow")} style={btnStyle(side === "cbow")}>聚焦 CBOW</button>
              <button type="button" onClick={() => setSide("skipgram")} style={btnStyle(side === "skipgram")}>聚焦 Skip-gram</button>
              <button type="button" onClick={() => setSide("both")} style={btnStyle(side === "both")}>同时看</button>
            </div>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 8, lineHeight: 1.4 }}>
              工程默认 c=5 / 300d / Skip-gram + NEG。小窗口偏 syntactic,
              大窗口偏 semantic。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
