import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { RAG_SOURCE_PATH } from "../lib/prose";
import { QUERIES } from "../lib/data";
import { ContextStitching } from "../widgets/ContextStitching";
import { AnswerCompare } from "../widgets/AnswerCompare";
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

export function ContextAugStage({ mechanism2Prose }: Props) {
  const [queryIdx, setQueryIdx] = useState(0);
  const [k, setK] = useState(3);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Context Augmentation — 检索结果拼进 prompt
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        把召回的 top-k chunks 按一个固定模板拼到 prompt 里,LLM 就能 \"看着\"
        这些资料回答 —— 不需要把答案记在参数里。这把 LLM 从\"记忆库\"切换到
        \"理解 + 综合\"的工作模式,事实准确性 / 时效性 / 私有知识全解决。
      </p>

      <ContextStitching queryIdx={queryIdx} k={k} />
      <p className={styles.caption}>
        ↑ 这是最终送给 LLM 的 prompt 长什么样:system 指令 + N 个 retrieved chunks +
        user question。调 query / top-k 看 prompt 实时重组。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={RAG_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <AnswerCompare queryIdx={queryIdx} />
          <p className={styles.caption}>
            没 RAG 时 LLM 只能用训练时记住的内容,新鲜或私有知识全部模糊或错;
            有 RAG 时模型把 chunks 当作\"参考资料\",答案准确且能 attribute。
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
              查询
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: "var(--space-3)" }}>
              {QUERIES.map((q, i) => (
                <button key={i} type="button" onClick={() => setQueryIdx(i)} style={btnStyle(i === queryIdx)}>
                  {q.text}
                </button>
              ))}
            </div>
            <label
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "var(--fs-sm)",
                color: "var(--ink-secondary)",
                marginBottom: 4,
              }}
            >
              <span>top-k</span>
              <strong>{k}</strong>
            </label>
            <input
              type="range"
              min={1}
              max={5}
              step={1}
              value={k}
              onChange={(e) => setK(parseInt(e.target.value, 10))}
              style={{ width: "100%" }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
