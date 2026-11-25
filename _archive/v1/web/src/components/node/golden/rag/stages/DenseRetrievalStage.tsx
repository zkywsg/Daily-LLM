import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { RAG_SOURCE_PATH } from "../lib/prose";
import { QUERIES } from "../lib/data";
import { CorpusEmbeddingMap } from "../widgets/CorpusEmbeddingMap";
import { TopKList } from "../widgets/TopKList";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
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

export function DenseRetrievalStage({ intuitionProse, mechanism1Prose }: Props) {
  const [queryIdx, setQueryIdx] = useState(0);
  const [k, setK] = useState(3);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Dense Retrieval — embedding 做语义检索
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        BM25 关键词检索找的是"字面匹配",问"光合作用化学式"找不到
        只写"植物用阳光合成糖"的文档。Dense Retrieval 用一个 encoder
        把 query 和 docs 都映射到同一向量空间,然后用 cosine sim 找最近 ——
        靠语义不靠字符串。
      </p>

      <CorpusEmbeddingMap queryIdx={queryIdx} k={k} />
      <p className={styles.caption}>
        ↑ 10 个 doc 按 group 着色聚成 cluster。粉色五角星是 query 投影位置,
        虚线指向召回的 top-{k}。注意 query 落在哪个 cluster 附近,
        就召回哪个 cluster 的文档 —— 这是\"语义检索\"的视觉证据。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            直觉
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={RAG_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={RAG_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <TopKList queryIdx={queryIdx} k={k} />
          <p className={styles.caption}>
            Top-{k} 召回列表:绿条 = labeler 标记的相关文档,灰条 = 同 cluster
            的近邻召回(也合理,但不是直接答案)。
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
              max={6}
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
