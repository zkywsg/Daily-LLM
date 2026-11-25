import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { FASTTEXT_SOURCE_PATH } from "../lib/prose";
import { HashBucketDiagram } from "../widgets/HashBucketDiagram";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

const BUCKET_OPTIONS = [4, 6, 8, 12];

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 10px",
  fontSize: "var(--fs-xs)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function HashingStage({ mechanism3Prose, synergyProse }: Props) {
  const [buckets, setBuckets] = useState(6);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Hashing Trick — 控制 subword 词表爆炸
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        词表 50K,每个词约 10-15 个 subword,理论 subword 总数可达数百万到上千万 —
        直接为每个 subword 学一个向量,显存爆炸。FastText 用 hashing trick:设 B 个
        bucket(论文 B=2,000,000),每个 subword 经 hash 函数映射到某个 bucket,
        同 bucket 内的 subword 共享同一向量。
      </p>

      <HashBucketDiagram buckets={buckets} />
      <p className={styles.caption}>
        ↑ bucket 数越小碰撞越明显。拖动/点选看 bucket 数变化时碰撞率如何变化
        (论文 B=2M 时碰撞对精度影响 &lt; 1%)。
      </p>
      <div style={{ display: "flex", gap: 4, marginTop: 8, flexWrap: "wrap" }}>
        {BUCKET_OPTIONS.map((b) => (
          <button key={b} type="button" onClick={() => setBuckets(b)} style={btnStyle(buckets === b)}>{b} bucket</button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={FASTTEXT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={FASTTEXT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              抽掉任何一件,FastText 都不成立
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>只有 n-gram 分解,没求和</strong>:退化成 Word2Vec,subword 优势全部消失</li>
              <li><strong>只有求和,没多尺度 n</strong>:n=3 噪声大、n=6 接近词级,多尺度才能兼顾细粒度与整词语义</li>
              <li><strong>没有 hashing trick</strong>:subword 词表上千万,显存放不下,工程上跑不起来</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
