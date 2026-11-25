import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { CLIP_SOURCE_PATH } from "../lib/prose";
import { ContrastiveMatrix } from "../widgets/ContrastiveMatrix";
import { DualSoftmax } from "../widgets/DualSoftmax";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function ContrastiveStage({ mechanism2Prose }: Props) {
  const [batchSize, setBatchSize] = useState(6);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Contrastive Loss — N×N 矩阵里"对角线对,其它对错"
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        每 batch 里 N 张图 × N 个 caption 得 N² 对相似度。其中真正配对的 N 个
        在对角线上,剩下 N²−N 都是负样本。训练就是让对角线最大、其它最小 ——
        N 越大,负样本质量越好,这是 CLIP 用 32K batch 的原因。
      </p>

      <ContrastiveMatrix batchSize={batchSize} hoverIdx={hoverIdx} onHoverIdxChange={setHoverIdx} />

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={CLIP_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <DualSoftmax />
          <p className={styles.caption}>
            最终 loss 是双向 CE 的平均:行 softmax(图找文)+ 列 softmax(文找图)。
            对称设计让 image 和 text encoder 都学到对齐方向。
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
            <label
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "var(--fs-sm)",
                color: "var(--ink-secondary)",
                marginBottom: 4,
              }}
            >
              <span>batch 大小 N</span>
              <strong>{batchSize} (负样本 {batchSize * batchSize - batchSize} 个)</strong>
            </label>
            <input
              type="range"
              min={2}
              max={6}
              step={1}
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value, 10))}
              style={{ width: "100%" }}
            />
            <div
              style={{
                fontSize: "var(--fs-xs)",
                color: "var(--ink-muted)",
                marginTop: 4,
                lineHeight: 1.4,
              }}
            >
              原论文 N=32,768 → 一次 forward 看到 32K 个负样本,这是 SimCLR、MoCo
              这类对比学习走规模化的标准配方。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
