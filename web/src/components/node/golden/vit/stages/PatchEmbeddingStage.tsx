import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { VIT_SOURCE_PATH } from "../lib/prose";
import { PatchGridSVG } from "../widgets/PatchGridSVG";
import { PatchToTokenFlow } from "../widgets/PatchToTokenFlow";
import { PatchSizeControls } from "../widgets/PatchSizeControls";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function PatchEmbeddingStage({
  intuitionProse,
  mechanism1Prose,
}: Props) {
  const [patchSize, setPatchSize] = useState(2);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Patch Embedding — 把图像切成 token 序列
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        ViT 的关键洞察:既然 Transformer 在文本上做得这么好,直接把图像
        切成 patches、当成 token 序列丢进去就行 —— 不需要卷积、不需要
        2D 归纳偏置。一刀切到底,把 CNN 的 "image" 改成 "句子"。
      </p>

      <PatchGridSVG patchSize={patchSize} hoverIdx={hoverIdx} onHoverIdxChange={setHoverIdx} />
      <p className={styles.caption}>
        ↑ 14×14 玩具 image 被切成 {(14 / patchSize) * (14 / patchSize)} 个 patch token,
        hover 任一 patch 看 image / token 序列里对应高亮。切 1×1 时就是
        "每个像素一个 token";切 14×14 时整张图就 1 个 token。
      </p>

      <PatchToTokenFlow />
      <p className={styles.caption}>
        每个 patch flatten 成 (patch² × 3) 维向量,过 W_e 线性投影到 d_model 维 →
        得到一个 patch token。N 个 patch token + 1 个 CLS + 位置嵌入 → Transformer encoder。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            直觉
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={VIT_SOURCE_PATH} />
          </div>

          <h3
            style={{
              fontSize: "var(--fs-xl)",
              marginTop: "var(--space-6)",
              marginBottom: "var(--space-4)",
            }}
          >
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={VIT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <PatchSizeControls patchSize={patchSize} onPatchSizeChange={setPatchSize} />
        </div>
      </div>
    </div>
  );
}
