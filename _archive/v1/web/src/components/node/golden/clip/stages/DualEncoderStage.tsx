import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { CLIP_SOURCE_PATH } from "../lib/prose";
import { DualEncoderFlow } from "../widgets/DualEncoderFlow";
import { SharedSpaceScatter } from "../widgets/SharedSpaceScatter";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function DualEncoderStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Dual Encoder — 图像 / 文本各一个 encoder
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        CLIP 不学"图像→类别"分类器,而是学"图像→512 维向量"和"文本→512 维向量"
        两个独立 encoder,然后强迫两个空间对齐 —— 这样推理时把任意类别变成文本就能算相似度。
      </p>

      <DualEncoderFlow />
      <p className={styles.caption}>
        ↑ 两塔独立,只有最后的 Linear 投影把维度对齐到 d_emb=512,然后 L2-normalize。
        归一化后余弦相似度退化成点积 —— 后续 contrastive loss 算起来便宜。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            直觉
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={CLIP_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={CLIP_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <SharedSpaceScatter />
          <p className={styles.caption}>
            训练后图文对在共享空间里聚得很近(虚线连接的两个点)。
            同语义类别(动物 / 交通工具 / 食物)进一步聚成大簇 —— 这是 zero-shot
            分类能 work 的基础。
          </p>
        </div>
      </div>
    </div>
  );
}
