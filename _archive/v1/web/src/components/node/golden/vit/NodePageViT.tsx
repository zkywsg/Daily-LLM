import { Link } from "react-router";

import vitMarkdown from "../../../../../../08-vit/01-vit.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, VIT_SOURCE_PATH } from "./lib/prose";
import { PatchEmbeddingStage } from "./stages/PatchEmbeddingStage";
import { ClsTransformerStage } from "./stages/ClsTransformerStage";
import { ScalingStage } from "./stages/ScalingStage";
import styles from "./NodePageViT.module.css";

const prose = extractProse(vitMarkdown);

export default function NodePageViT() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/08-vit" className={styles.back}>
          ← 返回 视觉 Transformer (ViT)
        </Link>
        <h1 className={styles.title}>ViT (2020)</h1>
        <div className={styles.metaLine}>
          作者:Alexey Dosovitskiy · Lucas Beyer · Alexander Kolesnikov · ... · Neil Houlsby · Google Brain
        </div>
        <div className={styles.metaLine}>
          论文:An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale
        </div>
        <p className={styles.keyIdea}>
          把图像切成 16×16 patches 当 token,塞进标准 Transformer encoder
          — 不要卷积、不要 2D 偏置,大数据下反超 CNN
        </p>
      </section>

      <section className={styles.stage}>
        <PatchEmbeddingStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <ClsTransformerStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <ScalingStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={VIT_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={VIT_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={VIT_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={VIT_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
