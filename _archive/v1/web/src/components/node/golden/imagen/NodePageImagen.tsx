import { Link } from "react-router";

import imagenMarkdown from "../../../../../../10-diffusion/03-imagen.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, IMAGEN_SOURCE_PATH } from "./lib/prose";
import { CfgStage } from "./stages/CfgStage";
import { CascadeStage } from "./stages/CascadeStage";
import { EncoderStage } from "./stages/EncoderStage";
import { SotaCompareBars } from "./widgets/SotaCompareBars";
import styles from "./NodePageImagen.module.css";

const prose = extractProse(imagenMarkdown);

export default function NodePageImagen() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/10-diffusion" className={styles.back}>
          ← 返回 Diffusion 扩散模型
        </Link>
        <h1 className={styles.title}>Imagen (2022)</h1>
        <div className={styles.metaLine}>
          作者:Chitwan Saharia · William Chan · Saurabh Saxena · Lala Li et al. · Google Brain
        </div>
        <div className={styles.metaLine}>
          论文:Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding
        </div>
        <p className={styles.keyIdea}>
          用大文本编码器(T5-XXL)+ classifier-free guidance,把文本理解和可控性推到 SOTA —
          COCO FID 7.27 击败 DALL-E 2,39% 人工偏好胜过真实照片,
          CFG 成为所有现代 diffusion 模型的标配
        </p>
      </section>

      <section className={styles.stage}>
        <CfgStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <CascadeStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <EncoderStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能对比</h2>
          <SotaCompareBars />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={IMAGEN_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={IMAGEN_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={IMAGEN_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={IMAGEN_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
