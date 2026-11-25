import { Link } from "react-router";

import ldmMarkdown from "../../../../../../10-diffusion/02-ldm.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, LDM_SOURCE_PATH } from "./lib/prose";
import { PerceptualCompressionStage } from "./stages/PerceptualCompressionStage";
import { LatentDiffusionStage } from "./stages/LatentDiffusionStage";
import { CrossAttentionStage } from "./stages/CrossAttentionStage";
import styles from "./NodePageLDM.module.css";

const prose = extractProse(ldmMarkdown);

export default function NodePageLDM() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/10-diffusion" className={styles.back}>
          ← 返回 Diffusion 扩散模型
        </Link>
        <h1 className={styles.title}>Stable Diffusion / LDM (2022)</h1>
        <div className={styles.metaLine}>
          作者:Robin Rombach · Andreas Blattmann · Dominik Lorenz · Patrick Esser · Björn Ommer
        </div>
        <div className={styles.metaLine}>
          论文:High-Resolution Image Synthesis with Latent Diffusion Models
        </div>
        <p className={styles.keyIdea}>
          把 diffusion 从 pixel 搬到 64²×4 VAE latent,显存降 49×,加 cross-attention
          统一接 text/class/segmap — 2022 年 8 月 4GB 开源模型在消费 GPU 跑 512² 文生图,
          引爆 AI 绘画时代
        </p>
      </section>

      <section className={styles.stage}>
        <PerceptualCompressionStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <LatentDiffusionStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <CrossAttentionStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>Stable Diffusion 的具体配置</h2>
          <MarkdownRenderer markdown={prose.sdConfig} sourcePath={LDM_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={LDM_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={LDM_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={LDM_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
