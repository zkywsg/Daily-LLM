import { Link } from "react-router";

import ddpmMarkdown from "../../../../../../10-diffusion/01-ddpm.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, DDPM_SOURCE_PATH } from "./lib/prose";
import { ForwardProcessStage } from "./stages/ForwardProcessStage";
import { ReverseProcessStage } from "./stages/ReverseProcessStage";
import { SimplifiedObjectiveStage } from "./stages/SimplifiedObjectiveStage";
import styles from "./NodePageDDPM.module.css";

const prose = extractProse(ddpmMarkdown);

export default function NodePageDDPM() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/10-diffusion" className={styles.back}>
          ← 返回 扩散模型
        </Link>
        <h1 className={styles.title}>DDPM (2020)</h1>
        <div className={styles.metaLine}>
          作者:Jonathan Ho · Ajay Jain · Pieter Abbeel · UC Berkeley
        </div>
        <div className={styles.metaLine}>
          论文:Denoising Diffusion Probabilistic Models
        </div>
        <p className={styles.keyIdea}>
          把图像生成拆成"逐步加噪 → 训一个网络逐步去噪",
          用 ε-prediction + L2 简化目标,首次让扩散模型生成质量超过 GAN
        </p>
      </section>

      <section className={styles.stage}>
        <ForwardProcessStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <ReverseProcessStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <SimplifiedObjectiveStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        {prose.sampling && (
          <div className={styles.footerSection}>
            <h2>采样过程</h2>
            <MarkdownRenderer markdown={prose.sampling} sourcePath={DDPM_SOURCE_PATH} />
          </div>
        )}
        {prose.uNet && (
          <div className={styles.footerSection}>
            <h2>U-Net Backbone 的选择</h2>
            <MarkdownRenderer markdown={prose.uNet} sourcePath={DDPM_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={DDPM_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={DDPM_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={DDPM_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
