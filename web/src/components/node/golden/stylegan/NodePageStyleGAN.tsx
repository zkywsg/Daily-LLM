import { Link } from "react-router";

import sgMarkdown from "../../../../../../04-gan/04-stylegan.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, STYLEGAN_SOURCE_PATH } from "./lib/prose";
import { MappingStage } from "./stages/MappingStage";
import { AdaINStage } from "./stages/AdaINStage";
import { StyleMixingStage } from "./stages/StyleMixingStage";
import { FFHQFidBars } from "./widgets/FFHQFidBars";
import styles from "./NodePageStyleGAN.module.css";

const prose = extractProse(sgMarkdown);

export default function NodePageStyleGAN() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/04-gan" className={styles.back}>
          ← 返回 GAN 生成对抗网络
        </Link>
        <h1 className={styles.title}>StyleGAN (2018)</h1>
        <div className={styles.metaLine}>
          作者:Tero Karras · Samuli Laine · Timo Aila · NVIDIA
        </div>
        <div className={styles.metaLine}>
          论文:A Style-Based Generator Architecture for Generative Adversarial Networks
        </div>
        <p className={styles.keyIdea}>
          Mapping Network(z→w 解纠缠) + AdaIN(style 控制统计量) + 分层 style 注入,
          1024² 人脸生成质量逼近真实照片 · "This Person Does Not Exist" 病毒式传播 ·
          GAN 时代质量与可控性的巅峰
        </p>
      </section>

      <section className={styles.stage}>
        <MappingStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <AdaINStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <StyleMixingStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <FFHQFidBars />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={STYLEGAN_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={STYLEGAN_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={STYLEGAN_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
