import { Link } from "react-router";

import efficientnetMarkdown from "../../../../../../01-cnn/07-efficientnet.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, EFFICIENTNET_SOURCE_PATH } from "./lib/prose";
import { CompoundScalingStage } from "./stages/CompoundScalingStage";
import { SeedArchitectureStage } from "./stages/SeedArchitectureStage";
import { RegularizationStage } from "./stages/RegularizationStage";
import styles from "./NodePageEfficientnet.module.css";

const prose = extractProse(efficientnetMarkdown);

export default function NodePageEfficientnet() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/01-cnn" className={styles.back}>
          ← 返回 CNN
        </Link>
        <h1 className={styles.title}>EfficientNet (2019)</h1>
        <div className={styles.metaLine}>作者:Mingxing Tan · Quoc V. Le</div>
        <div className={styles.metaLine}>
          论文:EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
        </div>
        <p className={styles.keyIdea}>
          用复合缩放系数把 depth/width/resolution 三轴联合缩放公式化,得到帕累托最优的
          B0–B7 模型族
        </p>
      </section>

      <section className={styles.stage}>
        <CompoundScalingStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <SeedArchitectureStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <RegularizationStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.performance} sourcePath={EFFICIENTNET_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={EFFICIENTNET_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={EFFICIENTNET_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
