import { Link } from "react-router";

import inceptionMarkdown from "../../../../../../01-cnn/04-inception.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, INCEPTION_SOURCE_PATH } from "./lib/prose";
import { MultiScaleModuleStage } from "./stages/MultiScaleModuleStage";
import { DimensionReductionStage } from "./stages/DimensionReductionStage";
import { EfficientArchitectureStage } from "./stages/EfficientArchitectureStage";
import styles from "./NodePageInception.module.css";

const prose = extractProse(inceptionMarkdown);

export default function NodePageInception() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/01-cnn" className={styles.back}>
          ← 返回 CNN
        </Link>
        <h1 className={styles.title}>GoogLeNet (Inception v1) (2014)</h1>
        <div className={styles.metaLine}>
          作者:Christian Szegedy · Wei Liu · Yangqing Jia · Pierre Sermanet · Scott Reed ·
          Dragomir Anguelov · Dumitru Erhan · Vincent Vanhoucke · Andrew Rabinovich
        </div>
        <div className={styles.metaLine}>论文:Going Deeper with Convolutions</div>
        <p className={styles.keyIdea}>
          用 1×1 卷积降维 + 多尺度并行的 Inception 模块,把参数量压到 VGG 的 1/12 同时拿下 ImageNet 冠军
        </p>
      </section>

      <section className={styles.stage}>
        <MultiScaleModuleStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <DimensionReductionStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <EfficientArchitectureStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>前作进展</h2>
          <MarkdownRenderer markdown={prose.previousWork} sourcePath={INCEPTION_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节 / 性能数据</h2>
          <MarkdownRenderer markdown={prose.performance} sourcePath={INCEPTION_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={INCEPTION_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={INCEPTION_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
