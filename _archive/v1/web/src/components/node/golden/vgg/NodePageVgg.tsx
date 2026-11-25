import { Link } from "react-router";

import vggMarkdown from "../../../../../../01-cnn/03-vgg.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, VGG_SOURCE_PATH } from "./lib/prose";
import { SmallKernelStackingStage } from "./stages/SmallKernelStackingStage";
import { ModularBlockStage } from "./stages/ModularBlockStage";
import { PretrainSeedingStage } from "./stages/PretrainSeedingStage";
import styles from "./NodePageVgg.module.css";

const prose = extractProse(vggMarkdown);

export default function NodePageVgg() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/01-cnn" className={styles.back}>
          ← 返回 CNN
        </Link>
        <h1 className={styles.title}>VGG (2014)</h1>
        <div className={styles.metaLine}>作者:Karen Simonyan · Andrew Zisserman</div>
        <div className={styles.metaLine}>
          论文:Very Deep Convolutional Networks for Large-Scale Image Recognition
        </div>
        <p className={styles.keyIdea}>
          把网络深度做到 16/19 层、并把所有卷积统一成 3×3,证明深度本身就是性能来源
        </p>
      </section>

      <section className={styles.stage}>
        <SmallKernelStackingStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <ModularBlockStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <PretrainSeedingStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>前作进展</h2>
          <MarkdownRenderer markdown={prose.previousWork} sourcePath={VGG_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节 / 性能数据</h2>
          <MarkdownRenderer markdown={prose.performance} sourcePath={VGG_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={VGG_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={VGG_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
