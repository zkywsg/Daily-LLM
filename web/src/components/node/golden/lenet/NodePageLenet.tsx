import { Link } from "react-router";

import lenetMarkdown from "../../../../../../01-cnn/01-lenet.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, LENET_SOURCE_PATH } from "./lib/prose";
import { ConvolutionStage } from "./stages/ConvolutionStage";
import { PoolingStage } from "./stages/PoolingStage";
import { FullyConnectedStage } from "./stages/FullyConnectedStage";
import styles from "./NodePageLenet.module.css";

const prose = extractProse(lenetMarkdown);

export default function NodePageLenet() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/01-cnn" className={styles.back}>
          ← 返回 CNN
        </Link>
        <h1 className={styles.title}>LeNet-5 (1998)</h1>
        <div className={styles.metaLine}>
          作者:Yann LeCun · Léon Bottou · Yoshua Bengio · Patrick Haffner
        </div>
        <div className={styles.metaLine}>
          论文:Gradient-Based Learning Applied to Document Recognition
        </div>
        <p className={styles.keyIdea}>
          把卷积+池化+全连接这套范式第一次系统化定义出来,在手写数字识别上跑通
        </p>
      </section>

      <section className={styles.stage}>
        <ConvolutionStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <PoolingStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <FullyConnectedStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>前作进展</h2>
          <MarkdownRenderer markdown={prose.previousWork} sourcePath={LENET_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={LENET_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={LENET_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
