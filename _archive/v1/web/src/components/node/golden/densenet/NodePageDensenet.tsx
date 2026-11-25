import { Link } from "react-router";

import densenetMarkdown from "../../../../../../01-cnn/06-densenet.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, DENSENET_SOURCE_PATH } from "./lib/prose";
import { DenseConnectivityStage } from "./stages/DenseConnectivityStage";
import { GrowthRateStage } from "./stages/GrowthRateStage";
import { TransitionLayerStage } from "./stages/TransitionLayerStage";
import styles from "./NodePageDensenet.module.css";

const prose = extractProse(densenetMarkdown);

export default function NodePageDensenet() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/01-cnn" className={styles.back}>
          ← 返回 CNN 卷积神经网络
        </Link>
        <h1 className={styles.title}>DenseNet (2017)</h1>
        <div className={styles.metaLine}>
          作者:Gao Huang · Zhuang Liu · Laurens van der Maaten · Kilian Q. Weinberger
        </div>
        <div className={styles.metaLine}>
          论文:Densely Connected Convolutional Networks
        </div>
        <p className={styles.keyIdea}>
          每层都直接接收前面所有层的输出(concat 而非加法),把特征复用推到极致
        </p>
      </section>

      <section className={styles.stage}>
        <DenseConnectivityStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <GrowthRateStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <TransitionLayerStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>前作进展</h2>
          <MarkdownRenderer markdown={prose.previousWork} sourcePath={DENSENET_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={DENSENET_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={DENSENET_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={DENSENET_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
