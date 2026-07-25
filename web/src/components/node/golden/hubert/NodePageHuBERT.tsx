import { Link } from "react-router";
import hubertMarkdown from "../../../../../../18-speech-audio/02-hubert.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, HUBERT_SOURCE_PATH } from "./lib/prose";
import { ClusteringStage } from "./stages/ClusteringStage";
import { MaskedClassifyStage } from "./stages/MaskedClassifyStage";
import { ReclusterStage } from "./stages/ReclusterStage";
import styles from "./NodePageHuBERT.module.css";

const prose = extractProse(hubertMarkdown);

export default function NodePageHuBERT() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/18-speech-audio" className={styles.back}>
          ← 返回语音/音频模型
        </Link>
        <h1 className={styles.title}>HuBERT (2021)</h1>
        <div className={styles.metaLine}>作者:Wei-Ning Hsu · Benjamin Bolte · Yao-Hung Hubert Tsai · Kushal Lakhotia · Ruslan Salakhutdinov · Abdelrahman Mohamed</div>
        <div className={styles.metaLine}>论文:HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units</div>
        <p className={styles.keyIdea}>
          用离线 k-means 聚类生成离散伪标签,再做 BERT 式掩码预测,配合迭代式重新聚类不断提纯伪标签的音素区分度
        </p>
      </section>

      <section className={styles.stage}>
        <ClusteringStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <MaskedClassifyStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <ReclusterStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={HUBERT_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={HUBERT_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={HUBERT_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
