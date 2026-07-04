import { Link } from "react-router";

import gloveMarkdown from "../../../../../../03-word-embedding/02-glove.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, GLOVE_SOURCE_PATH } from "./lib/prose";
import { RatioStage } from "./stages/RatioStage";
import { WeightedLossStage } from "./stages/WeightedLossStage";
import { GlobalMatrixStage } from "./stages/GlobalMatrixStage";
import { SimilarityCompareBars } from "./widgets/SimilarityCompareBars";
import styles from "./NodePageGloVe.module.css";

const prose = extractProse(gloveMarkdown);

export default function NodePageGloVe() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/03-word-embedding" className={styles.back}>
          ← 返回 Word Embedding 词嵌入
        </Link>
        <h1 className={styles.title}>GloVe (2014)</h1>
        <div className={styles.metaLine}>
          作者:Jeffrey Pennington · Richard Socher · Christopher D. Manning · Stanford NLP
        </div>
        <div className={styles.metaLine}>
          论文:GloVe: Global Vectors for Word Representation
        </div>
        <p className={styles.keyIdea}>
          直接对全局 word-word 共现矩阵做加权 log-bilinear 分解,而不是用局部窗口预测 —
          count-based 路线,理论比 Word2Vec 更清晰,
          Stanford 预训练 GloVe 向量成 2014-2018 开源标配
        </p>
      </section>

      <section className={styles.stage}>
        <RatioStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <WeightedLossStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <GlobalMatrixStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <SimilarityCompareBars />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={GLOVE_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={GLOVE_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={GLOVE_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
