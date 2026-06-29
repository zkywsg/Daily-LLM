import { Link } from "react-router";

import word2vecMarkdown from "../../../../../../03-word-embedding/01-word2vec.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, WORD2VEC_SOURCE_PATH } from "./lib/prose";
import { CbowSkipgramStage } from "./stages/CbowSkipgramStage";
import { NegSamplingStage } from "./stages/NegSamplingStage";
import { SubsamplingStage } from "./stages/SubsamplingStage";
import { LinearArithmetic2D } from "./widgets/LinearArithmetic2D";
import styles from "./NodePageWord2Vec.module.css";

const prose = extractProse(word2vecMarkdown);

export default function NodePageWord2Vec() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/03-word-embedding" className={styles.back}>
          ← 返回 Word Embedding 词嵌入
        </Link>
        <h1 className={styles.title}>Word2Vec (2013)</h1>
        <div className={styles.metaLine}>
          作者:Tomas Mikolov · Kai Chen · Greg Corrado · Jeffrey Dean · Google
        </div>
        <div className={styles.metaLine}>
          论文:Efficient Estimation of Word Representations in Vector Space / Distributed Representations of Words and Phrases
        </div>
        <p className={styles.keyIdea}>
          浅网络 + negative sampling + 高频词 subsampling 三件套,
          让 distributional hypothesis 第一次跑到工业速度,
          king − man + woman ≈ queen 把语义带进向量空间
        </p>
      </section>

      <section className={styles.stage}>
        <CbowSkipgramStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <NegSamplingStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <SubsamplingStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>线性结构的发现</h2>
          <LinearArithmetic2D />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.linearStructure} sourcePath={WORD2VEC_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={WORD2VEC_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <MarkdownRenderer markdown={prose.performance} sourcePath={WORD2VEC_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={WORD2VEC_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
