import { Link } from "react-router";

import albertMarkdown from "../../../../../../06-bert-family/03-albert.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, ALBERT_SOURCE_PATH } from "./lib/prose";
import { FactorizedEmbeddingStage } from "./stages/FactorizedEmbeddingStage";
import { ParameterSharingStage } from "./stages/ParameterSharingStage";
import { SopStage } from "./stages/SopStage";
import { AlbertBertScoreChart } from "./widgets/AlbertBertScoreChart";
import styles from "./NodePageALBERT.module.css";

const prose = extractProse(albertMarkdown);

export default function NodePageALBERT() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/06-bert-family" className={styles.back}>
          ← 返回 BERT Family
        </Link>
        <h1 className={styles.title}>ALBERT (2019)</h1>
        <div className={styles.metaLine}>
          作者:Zhenzhong Lan · Mingda Chen · Sebastian Goodman · Kevin Gimpel · Piyush Sharma · Radu Soricut(Google)
        </div>
        <div className={styles.metaLine}>
          论文:ALBERT: A Lite BERT for Self-supervised Learning of Language Representations
        </div>
        <p className={styles.keyIdea}>
          用跨层参数共享 + embedding 因式分解把 BERT-large 参数从 334M 压到 18M
          而效果接近,同时把 NSP 改成更难的 SOP(句子顺序预测)
        </p>
      </section>

      <section className={styles.stage}>
        <FactorizedEmbeddingStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <ParameterSharingStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <SopStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能与权衡</h2>
          <AlbertBertScoreChart />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={ALBERT_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={ALBERT_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={ALBERT_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={ALBERT_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
