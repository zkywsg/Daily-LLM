import { Link } from "react-router";

import distilbertMarkdown from "../../../../../../06-bert-family/04-distilbert.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, DISTILBERT_SOURCE_PATH } from "./lib/prose";
import { TripleLossStage } from "./stages/TripleLossStage";
import { LayerHalvingStage } from "./stages/LayerHalvingStage";
import { LayerInitStage } from "./stages/LayerInitStage";
import { GlueSpeedChart } from "./widgets/GlueSpeedChart";
import { ParamPerformanceScatterChart } from "./widgets/ParamPerformanceScatterChart";
import styles from "./NodePageDistilBERT.module.css";

const prose = extractProse(distilbertMarkdown);

export default function NodePageDistilBERT() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/06-bert-family" className={styles.back}>
          ← 返回 BERT Family
        </Link>
        <h1 className={styles.title}>DistilBERT (2019)</h1>
        <div className={styles.metaLine}>
          作者:Victor Sanh · Lysandre Debut · Julien Chaumond · Thomas Wolf(HuggingFace)
        </div>
        <div className={styles.metaLine}>
          论文:DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter
        </div>
        <p className={styles.keyIdea}>
          用知识蒸馏把 12 层 BERT teacher 压成 6 层 student,40% 参数 60% 速度
          保留 97% 性能,工业 BERT 部署的事实默认
        </p>
      </section>

      <section className={styles.stage}>
        <TripleLossStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <LayerHalvingStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <LayerInitStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能 vs 速度</h2>
          <GlueSpeedChart />
          <div style={{ marginTop: "var(--space-4)" }}>
            <ParamPerformanceScatterChart />
          </div>
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={DISTILBERT_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={DISTILBERT_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={DISTILBERT_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={DISTILBERT_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
