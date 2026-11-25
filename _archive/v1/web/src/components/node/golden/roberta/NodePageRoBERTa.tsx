import { Link } from "react-router";

import robertaMarkdown from "../../../../../../06-bert-family/02-roberta.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, ROBERTA_SOURCE_PATH } from "./lib/prose";
import { AblationStage } from "./stages/AblationStage";
import { NspStage } from "./stages/NspStage";
import { EngineeringStage } from "./stages/EngineeringStage";
import styles from "./NodePageRoBERTa.module.css";

const prose = extractProse(robertaMarkdown);

export default function NodePageRoBERTa() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/06-bert-family" className={styles.back}>
          ← 返回 BERT Family
        </Link>
        <h1 className={styles.title}>RoBERTa (2019)</h1>
        <div className={styles.metaLine}>
          作者:Yinhan Liu · Myle Ott · Naman Goyal · Jingfei Du · Mandar Joshi · Danqi Chen et al.(Facebook AI)
        </div>
        <div className={styles.metaLine}>
          论文:RoBERTa: A Robustly Optimized BERT Pretraining Approach
        </div>
        <p className={styles.keyIdea}>
          去掉 NSP + 动态 masking + 大 batch + 10× 数据 + 更长训练,
          证明 BERT 严重训练不足 — 架构一行不改,GLUE 再涨 5+ 分,
          确立"做架构改动前先确认 baseline 充分训练"的研究规范
        </p>
      </section>

      <section className={styles.stage}>
        <AblationStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <NspStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <EngineeringStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={ROBERTA_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={ROBERTA_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={ROBERTA_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
