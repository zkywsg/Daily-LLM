import { Link } from "react-router";

import elmoMarkdown from "../../../../../../03-word-embedding/04-elmo.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, ELMO_SOURCE_PATH } from "./lib/prose";
import { BiLMStage } from "./stages/BiLMStage";
import { LayerWeightStage } from "./stages/LayerWeightStage";
import { FrozenFeatureStage } from "./stages/FrozenFeatureStage";
import { TaskGainBars } from "./widgets/TaskGainBars";
import styles from "./NodePageELMo.module.css";

const prose = extractProse(elmoMarkdown);

export default function NodePageELMo() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/03-word-embedding" className={styles.back}>
          ← 返回 Word Embedding 词嵌入
        </Link>
        <h1 className={styles.title}>ELMo (2018)</h1>
        <div className={styles.metaLine}>
          作者:Matthew E. Peters · Mark Neumann · Mohit Iyyer · Matt Gardner et al. · AllenAI + UW
        </div>
        <div className={styles.metaLine}>
          论文:Deep Contextualized Word Representations(NAACL 2018 best paper)
        </div>
        <p className={styles.keyIdea}>
          双向 LSTM 预训练 + 多层 hidden 加权组合 + frozen feature 拼接 —
          同一个 "bank" 在 river bank / money bank 里向量不同 · 6 任务全面 SOTA ·
          "NLP 的 ImageNet 时刻" · BERT 直接前驱
        </p>
      </section>

      <section className={styles.stage}>
        <BiLMStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <LayerWeightStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <FrozenFeatureStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <TaskGainBars />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={ELMO_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={ELMO_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={ELMO_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
