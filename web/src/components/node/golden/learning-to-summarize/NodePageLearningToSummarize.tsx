import { Link } from "react-router";

import learningToSummarizeMarkdown from "../../../../../../12-rlhf-alignment/01-learning-to-summarize.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, LEARNING_TO_SUMMARIZE_SOURCE_PATH } from "./lib/prose";
import { PreferenceCollectionStage } from "./stages/PreferenceCollectionStage";
import { RewardModelStage } from "./stages/RewardModelStage";
import { PpoStage } from "./stages/PpoStage";
import { AlignmentBeatsScaleChart } from "./widgets/AlignmentBeatsScaleChart";
import styles from "./NodePageLearningToSummarize.module.css";

const prose = extractProse(learningToSummarizeMarkdown);

export default function NodePageLearningToSummarize() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/12-rlhf-alignment" className={styles.back}>
          ← 返回 RLHF / Alignment
        </Link>
        <h1 className={styles.title}>Learning to Summarize (2020)</h1>
        <div className={styles.metaLine}>
          作者:Nisan Stiennon · Long Ouyang · Jeff Wu · Daniel Ziegler · Ryan Lowe ·
          Chelsea Voss · Alec Radford · Dario Amodei · Paul Christiano(OpenAI)
        </div>
        <div className={styles.metaLine}>
          论文:Learning to Summarize from Human Feedback
        </div>
        <p className={styles.keyIdea}>
          用人工偏好比较训练 reward model + PPO 微调 LLM,摘要质量超过监督学习
          baseline 和参考摘要,确立 RLHF 在 NLP 上的完整方案
        </p>
      </section>

      <section className={styles.stage}>
        <PreferenceCollectionStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <RewardModelStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <PpoStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能:超过参考摘要</h2>
          <AlignmentBeatsScaleChart />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={LEARNING_TO_SUMMARIZE_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={LEARNING_TO_SUMMARIZE_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={LEARNING_TO_SUMMARIZE_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={LEARNING_TO_SUMMARIZE_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
