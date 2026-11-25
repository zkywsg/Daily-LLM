import { Link } from "react-router";

import gpt3Markdown from "../../../../../../07-gpt-scaling/03-gpt3.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, GPT3_SOURCE_PATH } from "./lib/prose";
import { ScalingLawStage } from "./stages/ScalingLawStage";
import { InContextLearningStage } from "./stages/InContextLearningStage";
import { SparseAttentionStage } from "./stages/SparseAttentionStage";
import styles from "./NodePageGPT3.module.css";

const prose = extractProse(gpt3Markdown);

export default function NodePageGPT3() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/07-gpt-scaling" className={styles.back}>
          ← 返回 大语言模型 (GPT 系 + Scaling)
        </Link>
        <h1 className={styles.title}>GPT-3 (2020)</h1>
        <div className={styles.metaLine}>
          作者:Tom Brown · Benjamin Mann · Nick Ryder · ... · Dario Amodei · OpenAI
        </div>
        <div className={styles.metaLine}>
          论文:Language Models are Few-Shot Learners
        </div>
        <p className={styles.keyIdea}>
          把 GPT-2 直接放大 117× 到 175B,in-context learning 自然涌现:
          不更新权重,只在 prompt 里塞几个示例就能解决新任务 — 大模型时代由此开局
        </p>
      </section>

      <section className={styles.stage}>
        <ScalingLawStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <InContextLearningStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <SparseAttentionStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={GPT3_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={GPT3_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={GPT3_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
