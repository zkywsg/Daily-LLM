import { Link } from "react-router";

import selfConsistencyMarkdown from "../../../../../../15-reasoning-o1-r1/02-self-consistency.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, SELF_CONSISTENCY_SOURCE_PATH } from "./lib/prose";
import { TemperatureSamplingStage } from "./stages/TemperatureSamplingStage";
import { AnswerNormalizationStage } from "./stages/AnswerNormalizationStage";
import { MajorityVoteStage } from "./stages/MajorityVoteStage";
import { BenchmarkComparisonChart } from "./widgets/BenchmarkComparisonChart";
import styles from "./NodePageSelfConsistency.module.css";

const prose = extractProse(selfConsistencyMarkdown);

export default function NodePageSelfConsistency() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/15-reasoning-o1-r1" className={styles.back}>
          ← 返回 Reasoning o1/R1
        </Link>
        <h1 className={styles.title}>Self-Consistency (2022)</h1>
        <div className={styles.metaLine}>
          作者:Xuezhi Wang · Jason Wei · Dale Schuurmans · Quoc Le · Ed Chi · Sharan Narang · Aakanksha Chowdhery · Denny Zhou
        </div>
        <div className={styles.metaLine}>
          论文:Self-Consistency Improves Chain of Thought Reasoning in Language Models
        </div>
        <p className={styles.keyIdea}>
          对同 prompt 采样 N 条 CoT 推理路径,投票选最一致答案;
          GSM8K 60% → 75%;第一次系统化 test-time compute scaling
        </p>
      </section>

      <section className={styles.stage}>
        <TemperatureSamplingStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <AnswerNormalizationStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <MajorityVoteStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>前作进展</h2>
          <BenchmarkComparisonChart />
          <MarkdownRenderer markdown={prose.previousWork} sourcePath={SELF_CONSISTENCY_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>与其他 reasoning prompt 技术的关系</h2>
          <MarkdownRenderer markdown={prose.relations} sourcePath={SELF_CONSISTENCY_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={SELF_CONSISTENCY_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={SELF_CONSISTENCY_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
