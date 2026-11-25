import { Link } from "react-router";

import o1Markdown from "../../../../../../15-reasoning-o1-r1/03-o1.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, O1_SOURCE_PATH } from "./lib/prose";
import { RlTrainingStage } from "./stages/RlTrainingStage";
import { TestTimeScalingStage } from "./stages/TestTimeScalingStage";
import { HiddenTraceStage } from "./stages/HiddenTraceStage";
import { BenchmarkCompareBars } from "./widgets/BenchmarkCompareBars";
import styles from "./NodePageO1.module.css";

const prose = extractProse(o1Markdown);

export default function NodePageO1() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/15-reasoning-o1-r1" className={styles.back}>
          ← 返回 Reasoning o1/R1
        </Link>
        <h1 className={styles.title}>OpenAI o1 (2024)</h1>
        <div className={styles.metaLine}>
          作者:OpenAI Reasoning Team
        </div>
        <div className={styles.metaLine}>
          论文:Learning to Reason with LLMs(OpenAI 技术博客)
        </div>
        <p className={styles.keyIdea}>
          把长链推理作为训练目标,用 RL 让 LLM 自己学到反思/回溯/自验证 —
          test-time compute 成为继训练算力之后的新 scaling 轴,
          在数学/科学/代码 benchmark 上击败 GPT-4 多倍,首次在 GPQA 上超越人类专家
        </p>
      </section>

      <section className={styles.stage}>
        <RlTrainingStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <TestTimeScalingStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <HiddenTraceStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <BenchmarkCompareBars />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={O1_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={O1_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={O1_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
