import { Link } from "react-router";

import r1Markdown from "../../../../../../15-reasoning-o1-r1/04-deepseek-r1.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, DEEPSEEK_R1_SOURCE_PATH } from "./lib/prose";
import { R1ZeroStage } from "./stages/R1ZeroStage";
import { GrpoStage } from "./stages/GrpoStage";
import { MultiStageStage } from "./stages/MultiStageStage";
import { BenchmarkCompareBars } from "./widgets/BenchmarkCompareBars";
import { DistillTableChart } from "./widgets/DistillTableChart";
import styles from "./NodePageDeepSeekR1.module.css";

const prose = extractProse(r1Markdown);

export default function NodePageDeepSeekR1() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/15-reasoning-o1-r1" className={styles.back}>
          ← 返回 Reasoning o1/R1
        </Link>
        <h1 className={styles.title}>DeepSeek-R1 (2025)</h1>
        <div className={styles.metaLine}>
          作者:DeepSeek-AI
        </div>
        <div className={styles.metaLine}>
          论文:DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
        </div>
        <p className={styles.keyIdea}>
          开源 o1 风格推理模型 — 先用纯 RL(GRPO)无 SFT cold start 训练 R1-Zero
          验证推理行为可从 RL 中涌现,再用少量 cold-start SFT + 多阶段 RL 训练 R1
          达到 o1 同级性能,推理 trace 全公开,开源第一次追上闭源 reasoning 旗舰
        </p>
      </section>

      <section className={styles.stage}>
        <R1ZeroStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <GrpoStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <MultiStageStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <BenchmarkCompareBars />
          <div style={{ marginTop: "var(--space-6)" }}>
            <DistillTableChart />
          </div>
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={DEEPSEEK_R1_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={DEEPSEEK_R1_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={DEEPSEEK_R1_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
