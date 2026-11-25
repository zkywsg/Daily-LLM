import { Link } from "react-router";

import dpoMarkdown from "../../../../../../12-rlhf-alignment/04-dpo.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, DPO_SOURCE_PATH } from "./lib/prose";
import { PipelineStage } from "./stages/PipelineStage";
import { LossStage } from "./stages/LossStage";
import { TrainingDynamicsStage } from "./stages/TrainingDynamicsStage";
import { BenchmarkBars } from "./widgets/BenchmarkBars";
import { DpoFamilyTimeline } from "./widgets/DpoFamilyTimeline";
import styles from "./NodePageDPO.module.css";

const prose = extractProse(dpoMarkdown);

export default function NodePageDPO() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/12-rlhf-alignment" className={styles.back}>
          ← 返回 RLHF 对齐
        </Link>
        <h1 className={styles.title}>DPO (2023)</h1>
        <div className={styles.metaLine}>
          作者:Rafael Rafailov · Archit Sharma · Eric Mitchell · Stefano Ermon · Chris Manning · Chelsea Finn
        </div>
        <div className={styles.metaLine}>
          论文:Direct Preference Optimization — Your Language Model is Secretly a Reward Model
        </div>
        <p className={styles.keyIdea}>
          数学推导把 RLHF 三阶段折成一个 SFT-style cross-entropy loss,
          跳过 reward model 和 PPO,工程复杂度降一个数量级,
          性能和 PPO 持平甚至略好 — 2024 开源 LLM 默认对齐方法
        </p>
      </section>

      <section className={styles.stage}>
        <PipelineStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <LossStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <TrainingDynamicsStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>DPO vs PPO 性能</h2>
          <BenchmarkBars />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.vsPpo} sourcePath={DPO_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>DPO 家族 — 2023.5 → 2024.5</h2>
          <DpoFamilyTimeline />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.family} sourcePath={DPO_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={DPO_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={DPO_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={DPO_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
