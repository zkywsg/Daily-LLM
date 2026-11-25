import { useState } from "react";
import { Link } from "react-router";

import slMarkdown from "../../../../../../07-gpt-scaling/04-scaling-laws.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, SCALING_SOURCE_PATH } from "./lib/prose";
import { PowerLawStage } from "./stages/PowerLawStage";
import { ChinchillaStage } from "./stages/ChinchillaStage";
import { BudgetStage } from "./stages/BudgetStage";
import { EmergenceDebate } from "./widgets/EmergenceDebate";
import styles from "./NodePageScalingLaws.module.css";

const prose = extractProse(slMarkdown);

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export default function NodePageScalingLaws() {
  const [metric, setMetric] = useState<"discrete" | "continuous" | "both">("both");

  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/07-gpt-scaling" className={styles.back}>
          ← 返回 GPT scaling
        </Link>
        <h1 className={styles.title}>Scaling Laws (2020 / 2022)</h1>
        <div className={styles.metaLine}>
          作者:Jared Kaplan et al. (OpenAI 2020) · Jordan Hoffmann et al. (DeepMind 2022)
        </div>
        <div className={styles.metaLine}>
          论文:Scaling Laws for Neural Language Models / Training Compute-Optimal Large Language Models
        </div>
        <p className={styles.keyIdea}>
          把 LM loss 随 N / D / C 的关系刻画成幂律 · Kaplan 给出粗略最优,
          Chinchilla 修正到 N:D ≈ 1:20,LLaMA 路线"过训练小模型"是部署阶段最优 —
          三者共同把 LLM 投资决策变成可计算工程问题
        </p>
      </section>

      <section className={styles.stage}>
        <PowerLawStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <ChinchillaStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <BudgetStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>涌现的争议</h2>
          <EmergenceDebate metric={metric} />
          <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
            <button type="button" onClick={() => setMetric("both")} style={btnStyle(metric === "both")}>对比</button>
            <button type="button" onClick={() => setMetric("discrete")} style={btnStyle(metric === "discrete")}>离散 acc</button>
            <button type="button" onClick={() => setMetric("continuous")} style={btnStyle(metric === "continuous")}>连续 score</button>
          </div>
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.emergenceDebate} sourcePath={SCALING_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={SCALING_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={SCALING_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
