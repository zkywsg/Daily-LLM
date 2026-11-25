import { Link } from "react-router";

import constitutionalAiMarkdown from "../../../../../../12-rlhf-alignment/03-constitutional-ai.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, CONSTITUTIONAL_AI_SOURCE_PATH } from "./lib/prose";
import { TRAINING_DETAILS } from "./lib/data";
import { ConstitutionStage } from "./stages/ConstitutionStage";
import { SlCaiStage } from "./stages/SlCaiStage";
import { RlaifStage } from "./stages/RlaifStage";
import styles from "./NodePageConstitutionalAI.module.css";

const prose = extractProse(constitutionalAiMarkdown);

export default function NodePageConstitutionalAI() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/12-rlhf-alignment" className={styles.back}>
          ← 返回 RLHF / Alignment
        </Link>
        <h1 className={styles.title}>Constitutional AI (2022)</h1>
        <div className={styles.metaLine}>
          作者:Yuntao Bai · Saurav Kadavath · Sandipan Kundu · Amanda Askell · Jackson Kernion · Andy Jones et al.(Anthropic)
        </div>
        <div className={styles.metaLine}>
          论文:Constitutional AI: Harmlessness from AI Feedback
        </div>
        <p className={styles.keyIdea}>
          用一套书面原则(constitution)让 AI 自评自身输出,生成 AI feedback 替代人类偏好标注 —
          把对齐从"人力密集"压成"算力密集",是 Claude 系列的核心方法
        </p>
      </section>

      <section className={styles.stage}>
        <ConstitutionStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <SlCaiStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <RlaifStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <div style={{ overflowX: "auto", marginBottom: "var(--space-4)" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "var(--fs-sm)", fontFamily: "var(--font-sans)" }}>
              <thead>
                <tr style={{ borderBottom: "2px solid var(--border)" }}>
                  <th style={{ textAlign: "left", padding: 8 }}>维度</th>
                  <th style={{ textAlign: "left", padding: 8 }}>Constitutional AI(论文中 52B 模型)</th>
                </tr>
              </thead>
              <tbody>
                {TRAINING_DETAILS.map((row) => (
                  <tr key={row.dimension} style={{ borderBottom: "1px solid var(--border)" }}>
                    <td style={{ padding: 8, fontWeight: 600 }}>{row.dimension}</td>
                    <td style={{ padding: 8 }}>{row.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={CONSTITUTIONAL_AI_SOURCE_PATH} />
        </div>

        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={CONSTITUTIONAL_AI_SOURCE_PATH} />
        </div>

        <div className={styles.footerSection}>
          <h2>Anthropic 体系:Claude 系列</h2>
          <MarkdownRenderer markdown={prose.claudeLineage} sourcePath={CONSTITUTIONAL_AI_SOURCE_PATH} />
        </div>

        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={CONSTITUTIONAL_AI_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
