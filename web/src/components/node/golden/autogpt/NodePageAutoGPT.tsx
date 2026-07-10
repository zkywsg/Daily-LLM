import { Link } from "react-router";

import autogptMarkdown from "../../../../../../14-rag-agent/04-autogpt.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, AUTOGPT_SOURCE_PATH } from "./lib/prose";
import { DecompositionStage } from "./stages/DecompositionStage";
import { ToolLoopStage } from "./stages/ToolLoopStage";
import { CritiqueStage } from "./stages/CritiqueStage";
import { BenchmarkCompareChart } from "./widgets/BenchmarkCompareChart";
import styles from "./NodePageAutoGPT.module.css";

const prose = extractProse(autogptMarkdown);

export default function NodePageAutoGPT() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/14-rag-agent" className={styles.back}>
          ← 返回 RAG / Agent
        </Link>
        <h1 className={styles.title}>AutoGPT (2023)</h1>
        <div className={styles.metaLine}>
          作者:Toran Bruce Richards("Significant Gravitas")· AutoGPT contributors
        </div>
        <div className={styles.metaLine}>
          论文:AutoGPT(开源项目,无正式论文);相关综述:A Survey on LLM-based Autonomous Agents
        </div>
        <p className={styles.keyIdea}>
          把 ReAct 推到极限 — LLM 拿到高级目标后自己分解为子任务、规划
          执行步骤、循环调工具直到完成,无人干预;启动自主 agent 范式
        </p>
      </section>

      <section className={styles.stage}>
        <DecompositionStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <ToolLoopStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <CritiqueStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <BenchmarkCompareChart />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={AUTOGPT_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={AUTOGPT_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={AUTOGPT_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
