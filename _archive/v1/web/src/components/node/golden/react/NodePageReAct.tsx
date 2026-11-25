import { Link } from "react-router";

import reactMarkdown from "../../../../../../14-rag-agent/02-react.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, REACT_SOURCE_PATH } from "./lib/prose";
import { ThoughtStage } from "./stages/ThoughtStage";
import { ActionStage } from "./stages/ActionStage";
import { ObservationStage } from "./stages/ObservationStage";
import { AgentTaskCompareChart } from "./widgets/AgentTaskCompareChart";
import styles from "./NodePageReAct.module.css";

const prose = extractProse(reactMarkdown);

export default function NodePageReAct() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/14-rag-agent" className={styles.back}>
          ← 返回 RAG / Agent
        </Link>
        <h1 className={styles.title}>ReAct (2022)</h1>
        <div className={styles.metaLine}>
          作者:Shunyu Yao · Jeffrey Zhao · Dian Yu · Nan Du · Izhak Shafran · Karthik Narasimhan · Yuan Cao(普林斯顿 + Google)
        </div>
        <div className={styles.metaLine}>
          论文:ReAct: Synergizing Reasoning and Acting in Language Models
        </div>
        <p className={styles.keyIdea}>
          把 LLM 的推理(Thought)和行动(Action)交错进行,thought 推理
          下一步要查什么,action 调外部工具,observation 反馈给 LLM
          继续推理 — Agent 范式的起源
        </p>
      </section>

      <section className={styles.stage}>
        <ThoughtStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <ActionStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <ObservationStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <AgentTaskCompareChart />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={REACT_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={REACT_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={REACT_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
