import { Link } from "react-router";

import ragMarkdown from "../../../../../../14-rag-agent/01-rag.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, RAG_SOURCE_PATH } from "./lib/prose";
import { DenseRetrievalStage } from "./stages/DenseRetrievalStage";
import { ContextAugStage } from "./stages/ContextAugStage";
import { ArchitectureStage } from "./stages/ArchitectureStage";
import styles from "./NodePageRAG.module.css";

const prose = extractProse(ragMarkdown);

export default function NodePageRAG() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/14-rag-agent" className={styles.back}>
          ← 返回 RAG 与 Agent
        </Link>
        <h1 className={styles.title}>RAG (2020)</h1>
        <div className={styles.metaLine}>
          作者:Patrick Lewis · Ethan Perez · Aleksandra Piktus · ... · Douwe Kiela · Facebook AI
        </div>
        <div className={styles.metaLine}>
          论文:Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
        </div>
        <p className={styles.keyIdea}>
          dense retriever 把私有 / 新鲜知识当 \"非参数化记忆\" 检索回来,
          拼进 LLM prompt — 让模型 \"看着资料回答\" 而不是凭参数硬编
        </p>
      </section>

      <section className={styles.stage}>
        <DenseRetrievalStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <ContextAugStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <ArchitectureStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
          implementationProse={prose.implementation}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={RAG_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={RAG_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={RAG_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
