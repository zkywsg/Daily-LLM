import { Link } from "react-router";

import toolformerMarkdown from "../../../../../../14-rag-agent/03-toolformer.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, TOOLFORMER_SOURCE_PATH } from "./lib/prose";
import { SamplingStage } from "./stages/SamplingStage";
import { FilterStage } from "./stages/FilterStage";
import { FinetuneStage } from "./stages/FinetuneStage";
import styles from "./NodePageToolformer.module.css";

const prose = extractProse(toolformerMarkdown);

export default function NodePageToolformer() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/14-rag-agent" className={styles.back}>
          ← 返回 RAG / Agent
        </Link>
        <h1 className={styles.title}>Toolformer (2023)</h1>
        <div className={styles.metaLine}>
          作者:Timo Schick · Jane Dwivedi-Yu · Roberto Dessì · Roberta Raileanu · Maria Lomeli et al.(Meta AI)
        </div>
        <div className={styles.metaLine}>
          论文:Toolformer: Language Models Can Teach Themselves to Use Tools
        </div>
        <p className={styles.keyIdea}>
          让 LLM 在预训练语料上自监督学习何时何处插入工具调用 —
          给候选位置加 tool call,如果调用后 perplexity 降低就保留;
          tool use 从 prompt 技巧内化为模型本身能力
        </p>
      </section>

      <section className={styles.stage}>
        <SamplingStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <FilterStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <FinetuneStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <MarkdownRenderer markdown={prose.performance} sourcePath={TOOLFORMER_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={TOOLFORMER_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={TOOLFORMER_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
