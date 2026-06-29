import { Link } from "react-router";

import mixtralMarkdown from "../../../../../../13-moe-efficient/03-mixtral.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, MIXTRAL_SOURCE_PATH } from "./lib/prose";
import { SparseMoEStage } from "./stages/SparseMoEStage";
import { LoadBalanceStage } from "./stages/LoadBalanceStage";
import { TopKParallelismStage } from "./stages/TopKParallelismStage";
import styles from "./NodePageMixtral.module.css";

const prose = extractProse(mixtralMarkdown);

export default function NodePageMixtral() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/13-moe-efficient" className={styles.back}>
          ← 返回 MoE 与高效大模型
        </Link>
        <h1 className={styles.title}>Mixtral 8×7B (2023)</h1>
        <div className={styles.metaLine}>
          作者:Albert Q. Jiang · Alexandre Sablayrolles · ... · Mistral AI
        </div>
        <div className={styles.metaLine}>
          论文:Mixtral of Experts
        </div>
        <p className={styles.keyIdea}>
          8 个 expert + per-token top-2 routing,总参数 47B / 激活 13B —
          首个开源 SOTA 的 MoE,推理算力 dense-13B 但质量 dense-70B
        </p>
      </section>

      <section className={styles.stage}>
        <SparseMoEStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <LoadBalanceStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <TopKParallelismStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={MIXTRAL_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={MIXTRAL_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={MIXTRAL_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
