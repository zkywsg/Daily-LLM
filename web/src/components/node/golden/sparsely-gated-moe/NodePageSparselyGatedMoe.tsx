import { Link } from "react-router";

import sparselyGatedMoeMarkdown from "../../../../../../13-moe-efficient/01-sparsely-gated-moe.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, SPARSELY_GATED_MOE_SOURCE_PATH } from "./lib/prose";
import { TopKGatingStage } from "./stages/TopKGatingStage";
import { AuxLossStage } from "./stages/AuxLossStage";
import { ExpertParallelismStage } from "./stages/ExpertParallelismStage";
import styles from "./NodePageSparselyGatedMoe.module.css";

const prose = extractProse(sparselyGatedMoeMarkdown);

export default function NodePageSparselyGatedMoe() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/13-moe-efficient" className={styles.back}>
          ← 返回 MoE 与高效大模型
        </Link>
        <h1 className={styles.title}>Sparsely-Gated MoE (2017)</h1>
        <div className={styles.metaLine}>
          作者:Noam Shazeer · Azalia Mirhoseini · Krzysztof Maziarz · Andy Davis ·
          Quoc Le · Geoffrey Hinton · Jeff Dean · Google Brain
        </div>
        <div className={styles.metaLine}>
          论文:Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
        </div>
        <p className={styles.keyIdea}>
          在 LSTM 之间插入 sparsely-gated MoE 层:每 token 用 gate 选 top-K
          个 expert(1370 亿参数中只激活几亿),配 auxiliary loss 防止 expert
          塌缩;首次证明稀疏激活能突破 dense 模型的参数 / 算力锁死。
        </p>
      </section>

      <section className={styles.stage}>
        <TopKGatingStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <AuxLossStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <ExpertParallelismStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={SPARSELY_GATED_MOE_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={SPARSELY_GATED_MOE_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={SPARSELY_GATED_MOE_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
