import { Link } from "react-router";
import gatMarkdown from "../../../../../../17-graph-neural-networks/03-gat.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, GAT_SOURCE_PATH } from "./lib/prose";
import { AttentionCoeffStage } from "./stages/AttentionCoeffStage";
import { MaskSoftmaxStage } from "./stages/MaskSoftmaxStage";
import { MultiHeadStage } from "./stages/MultiHeadStage";
import styles from "./NodePageGAT.module.css";

const prose = extractProse(gatMarkdown);

export default function NodePageGAT() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/17-graph-neural-networks" className={styles.back}>
          ← 返回图神经网络(GNN)
        </Link>
        <h1 className={styles.title}>GAT (2018)</h1>
        <div className={styles.metaLine}>
          作者:Petar Veličković · Guillem Cucurull · Arantxa Casanova · Adriana Romero · Pietro Liò · Yoshua Bengio
        </div>
        <div className={styles.metaLine}>论文:Graph Attention Networks</div>
        <p className={styles.keyIdea}>
          用可学习的 attention 权重替代 GCN 里固定的度数归一化系数,让模型隐式学会"哪个邻居更重要"
        </p>
      </section>

      <section className={styles.stage}>
        <AttentionCoeffStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <MaskSoftmaxStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <MultiHeadStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={GAT_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={GAT_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={GAT_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
