import { Link } from "react-router";
import gcnMarkdown from "../../../../../../17-graph-neural-networks/01-gcn.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, GCN_SOURCE_PATH } from "./lib/prose";
import { SelfLoopStage } from "./stages/SelfLoopStage";
import { NormalizationStage } from "./stages/NormalizationStage";
import { PropagationStage } from "./stages/PropagationStage";
import styles from "./NodePageGCN.module.css";

const prose = extractProse(gcnMarkdown);

export default function NodePageGCN() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/17-graph-neural-networks" className={styles.back}>
          ← 返回图神经网络(GNN)
        </Link>
        <h1 className={styles.title}>GCN (2017)</h1>
        <div className={styles.metaLine}>作者:Thomas N. Kipf · Max Welling</div>
        <div className={styles.metaLine}>论文:Semi-Supervised Classification with Graph Convolutional Networks</div>
        <p className={styles.keyIdea}>
          把谱图卷积简化到一阶邻域聚合,一层 D̃^(-1/2) Ã D̃^(-1/2) H W 传播规则定义了"现代 GNN"这个范式的起点
        </p>
      </section>

      <section className={styles.stage}>
        <SelfLoopStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <NormalizationStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <PropagationStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={GCN_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={GCN_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={GCN_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
