import { Link } from "react-router";
import graphsageMarkdown from "../../../../../../17-graph-neural-networks/02-graphsage.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, GRAPHSAGE_SOURCE_PATH } from "./lib/prose";
import { SamplingStage } from "./stages/SamplingStage";
import { AggregatorStage } from "./stages/AggregatorStage";
import { InductiveStage } from "./stages/InductiveStage";
import styles from "./NodePageGraphSAGE.module.css";

const prose = extractProse(graphsageMarkdown);

export default function NodePageGraphSAGE() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/17-graph-neural-networks" className={styles.back}>
          ← 返回图神经网络(GNN)
        </Link>
        <h1 className={styles.title}>GraphSAGE (2017)</h1>
        <div className={styles.metaLine}>作者:William L. Hamilton · Rex Ying · Jure Leskovec</div>
        <div className={styles.metaLine}>论文:Inductive Representation Learning on Large Graphs</div>
        <p className={styles.keyIdea}>
          SAmple + aggreGatE:固定大小邻域采样 + 可学习聚合函数,让 GNN 第一次能泛化到训练时没见过的节点/图
        </p>
      </section>

      <section className={styles.stage}>
        <SamplingStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <AggregatorStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <InductiveStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={GRAPHSAGE_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={GRAPHSAGE_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={GRAPHSAGE_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
