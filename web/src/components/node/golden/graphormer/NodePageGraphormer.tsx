import { Link } from "react-router";
import graphormerMarkdown from "../../../../../../17-graph-neural-networks/05-graphormer.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, GRAPHORMER_SOURCE_PATH } from "./lib/prose";
import { CentralityStage } from "./stages/CentralityStage";
import { SpatialStage } from "./stages/SpatialStage";
import { EdgeStage } from "./stages/EdgeStage";
import styles from "./NodePageGraphormer.module.css";

const prose = extractProse(graphormerMarkdown);

export default function NodePageGraphormer() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/17-graph-neural-networks" className={styles.back}>
          ← 返回图神经网络(GNN)
        </Link>
        <h1 className={styles.title}>Graphormer (2021)</h1>
        <div className={styles.metaLine}>
          作者:Chengxuan Ying · Tianle Cai · Shengjie Luo · Shuxin Zheng · Guolin Ke · Di He · Yanming Shen · Tie-Yan Liu
        </div>
        <div className={styles.metaLine}>论文:Do Transformers Really Perform Bad for Graph Representation?</div>
        <p className={styles.keyIdea}>
          中心性编码 + 空间编码(最短路径距离)+ 边编码把图结构信息直接注入 attention,用全局注意力替代逐跳消息传递
        </p>
      </section>

      <section className={styles.stage}>
        <CentralityStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <SpatialStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <EdgeStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={GRAPHORMER_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={GRAPHORMER_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={GRAPHORMER_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
