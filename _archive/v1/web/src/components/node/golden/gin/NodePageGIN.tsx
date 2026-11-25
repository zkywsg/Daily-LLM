import { Link } from "react-router";
import ginMarkdown from "../../../../../../17-graph-neural-networks/04-gin.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, GIN_SOURCE_PATH } from "./lib/prose";
import { InjectivityStage } from "./stages/InjectivityStage";
import { EpsilonStage } from "./stages/EpsilonStage";
import { WLStage } from "./stages/WLStage";
import styles from "./NodePageGIN.module.css";

const prose = extractProse(ginMarkdown);

export default function NodePageGIN() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/17-graph-neural-networks" className={styles.back}>
          ← 返回图神经网络(GNN)
        </Link>
        <h1 className={styles.title}>GIN (2019)</h1>
        <div className={styles.metaLine}>作者:Keyulu Xu · Weihua Hu · Jure Leskovec · Stefanie Jegelka</div>
        <div className={styles.metaLine}>论文:How Powerful are Graph Neural Networks?</div>
        <p className={styles.keyIdea}>
          用 Weisfeiler-Lehman 图同构测试给 GNN 表达力定理上界,提出 sum 聚合 + MLP 的 GIN 达到 WL test 同等的最大可能表达力
        </p>
      </section>

      <section className={styles.stage}>
        <InjectivityStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <EpsilonStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <WLStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={GIN_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={GIN_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={GIN_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
