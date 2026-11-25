import { Link } from "react-router";
import dreamerv3Markdown from "../../../../../../16-world-models/03-dreamerv3.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, DREAMERV3_SOURCE_PATH } from "./lib/prose";
import { RssmStage } from "./stages/RssmStage";
import { SymlogStage } from "./stages/SymlogStage";
import { ImaginationStage } from "./stages/ImaginationStage";
import styles from "./NodePageDreamerV3.module.css";

const prose = extractProse(dreamerv3Markdown);

export default function NodePageDreamerV3() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/16-world-models" className={styles.back}>
          ← 返回世界模型 / 视频生成
        </Link>
        <h1 className={styles.title}>DreamerV3 (2023)</h1>
        <div className={styles.metaLine}>作者:Danijar Hafner · Jurgis Pasukonis · Jimmy Ba · Timothy Lillicrap</div>
        <div className={styles.metaLine}>论文:Mastering Diverse Domains through World Models</div>
        <p className={styles.keyIdea}>
          把 latent imagination 式的 model-based RL 规模化到跨领域通吃,固定同一套超参数不调参就能匹配甚至超过各领域的 model-free SOTA
        </p>
      </section>

      <section className={styles.stage}>
        <RssmStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <SymlogStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <ImaginationStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={DREAMERV3_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={DREAMERV3_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={DREAMERV3_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
