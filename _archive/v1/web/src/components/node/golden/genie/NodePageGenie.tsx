import { Link } from "react-router";
import genieMarkdown from "../../../../../../16-world-models/05-genie.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, GENIE_SOURCE_PATH } from "./lib/prose";
import { TokenizerStage } from "./stages/TokenizerStage";
import { LamStage } from "./stages/LamStage";
import { PlayStage } from "./stages/PlayStage";
import styles from "./NodePageGenie.module.css";

const prose = extractProse(genieMarkdown);

export default function NodePageGenie() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/16-world-models" className={styles.back}>
          ← 返回世界模型 / 视频生成
        </Link>
        <h1 className={styles.title}>Genie (2024)</h1>
        <div className={styles.metaLine}>作者:Jake Bruce · Michael Dennis · Ashley Edwards 等(DeepMind)</div>
        <div className={styles.metaLine}>论文:Genie: Generative Interactive Environments</div>
        <p className={styles.keyIdea}>
          无监督地从海量无标注互联网视频里学出逐帧可控制的生成式环境:隐式学习出离散的 latent action 空间
        </p>
      </section>

      <section className={styles.stage}>
        <TokenizerStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <LamStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <PlayStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={GENIE_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={GENIE_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={GENIE_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
