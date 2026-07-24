import { Link } from "react-router";
import gamengenMarkdown from "../../../../../../16-world-models/06-gamengen.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, GAMENGEN_SOURCE_PATH } from "./lib/prose";
import { RlDataStage } from "./stages/RlDataStage";
import { DiffusionPredictStage } from "./stages/DiffusionPredictStage";
import { NoiseAugmentationStage } from "./stages/NoiseAugmentationStage";
import styles from "./NodePageGameNGen.module.css";

const prose = extractProse(gamengenMarkdown);

export default function NodePageGameNGen() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/16-world-models" className={styles.back}>
          ← 返回世界模型 / 视频生成
        </Link>
        <h1 className={styles.title}>GameNGen (2024)</h1>
        <div className={styles.metaLine}>作者:Dani Valevski · Yaniv Leviathan · Moab Arar · Shlomi Fruchter</div>
        <div className={styles.metaLine}>论文:Diffusion Models Are Real-Time Game Engines</div>
        <p className={styles.keyIdea}>
          用条件 diffusion 模型完全替代传统游戏引擎的渲染循环,实时交互式生成可玩的 DOOM 画面
        </p>
      </section>

      <section className={styles.stage}>
        <RlDataStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <DiffusionPredictStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <NoiseAugmentationStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={GAMENGEN_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={GAMENGEN_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={GAMENGEN_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
