import { Link } from "react-router";
import worldModelsMarkdown from "../../../../../../16-world-models/01-world-models.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, WORLD_MODELS_SOURCE_PATH } from "./lib/prose";
import { VisionStage } from "./stages/VisionStage";
import { MemoryStage } from "./stages/MemoryStage";
import { ControllerStage } from "./stages/ControllerStage";
import styles from "./NodePageWorldModels.module.css";

const prose = extractProse(worldModelsMarkdown);

export default function NodePageWorldModels() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/16-world-models" className={styles.back}>
          ← 返回世界模型 / 视频生成
        </Link>
        <h1 className={styles.title}>World Models (2018)</h1>
        <div className={styles.metaLine}>作者:David Ha · Jürgen Schmidhuber</div>
        <div className={styles.metaLine}>论文:World Models</div>
        <p className={styles.keyIdea}>
          把智能体拆成 V(VAE 视觉压缩)+ M(MDN-RNN 时序预测)+ C(极小线性控制器)三部分,C 完全在 M 生成的"梦境"里训练
        </p>
      </section>

      <section className={styles.stage}>
        <VisionStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <MemoryStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <ControllerStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={WORLD_MODELS_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={WORLD_MODELS_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={WORLD_MODELS_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
