import { Link } from "react-router";
import vdmMarkdown from "../../../../../../16-world-models/02-video-diffusion-models.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, VDM_SOURCE_PATH } from "./lib/prose";
import { FactorizedStage } from "./stages/FactorizedStage";
import { JointTrainingStage } from "./stages/JointTrainingStage";
import { ExtensionStage } from "./stages/ExtensionStage";
import styles from "./NodePageVideoDiffusionModels.module.css";

const prose = extractProse(vdmMarkdown);

export default function NodePageVideoDiffusionModels() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/16-world-models" className={styles.back}>
          ← 返回世界模型 / 视频生成
        </Link>
        <h1 className={styles.title}>Video Diffusion Models (2022)</h1>
        <div className={styles.metaLine}>作者:Jonathan Ho · Tim Salimans · Alexey Gritsenko · William Chan · Mohammad Norouzi · David J. Fleet</div>
        <div className={styles.metaLine}>论文:Video Diffusion Models</div>
        <p className={styles.keyIdea}>
          把 DDPM 的去噪框架从图像推广到视频:时空分解卷积代替昂贵的 3D 卷积,图像/视频联合训练复用大规模图像数据
        </p>
      </section>

      <section className={styles.stage}>
        <FactorizedStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <JointTrainingStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <ExtensionStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={VDM_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={VDM_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={VDM_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
