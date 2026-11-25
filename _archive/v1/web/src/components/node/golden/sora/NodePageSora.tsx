import { Link } from "react-router";
import soraMarkdown from "../../../../../../16-world-models/04-sora.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, SORA_SOURCE_PATH } from "./lib/prose";
import { PatchifyStage } from "./stages/PatchifyStage";
import { ScalingStage } from "./stages/ScalingStage";
import { NativeResolutionStage } from "./stages/NativeResolutionStage";
import styles from "./NodePageSora.module.css";

const prose = extractProse(soraMarkdown);

export default function NodePageSora() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/16-world-models" className={styles.back}>
          ← 返回世界模型 / 视频生成
        </Link>
        <h1 className={styles.title}>Sora (2024)</h1>
        <div className={styles.metaLine}>作者:OpenAI</div>
        <div className={styles.metaLine}>论文:Video generation models as world simulators</div>
        <p className={styles.keyIdea}>
          把 DiT 规模化到分钟级、多分辨率、多时长连贯视频:用 spacetime patches 统一表示不同长宽比/时长的时空数据
        </p>
      </section>

      <section className={styles.stage}>
        <PatchifyStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <ScalingStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <NativeResolutionStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={SORA_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={SORA_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={SORA_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
