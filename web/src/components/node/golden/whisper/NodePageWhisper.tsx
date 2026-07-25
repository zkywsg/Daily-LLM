import { Link } from "react-router";
import whisperMarkdown from "../../../../../../18-speech-audio/03-whisper.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, WHISPER_SOURCE_PATH } from "./lib/prose";
import { SpectrogramStage } from "./stages/SpectrogramStage";
import { DataFilterStage } from "./stages/DataFilterStage";
import { TaskPrefixStage } from "./stages/TaskPrefixStage";
import styles from "./NodePageWhisper.module.css";

const prose = extractProse(whisperMarkdown);

export default function NodePageWhisper() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/18-speech-audio" className={styles.back}>
          ← 返回语音/音频模型
        </Link>
        <h1 className={styles.title}>Whisper (2022)</h1>
        <div className={styles.metaLine}>作者:Alec Radford · Jong Wook Kim · Tao Xu · Greg Brockman · Christine McLeavey · Ilya Sutskever</div>
        <div className={styles.metaLine}>论文:Robust Speech Recognition via Large-Scale Weak Supervision</div>
        <p className={styles.keyIdea}>
          68 万小时弱监督多语言多任务数据 + 标准 Transformer encoder-decoder,零样本鲁棒性接近或超过针对特定数据集微调的模型
        </p>
      </section>

      <section className={styles.stage}>
        <SpectrogramStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <DataFilterStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <TaskPrefixStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={WHISPER_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={WHISPER_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={WHISPER_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
