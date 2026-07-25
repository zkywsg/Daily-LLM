import { Link } from "react-router";
import wav2vec2Markdown from "../../../../../../18-speech-audio/01-wav2vec2.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, WAV2VEC2_SOURCE_PATH } from "./lib/prose";
import { FeatureEncoderStage } from "./stages/FeatureEncoderStage";
import { QuantizeStage } from "./stages/QuantizeStage";
import { MaskedPredictStage } from "./stages/MaskedPredictStage";
import styles from "./NodePageWav2Vec2.module.css";

const prose = extractProse(wav2vec2Markdown);

export default function NodePageWav2Vec2() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/18-speech-audio" className={styles.back}>
          ← 返回语音/音频模型
        </Link>
        <h1 className={styles.title}>Wav2Vec 2.0 (2020)</h1>
        <div className={styles.metaLine}>作者:Alexei Baevski · Henry Zhou · Abdelrahman Mohamed · Michael Auli</div>
        <div className={styles.metaLine}>论文:wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations</div>
        <p className={styles.keyIdea}>
          CNN 特征编码器 + 可学习量化模块生成离散对比目标 + Transformer 掩码预测,用对比学习从原始波形自监督学到可迁移的语音表征
        </p>
      </section>

      <section className={styles.stage}>
        <FeatureEncoderStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <QuantizeStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <MaskedPredictStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={WAV2VEC2_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={WAV2VEC2_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={WAV2VEC2_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
