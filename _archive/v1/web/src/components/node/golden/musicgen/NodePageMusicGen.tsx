import { Link } from "react-router";
import musicgenMarkdown from "../../../../../../18-speech-audio/05-musicgen.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, MUSICGEN_SOURCE_PATH } from "./lib/prose";
import { EncodecStage } from "./stages/EncodecStage";
import { DelayPatternStage } from "./stages/DelayPatternStage";
import { ConditionStage } from "./stages/ConditionStage";
import styles from "./NodePageMusicGen.module.css";

const prose = extractProse(musicgenMarkdown);

export default function NodePageMusicGen() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/18-speech-audio" className={styles.back}>
          ← 返回语音/音频模型
        </Link>
        <h1 className={styles.title}>MusicGen (2023)</h1>
        <div className={styles.metaLine}>作者:Jade Copet · Felix Kreuk · Itai Gat · Tal Remez · David Kant · Gabriel Synnaeve · Yossi Adi · Alexandre Défossez</div>
        <div className={styles.metaLine}>论文:Simple and Controllable Music Generation</div>
        <p className={styles.keyIdea}>
          单阶段 Transformer decoder + EnCodec 码本交错技巧,把 AudioLM/MusicLM 的多阶段级联简化成单阶段模型
        </p>
      </section>

      <section className={styles.stage}>
        <EncodecStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <DelayPatternStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <ConditionStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={MUSICGEN_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={MUSICGEN_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={MUSICGEN_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
