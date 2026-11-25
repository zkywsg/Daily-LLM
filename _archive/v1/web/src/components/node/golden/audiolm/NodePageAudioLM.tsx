import { Link } from "react-router";
import audiolmMarkdown from "../../../../../../18-speech-audio/04-audiolm.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, AUDIOLM_SOURCE_PATH } from "./lib/prose";
import { SemanticTokenStage } from "./stages/SemanticTokenStage";
import { RvqStage } from "./stages/RvqStage";
import { CascadeStage } from "./stages/CascadeStage";
import styles from "./NodePageAudioLM.module.css";

const prose = extractProse(audiolmMarkdown);

export default function NodePageAudioLM() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/18-speech-audio" className={styles.back}>
          ← 返回语音/音频模型
        </Link>
        <h1 className={styles.title}>AudioLM (2022)</h1>
        <div className={styles.metaLine}>作者:Zalán Borsos · Raphaël Marinier · Damien Vincent 等(Google)</div>
        <div className={styles.metaLine}>论文:AudioLM: a Language Modeling Approach to Audio Generation</div>
        <p className={styles.keyIdea}>
          把音频离散化成语义 token 和声学 token 两级表示,用语言模型对两级 token 做层级式 next-token 预测
        </p>
      </section>

      <section className={styles.stage}>
        <SemanticTokenStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <RvqStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <CascadeStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={AUDIOLM_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={AUDIOLM_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={AUDIOLM_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
