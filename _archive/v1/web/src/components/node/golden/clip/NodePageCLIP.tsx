import { Link } from "react-router";

import clipMarkdown from "../../../../../../09-multimodal-clip/01-clip.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, CLIP_SOURCE_PATH } from "./lib/prose";
import { DualEncoderStage } from "./stages/DualEncoderStage";
import { ContrastiveStage } from "./stages/ContrastiveStage";
import { ZeroShotStage } from "./stages/ZeroShotStage";
import styles from "./NodePageCLIP.module.css";

const prose = extractProse(clipMarkdown);

export default function NodePageCLIP() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/09-multimodal-clip" className={styles.back}>
          ← 返回 多模态对齐 (CLIP / 跨模态)
        </Link>
        <h1 className={styles.title}>CLIP (2021)</h1>
        <div className={styles.metaLine}>
          作者:Alec Radford · Jong Wook Kim · Chris Hallacy · ... · Ilya Sutskever · OpenAI
        </div>
        <div className={styles.metaLine}>
          论文:Learning Transferable Visual Models From Natural Language Supervision
        </div>
        <p className={styles.keyIdea}>
          400M 图文对 + 双塔 contrastive 学一个共享 embedding 空间,
          再用 "a photo of a {"{"}class{"}"}" prompt 把分类变成 zero-shot 检索
        </p>
      </section>

      <section className={styles.stage}>
        <DualEncoderStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <ContrastiveStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <ZeroShotStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={CLIP_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={CLIP_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={CLIP_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={CLIP_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
