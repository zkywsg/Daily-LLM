import { Link } from "react-router";

import ropeMarkdown from "../../../../../../05-transformer/04-rope.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, ROPE_SOURCE_PATH } from "./lib/prose";
import { RotationStage } from "./stages/RotationStage";
import { FrequencyStage } from "./stages/FrequencyStage";
import { EngineeringStage } from "./stages/EngineeringStage";
import styles from "./NodePageRoPE.module.css";

const prose = extractProse(ropeMarkdown);

export default function NodePageRoPE() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/05-transformer" className={styles.back}>
          ← 返回 Transformer
        </Link>
        <h1 className={styles.title}>RoPE (2021)</h1>
        <div className={styles.metaLine}>
          作者:Jianlin Su(苏剑林)· Yu Lu · Shengfeng Pan · Bo Wen · Yunfeng Liu
        </div>
        <div className={styles.metaLine}>
          论文:RoFormer: Enhanced Transformer with Rotary Position Embedding
        </div>
        <p className={styles.keyIdea}>
          把位置信息编码进 Q/K 的旋转里而不是加在 token embedding 上,
          attention 内积天然只依赖相对位置 · 表达力+外推性+实现复杂度三维帕累托最优 ·
          LLaMA/PaLM/Mistral/Qwen 等几乎所有现代 LLM 的事实标准
        </p>
      </section>

      <section className={styles.stage}>
        <RotationStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <FrequencyStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <EngineeringStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>Pre-LN 和 RMSNorm:配套的现代化</h2>
          <MarkdownRenderer markdown={prose.modernization} sourcePath={ROPE_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={ROPE_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={ROPE_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
