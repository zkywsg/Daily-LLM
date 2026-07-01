import { Link } from "react-router";

import flamingoMarkdown from "../../../../../../09-multimodal-clip/03-flamingo.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, FLAMINGO_SOURCE_PATH } from "./lib/prose";
import { FrozenLlmStage } from "./stages/FrozenLlmStage";
import { BridgeStage } from "./stages/BridgeStage";
import { M3wStage } from "./stages/M3wStage";
import { VlmCompareTable } from "./widgets/VlmCompareTable";
import styles from "./NodePageFlamingo.module.css";

const prose = extractProse(flamingoMarkdown);

export default function NodePageFlamingo() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/09-multimodal-clip" className={styles.back}>
          ← 返回 Multimodal / CLIP
        </Link>
        <h1 className={styles.title}>Flamingo (2022)</h1>
        <div className={styles.metaLine}>
          作者:Jean-Baptiste Alayrac · Jeff Donahue · Pauline Luc · Antoine Miech et al. · DeepMind
        </div>
        <div className={styles.metaLine}>
          论文:Flamingo: a Visual Language Model for Few-Shot Learning
        </div>
        <p className={styles.keyIdea}>
          冻结大 LLM(Chinchilla 70B)+ Perceiver Resampler 视觉适配 + 间隔 cross-attention 注入,
          8 例 in-context 学新视觉任务 · 16 个 benchmark 4-shot SOTA ·
          定义了"冻结 LLM + 视觉接口"的现代 multimodal LLM 范式
        </p>
      </section>

      <section className={styles.stage}>
        <FrozenLlmStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <BridgeStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <M3wStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>三条 VLM 路线对比</h2>
          <VlmCompareTable />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.icl} sourcePath={FLAMINGO_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={FLAMINGO_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={FLAMINGO_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={FLAMINGO_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
