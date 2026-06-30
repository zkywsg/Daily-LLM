import { Link } from "react-router";

import gpt2Markdown from "../../../../../../07-gpt-scaling/02-gpt2.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, GPT2_SOURCE_PATH } from "./lib/prose";
import { ScaleStage } from "./stages/ScaleStage";
import { ZeroShotPromptStage } from "./stages/ZeroShotPromptStage";
import { PreLnSamplingStage } from "./stages/PreLnSamplingStage";
import styles from "./NodePageGPT2.module.css";

const prose = extractProse(gpt2Markdown);

export default function NodePageGPT2() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/07-gpt-scaling" className={styles.back}>
          ← 返回 GPT scaling
        </Link>
        <h1 className={styles.title}>GPT-2 (2019)</h1>
        <div className={styles.metaLine}>
          作者:Alec Radford · Jeffrey Wu · Rewon Child · David Luan · Dario Amodei · Ilya Sutskever
        </div>
        <div className={styles.metaLine}>
          论文:Language Models are Unsupervised Multitask Learners
        </div>
        <p className={styles.keyIdea}>
          把 GPT-1 推到 1.5B + WebText 40B,zero-shot 多任务能力首次涌现 —
          LM 第一次显示出 "通用学习器" 形态,Pre-LN 让 48 层稳定训出来,
          是 2022 之后 prompt engineering 整个学科的种子
        </p>
      </section>

      <section className={styles.stage}>
        <ScaleStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <ZeroShotPromptStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <PreLnSamplingStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={GPT2_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={GPT2_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={GPT2_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
