import { Link } from "react-router";

import gpt1Markdown from "../../../../../../07-gpt-scaling/01-gpt1.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, GPT1_SOURCE_PATH } from "./lib/prose";
import { DecoderStage } from "./stages/DecoderStage";
import { PretrainStage } from "./stages/PretrainStage";
import { UnifiedInterfaceStage } from "./stages/UnifiedInterfaceStage";
import { BenchmarkGainBars } from "./widgets/BenchmarkGainBars";
import styles from "./NodePageGPT1.module.css";

const prose = extractProse(gpt1Markdown);

export default function NodePageGPT1() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/07-gpt-scaling" className={styles.back}>
          ← 返回 GPT scaling
        </Link>
        <h1 className={styles.title}>GPT-1 (2018)</h1>
        <div className={styles.metaLine}>
          作者:Alec Radford · Karthik Narasimhan · Tim Salimans · Ilya Sutskever · OpenAI
        </div>
        <div className={styles.metaLine}>
          论文:Improving Language Understanding by Generative Pre-Training
        </div>
        <p className={styles.keyIdea}>
          Decoder-only Transformer + 无监督自回归预训练 + 统一任务接口,
          第一次系统跑通"预训练 + 微调"范式 · 12 个 benchmark 拿 9 个 SOTA ·
          定义了之后 5 年 LLM 路线,连 GPT-3 的 in-context learning 都是它的极端化
        </p>
      </section>

      <section className={styles.stage}>
        <DecoderStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <PretrainStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <UnifiedInterfaceStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <BenchmarkGainBars />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={GPT1_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={GPT1_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={GPT1_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={GPT1_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
