import { Link } from "react-router";

import qloraMarkdown from "../../../../../../11-peft-lora/04-qlora.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, QLORA_SOURCE_PATH } from "./lib/prose";
import { Nf4Stage } from "./stages/Nf4Stage";
import { DoubleQuantStage } from "./stages/DoubleQuantStage";
import { PagedOptimizerStage } from "./stages/PagedOptimizerStage";
import { GuanacoScoreChart } from "./widgets/GuanacoScoreChart";
import styles from "./NodePageQLoRA.module.css";

const prose = extractProse(qloraMarkdown);

export default function NodePageQLoRA() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/11-peft-lora" className={styles.back}>
          ← 返回 PEFT / LoRA
        </Link>
        <h1 className={styles.title}>QLoRA (2023)</h1>
        <div className={styles.metaLine}>
          作者:Tim Dettmers · Artidoro Pagnoni · Ari Holtzman · Luke Zettlemoyer(华盛顿大学)
        </div>
        <div className={styles.metaLine}>
          论文:QLoRA: Efficient Finetuning of Quantized LLMs
        </div>
        <p className={styles.keyIdea}>
          Base 模型量化到 4-bit NF4 + LoRA 微调,配 double quantization + paged
          optimizer — 让 65B 模型能在单卡 48GB GPU 上微调,
          LLM 微调从"机构级"工程降到"个人级"工程
        </p>
      </section>

      <section className={styles.stage}>
        <Nf4Stage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <DoubleQuantStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <PagedOptimizerStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <GuanacoScoreChart />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={QLORA_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={QLORA_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={QLORA_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
