import { Link } from "react-router";

import loraMarkdown from "../../../../../../11-peft-lora/03-lora.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, LORA_SOURCE_PATH } from "./lib/prose";
import { LowRankDecompositionStage } from "./stages/LowRankDecompositionStage";
import { InitAndParamAccountStage } from "./stages/InitAndParamAccountStage";
import { InferenceMergeStage } from "./stages/InferenceMergeStage";
import styles from "./NodePageLoRA.module.css";

const prose = extractProse(loraMarkdown);

export default function NodePageLoRA() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/11-peft-lora" className={styles.back}>
          ← 返回 参数高效微调 (PEFT)
        </Link>
        <h1 className={styles.title}>LoRA (2021)</h1>
        <div className={styles.metaLine}>
          作者:Edward Hu · Yelong Shen · Phillip Wallis · Zeyuan Allen-Zhu · Yuanzhi Li · Shean Wang · Lu Wang · Weizhu Chen · Microsoft
        </div>
        <div className={styles.metaLine}>
          论文:LoRA: Low-Rank Adaptation of Large Language Models
        </div>
        <p className={styles.keyIdea}>
          冻结 W₀ 只学一对低秩矩阵 BA,trainable 参数缩 1000×,
          推理时合并回 W₀ 零延迟 —— 大模型时代的 PEFT 工业标准
        </p>
      </section>

      <section className={styles.stage}>
        <LowRankDecompositionStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <InitAndParamAccountStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <InferenceMergeStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={LORA_SOURCE_PATH} />
        </div>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={LORA_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={LORA_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
