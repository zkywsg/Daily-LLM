import { Link } from "react-router";

import prefixTuningMarkdown from "../../../../../../11-peft-lora/02-prefix-tuning.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, PREFIX_TUNING_SOURCE_PATH } from "./lib/prose";
import { LayerwisePrefixStage } from "./stages/LayerwisePrefixStage";
import { MlpReparamStage } from "./stages/MlpReparamStage";
import { FrozenBaseCompareStage } from "./stages/FrozenBaseCompareStage";
import { PerformanceCompareChart } from "./widgets/PerformanceCompareChart";
import { ScalingTrendChart } from "./widgets/ScalingTrendChart";
import styles from "./NodePagePrefixTuning.module.css";

const prose = extractProse(prefixTuningMarkdown);

export default function NodePagePrefixTuning() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/11-peft-lora" className={styles.back}>
          ← 返回 PEFT / LoRA
        </Link>
        <h1 className={styles.title}>Prefix Tuning (2021)</h1>
        <div className={styles.metaLine}>作者:Xiang Lisa Li · Percy Liang(Stanford)</div>
        <div className={styles.metaLine}>
          论文:Prefix-Tuning: Optimizing Continuous Prompts for Generation
        </div>
        <p className={styles.keyIdea}>
          在每层 attention 的 K/V 前面加一段可学习的"soft prefix" embedding,
          base 模型完全冻结,只训这段 prefix(~0.1% 参数);极致参数效率,
          1000+ task 用同一 base 共享
        </p>
      </section>

      <section className={styles.stage}>
        <LayerwisePrefixStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <MlpReparamStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <FrozenBaseCompareStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <PerformanceCompareChart />
          <div style={{ marginTop: "var(--space-6)" }}>
            <ScalingTrendChart />
          </div>
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={PREFIX_TUNING_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={PREFIX_TUNING_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={PREFIX_TUNING_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
