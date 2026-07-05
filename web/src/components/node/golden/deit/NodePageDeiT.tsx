import { Link } from "react-router";

import deitMarkdown from "../../../../../../08-vit/02-deit.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, DEIT_SOURCE_PATH } from "./lib/prose";
import { RecipeStage } from "./stages/RecipeStage";
import { DistillationStage } from "./stages/DistillationStage";
import { LongTrainingStage } from "./stages/LongTrainingStage";
import { DataEfficiencyChart } from "./widgets/DataEfficiencyChart";
import styles from "./NodePageDeiT.module.css";

const prose = extractProse(deitMarkdown);

export default function NodePageDeiT() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/08-vit" className={styles.back}>
          ← 返回 ViT
        </Link>
        <h1 className={styles.title}>DeiT (2021)</h1>
        <div className={styles.metaLine}>
          作者:Hugo Touvron · Matthieu Cord · Matthijs Douze · Francisco Massa ·
          Alexandre Sablayrolles · Hervé Jégou · Facebook AI Research
        </div>
        <div className={styles.metaLine}>
          论文:Training data-efficient image transformers &amp; distillation through attention
        </div>
        <p className={styles.keyIdea}>
          用 distillation token + 强增强 + AdamW + 蒸馏让 ViT 在 ImageNet-1K 上从零训练击败 ResNet,
          不再依赖 JFT-300M,把 ViT 带给学界
        </p>
      </section>

      <section className={styles.stage}>
        <RecipeStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <DistillationStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <LongTrainingStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        {prose.previousWork && (
          <div className={styles.footerSection}>
            <h2>前作进展</h2>
            <MarkdownRenderer markdown={prose.previousWork} sourcePath={DEIT_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>性能 vs 资源</h2>
          <DataEfficiencyChart />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={DEIT_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={DEIT_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={DEIT_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={DEIT_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
