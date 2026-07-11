import { Link } from "react-router";

import convnextMarkdown from "../../../../../../01-cnn/08-convnext.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, CONVNEXT_SOURCE_PATH } from "./lib/prose";
import { TrainingRecipeStage } from "./stages/TrainingRecipeStage";
import { StructuralModernizationStage } from "./stages/StructuralModernizationStage";
import { NormActStage } from "./stages/NormActStage";
import styles from "./NodePageConvnext.module.css";

const prose = extractProse(convnextMarkdown);

export default function NodePageConvnext() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/01-cnn" className={styles.back}>
          ← 返回 CNN
        </Link>
        <h1 className={styles.title}>ConvNeXt (2022)</h1>
        <div className={styles.metaLine}>
          作者:Zhuang Liu · Hanzi Mao · Chao-Yuan Wu · Christoph Feichtenhofer · Trevor Darrell · Saining Xie
        </div>
        <div className={styles.metaLine}>论文:A ConvNet for the 2020s</div>
        <p className={styles.keyIdea}>
          把 ViT 的所有现代化设计选择(大 kernel · LayerNorm · GELU · 强增强)逐项搬回
          ResNet,CNN 反超 ViT
        </p>
      </section>

      <section className={styles.stage}>
        <TrainingRecipeStage intuitionProse={prose.intuition} mechanism1Prose={prose.mechanism1} />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <StructuralModernizationStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <NormActStage mechanism3Prose={prose.mechanism3} synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <MarkdownRenderer markdown={prose.performance} sourcePath={CONVNEXT_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={CONVNEXT_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={CONVNEXT_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
