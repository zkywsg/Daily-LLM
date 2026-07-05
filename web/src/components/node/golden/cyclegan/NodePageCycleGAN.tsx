import { Link } from "react-router";

import cycleganMarkdown from "../../../../../../04-gan/03-cyclegan.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, CYCLEGAN_SOURCE_PATH } from "./lib/prose";
import { DualMappingStage } from "./stages/DualMappingStage";
import { CycleConsistencyStage } from "./stages/CycleConsistencyStage";
import { EngineeringStage } from "./stages/EngineeringStage";
import { CityscapesCompareChart } from "./widgets/CityscapesCompareChart";
import styles from "./NodePageCycleGAN.module.css";

const prose = extractProse(cycleganMarkdown);

export default function NodePageCycleGAN() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/04-gan" className={styles.back}>
          ← 返回 GAN
        </Link>
        <h1 className={styles.title}>CycleGAN (2017)</h1>
        <div className={styles.metaLine}>
          作者:Jun-Yan Zhu · Taesung Park · Phillip Isola · Alexei A. Efros
        </div>
        <div className={styles.metaLine}>
          论文:Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
        </div>
        <p className={styles.keyIdea}>
          用 cycle consistency loss 实现无配对图像翻译——两个 G 互相 mapping(X→Y 和 Y→X),
          要求 F(G(x)) ≈ x;不需要成对训练数据就能做马↔斑马、夏↔冬、照片↔画风的转换
        </p>
      </section>

      <section className={styles.stage}>
        <DualMappingStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <CycleConsistencyStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <EngineeringStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <CityscapesCompareChart />
          <MarkdownRenderer markdown={prose.performance} sourcePath={CYCLEGAN_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={CYCLEGAN_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={CYCLEGAN_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
