import { Link } from "react-router";

import swinMarkdown from "../../../../../../08-vit/03-swin.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, SWIN_SOURCE_PATH } from "./lib/prose";
import { WindowedAttnStage } from "./stages/WindowedAttnStage";
import { ShiftedWindowStage } from "./stages/ShiftedWindowStage";
import { PatchMergingStage } from "./stages/PatchMergingStage";
import { TaskCompareBars } from "./widgets/TaskCompareBars";
import styles from "./NodePageSwin.module.css";

const prose = extractProse(swinMarkdown);

export default function NodePageSwin() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/08-vit" className={styles.back}>
          ← 返回 ViT 视觉 Transformer
        </Link>
        <h1 className={styles.title}>Swin Transformer (2021)</h1>
        <div className={styles.metaLine}>
          作者:Ze Liu · Yutong Lin · Yue Cao · Han Hu et al. · Microsoft Research Asia
        </div>
        <div className={styles.metaLine}>
          论文:Swin Transformer: Hierarchical Vision Transformer using Shifted Windows(ICCV 2021 Best Paper)
        </div>
        <p className={styles.keyIdea}>
          用 shifted window attention 把复杂度从 O(N²) 降到 O(N)+ 层级化下采样产出多尺度特征图,
          让 ViT 第一次能直接做 detection / segmentation —
          Swin-T 击败 ResNet-50 达 7.4 mAP
        </p>
      </section>

      <section className={styles.stage}>
        <WindowedAttnStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <ShiftedWindowStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <PatchMergingStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>模型规格</h2>
          <MarkdownRenderer markdown={prose.modelSpecs} sourcePath={SWIN_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>性能:多任务全面 SOTA</h2>
          <TaskCompareBars />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={SWIN_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={SWIN_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={SWIN_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={SWIN_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
