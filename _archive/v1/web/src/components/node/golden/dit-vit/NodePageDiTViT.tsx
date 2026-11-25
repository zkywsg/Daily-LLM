import { Link } from "react-router";

import ditMarkdown from "../../../../../../08-vit/04-dit.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, DIT_SOURCE_PATH } from "../dit/lib/prose";
import { PatchifyStage } from "../dit/stages/PatchifyStage";
import { AdaLNStage } from "../dit/stages/AdaLNStage";
import { ScalingStage } from "../dit/stages/ScalingStage";
import { SotaCompareBars } from "../dit/widgets/SotaCompareBars";
import { DiTFamilyTimeline } from "../dit/widgets/DiTFamilyTimeline";
import styles from "./NodePageDiTViT.module.css";

// 内容与 10-diffusion/05-dit.md 完全一致(同一篇 DiT 论文,在两个家族的
// 时间线里都占一个条目),只有 frontmatter 的 family/order 不同 —— 因此
// 复用 dit/ 目录下已有的 stages/widgets/prose 逻辑,这里只是一个换了
// back-link 和渐变色(ViT 翠 var(--family-08))的独立入口,让 08-vit
// 家族内的导航(back-link / prev-next)始终留在 08-vit 而不是被带去
// 10-diffusion。
const prose = extractProse(ditMarkdown);

export default function NodePageDiTViT() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/08-vit" className={styles.back}>
          ← 返回 视觉 Transformer (ViT)
        </Link>
        <h1 className={styles.title}>DiT (2022)</h1>
        <div className={styles.metaLine}>
          作者:William Peebles · Saining Xie · UC Berkeley
        </div>
        <div className={styles.metaLine}>
          论文:Scalable Diffusion Models with Transformers
        </div>
        <p className={styles.keyIdea}>
          把 diffusion U-Net 整个换成 Transformer · patchify + adaLN-Zero + scaling 三件套,
          FLOPs 越大 FID 越低单调成立 · DiT-XL/2 拿 ImageNet 256 SOTA 2.27,
          为 Sora / SD3 / FLUX 提供骨架
        </p>
      </section>

      <section className={styles.stage}>
        <PatchifyStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <AdaLNStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <ScalingStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>DiT vs U-Net SOTA</h2>
          <SotaCompareBars />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.vsUnet} sourcePath={DIT_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>DiT 家族 — 2022.12 → 2024.8</h2>
          <DiTFamilyTimeline />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.family} sourcePath={DIT_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={DIT_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={DIT_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={DIT_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
