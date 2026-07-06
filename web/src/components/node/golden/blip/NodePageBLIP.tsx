import { Link } from "react-router";

import blipMarkdown from "../../../../../../09-multimodal-clip/02-blip.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, BLIP_SOURCE_PATH } from "./lib/prose";
import { ThreeTaskStage } from "./stages/ThreeTaskStage";
import { CapFiltStage } from "./stages/CapFiltStage";
import { QFormerStage } from "./stages/QFormerStage";
import { BenchmarkCompareBars } from "./widgets/BenchmarkCompareBars";
import styles from "./NodePageBLIP.module.css";

const prose = extractProse(blipMarkdown);

export default function NodePageBLIP() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/09-multimodal-clip" className={styles.back}>
          ← 返回 Multimodal / CLIP
        </Link>
        <h1 className={styles.title}>BLIP / BLIP-2 (2022)</h1>
        <div className={styles.metaLine}>
          作者:Junnan Li · Dongxu Li · Caiming Xiong · Steven Hoi · Silvio Savarese(Salesforce)
        </div>
        <div className={styles.metaLine}>
          论文:BLIP: Bootstrapping Language-Image Pre-training / BLIP-2: Q-Former for Vision-Language Pre-training
        </div>
        <p className={styles.keyIdea}>
          在 CLIP 对比之上加入生成和匹配两个任务联合训练;BLIP-2 进一步用
          Q-Former 桥接冻结视觉编码器和冻结 LLM,把训练成本降一个数量级 —
          定义了"冻结大模型 + 轻量桥接"的开源 VLM 范式
        </p>
      </section>

      <section className={styles.stage}>
        <ThreeTaskStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <CapFiltStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <QFormerStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <BenchmarkCompareBars />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={BLIP_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={BLIP_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={BLIP_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={BLIP_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
