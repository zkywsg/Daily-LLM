import { Link } from "react-router";

import llavaMarkdown from "../../../../../../09-multimodal-clip/04-llava.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, LLAVA_SOURCE_PATH } from "./lib/prose";
import { ClipEncoderStage } from "./stages/ClipEncoderStage";
import { ProjectionStage } from "./stages/ProjectionStage";
import { InstructionTuningStage } from "./stages/InstructionTuningStage";
import { LlavaBenchChart } from "./widgets/LlavaBenchChart";
import styles from "./NodePageLLaVA.module.css";

const prose = extractProse(llavaMarkdown);

export default function NodePageLLaVA() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/09-multimodal-clip" className={styles.back}>
          ← 返回 Multimodal / CLIP
        </Link>
        <h1 className={styles.title}>LLaVA (2023)</h1>
        <div className={styles.metaLine}>
          作者:Haotian Liu · Chunyuan Li · Qingyang Wu · Yong Jae Lee(威斯康星麦迪逊 + 微软研究院)
        </div>
        <div className={styles.metaLine}>
          论文:Visual Instruction Tuning
        </div>
        <p className={styles.keyIdea}>
          Visual instruction tuning:用 GPT-4 自动生成视觉指令数据,把
          CLIP 视觉特征用单 linear projection 接到 LLaMA,把开源 VLM
          范式定型在 GPT-4V 之前
        </p>
      </section>

      <section className={styles.stage}>
        <ClipEncoderStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <ProjectionStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <InstructionTuningStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <LlavaBenchChart />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={LLAVA_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>LLaVA-1.5 的改进</h2>
          <MarkdownRenderer markdown={prose.improvements} sourcePath={LLAVA_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={LLAVA_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={LLAVA_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={LLAVA_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
