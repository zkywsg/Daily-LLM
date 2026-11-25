import { Link } from "react-router";

import flowMatchingMarkdown from "../../../../../../10-diffusion/04-flow-matching.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, FLOW_MATCHING_SOURCE_PATH } from "./lib/prose";
import { StraightPathStage } from "./stages/StraightPathStage";
import { VelocityFieldStage } from "./stages/VelocityFieldStage";
import { SynergyStage } from "./stages/SynergyStage";
import { CfgScaleCompareChart } from "./widgets/CfgScaleCompareChart";
import styles from "./NodePageFlowMatching.module.css";

const prose = extractProse(flowMatchingMarkdown);

export default function NodePageFlowMatching() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/10-diffusion" className={styles.back}>
          ← 返回 Diffusion
        </Link>
        <h1 className={styles.title}>Flow Matching (2023)</h1>
        <div className={styles.metaLine}>
          作者:Yaron Lipman · Ricky T. Q. Chen · Heli Ben-Hamu · Maximilian Nickel · Xingchao Liu · Chengyue Gong · Qiang Liu
        </div>
        <div className={styles.metaLine}>
          论文:Flow Matching for Generative Modeling / Rectified Flow
        </div>
        <p className={styles.keyIdea}>
          把 diffusion 的 ε-prediction 推广到任意流形的"速度场学习",训练更稳 +
          采样路径更直 + 数学更简洁,SD3 / Flux 默认
        </p>
      </section>

      <section className={styles.stage}>
        <StraightPathStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <VelocityFieldStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <SynergyStage synergyProse={prose.synergy} />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>Stable Diffusion 3 的采用</h2>
          <CfgScaleCompareChart />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.sd3Adoption} sourcePath={FLOW_MATCHING_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>为什么"直线"比"曲线"好</h2>
          <MarkdownRenderer markdown={prose.whyStraight} sourcePath={FLOW_MATCHING_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={FLOW_MATCHING_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={FLOW_MATCHING_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={FLOW_MATCHING_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
