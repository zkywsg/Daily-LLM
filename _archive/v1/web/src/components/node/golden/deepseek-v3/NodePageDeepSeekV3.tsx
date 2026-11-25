import { Link } from "react-router";

import deepseekV3Markdown from "../../../../../../13-moe-efficient/04-deepseek-v3.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, DEEPSEEK_V3_SOURCE_PATH } from "./lib/prose";
import { FineGrainedExpertsStage } from "./stages/FineGrainedExpertsStage";
import { AuxLossFreeStage } from "./stages/AuxLossFreeStage";
import { TrainingSystemStage } from "./stages/TrainingSystemStage";
import { MathBenchmarkChart } from "./widgets/MathBenchmarkChart";
import styles from "./NodePageDeepSeekV3.module.css";

const prose = extractProse(deepseekV3Markdown);

export default function NodePageDeepSeekV3() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/13-moe-efficient" className={styles.back}>
          ← 返回 MoE / Efficient
        </Link>
        <h1 className={styles.title}>DeepSeek-V3 (2024)</h1>
        <div className={styles.metaLine}>作者:DeepSeek-AI</div>
        <div className={styles.metaLine}>论文:DeepSeek-V3 Technical Report</div>
        <p className={styles.keyIdea}>
          671B 总参 / 37B 激活的开源 MoE 旗舰,集成 fine-grained experts(256 细粒度 expert)+
          shared experts + aux-loss-free load balancing + MTP(Multi-Token Prediction)等十余项创新;
          首次让开源 MoE 追上 GPT-4 级闭源模型,也是 DeepSeek-R1 的 base
        </p>
      </section>

      <section className={styles.stage}>
        <FineGrainedExpertsStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <AuxLossFreeStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <TrainingSystemStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <MathBenchmarkChart />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={DEEPSEEK_V3_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={DEEPSEEK_V3_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={DEEPSEEK_V3_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
