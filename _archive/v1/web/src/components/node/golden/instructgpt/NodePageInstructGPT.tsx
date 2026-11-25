import { Link } from "react-router";

import instructgptMarkdown from "../../../../../../12-rlhf-alignment/02-instructgpt.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, INSTRUCTGPT_SOURCE_PATH } from "./lib/prose";
import { SFTStage } from "./stages/SFTStage";
import { RewardModelStage } from "./stages/RewardModelStage";
import { PPOStage } from "./stages/PPOStage";
import styles from "./NodePageInstructGPT.module.css";

const prose = extractProse(instructgptMarkdown);

export default function NodePageInstructGPT() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/12-rlhf-alignment" className={styles.back}>
          ← 返回 对齐与 RLHF
        </Link>
        <h1 className={styles.title}>InstructGPT (2022)</h1>
        <div className={styles.metaLine}>
          作者:Long Ouyang · Jeff Wu · Xu Jiang · ... · Ryan Lowe · OpenAI
        </div>
        <div className={styles.metaLine}>
          论文:Training language models to follow instructions with human feedback
        </div>
        <p className={styles.keyIdea}>
          SFT 热启动 → RM 学人类偏好 → PPO 加 KL 防漂移 三阶段对齐,
          1.3B 对齐胜过 175B 原版 — ChatGPT 的直接前身
        </p>
      </section>

      <section className={styles.stage}>
        <SFTStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <RewardModelStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <PPOStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
          alignmentTaxProse={prose.alignmentTax}
        />
      </section>

      <section className={styles.footer}>
        {prose.promptDiversity && (
          <div className={styles.footerSection}>
            <h2>Prompt 多样性的关键</h2>
            <MarkdownRenderer markdown={prose.promptDiversity} sourcePath={INSTRUCTGPT_SOURCE_PATH} />
          </div>
        )}
        {prose.alignmentBeatsScale && (
          <div className={styles.footerSection}>
            <h2>"对齐胜过规模"的实证</h2>
            <MarkdownRenderer markdown={prose.alignmentBeatsScale} sourcePath={INSTRUCTGPT_SOURCE_PATH} />
          </div>
        )}
        {prose.chatgptDeployment && (
          <div className={styles.footerSection}>
            <h2>ChatGPT 是 InstructGPT 的部署版</h2>
            <MarkdownRenderer markdown={prose.chatgptDeployment} sourcePath={INSTRUCTGPT_SOURCE_PATH} />
          </div>
        )}
        {prose.dataScale && (
          <div className={styles.footerSection}>
            <h2>数据规模</h2>
            <MarkdownRenderer markdown={prose.dataScale} sourcePath={INSTRUCTGPT_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={INSTRUCTGPT_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={INSTRUCTGPT_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={INSTRUCTGPT_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
