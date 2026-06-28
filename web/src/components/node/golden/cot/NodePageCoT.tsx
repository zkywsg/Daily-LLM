import { Link } from "react-router";

import cotMarkdown from "../../../../../../15-reasoning-o1-r1/01-cot.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, COT_SOURCE_PATH } from "./lib/prose";
import { FewShotCoTStage } from "./stages/FewShotCoTStage";
import { ZeroShotCoTStage } from "./stages/ZeroShotCoTStage";
import { ScaleEmergenceStage } from "./stages/ScaleEmergenceStage";
import styles from "./NodePageCoT.module.css";

const prose = extractProse(cotMarkdown);

export default function NodePageCoT() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/15-reasoning-o1-r1" className={styles.back}>
          ← 返回 推理模型 (Test-time Compute)
        </Link>
        <h1 className={styles.title}>Chain-of-Thought (2022)</h1>
        <div className={styles.metaLine}>
          作者:Jason Wei · Xuezhi Wang · Dale Schuurmans · ... · Quoc Le · Denny Zhou · Google Research
        </div>
        <div className={styles.metaLine}>
          论文:Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
        </div>
        <p className={styles.keyIdea}>
          让模型把推理过程显式写下来,数学/逻辑/常识题准确率飞跃 —
          只在 ~62B+ 大模型上有效,是 test-time compute 时代的开端
        </p>
      </section>

      <section className={styles.stage}>
        <FewShotCoTStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <ZeroShotCoTStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <ScaleEmergenceStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={COT_SOURCE_PATH} />
          </div>
        )}
        {prose.extensions && (
          <div className={styles.footerSection}>
            <h2>CoT 的扩展</h2>
            <MarkdownRenderer markdown={prose.extensions} sourcePath={COT_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={COT_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={COT_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={COT_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
