import { Link } from "react-router";

import rnnMarkdown from "../../../../../../02-rnn-lstm/01-rnn.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, RNN_SOURCE_PATH } from "./lib/prose";
import { HiddenStateStage } from "./stages/HiddenStateStage";
import { WeightSharingStage } from "./stages/WeightSharingStage";
import { BpttStage } from "./stages/BpttStage";
import styles from "./NodePageRNN.module.css";

const prose = extractProse(rnnMarkdown);

export default function NodePageRNN() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/02-rnn-lstm" className={styles.back}>
          ← 返回 RNN / LSTM / GRU 循环网络
        </Link>
        <h1 className={styles.title}>RNN (1986)</h1>
        <div className={styles.metaLine}>
          作者:David Rumelhart · Geoffrey Hinton · Ronald Williams · Jeffrey Elman
        </div>
        <div className={styles.metaLine}>
          论文:Learning Internal Representations by Error Propagation / Finding Structure in Time
        </div>
        <p className={styles.keyIdea}>
          把上一时刻的隐状态接回当前时刻输入,用一组共享权重在时间上递推,
          任意长度序列被压进一个固定维度向量 — 序列建模的奠基性范式,
          也是 LSTM / GRU / Seq2Seq / Transformer 的共同祖先
        </p>
      </section>

      <section className={styles.stage}>
        <HiddenStateStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <WeightSharingStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <BpttStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>两个早期任务</h2>
          <MarkdownRenderer markdown={prose.earlyTasks} sourcePath={RNN_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={RNN_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={RNN_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={RNN_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
