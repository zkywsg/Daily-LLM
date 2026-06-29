import { Link } from "react-router";

import lstmMarkdown from "../../../../../../02-rnn-lstm/02-lstm.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, LSTM_SOURCE_PATH } from "./lib/prose";
import { CellHighwayStage } from "./stages/CellHighwayStage";
import { ThreeGatesStage } from "./stages/ThreeGatesStage";
import { HiddenCellSeparationStage } from "./stages/HiddenCellSeparationStage";
import styles from "./NodePageLSTM.module.css";

const prose = extractProse(lstmMarkdown);

export default function NodePageLSTM() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/02-rnn-lstm" className={styles.back}>
          ← 返回 RNN / LSTM / GRU 循环网络
        </Link>
        <h1 className={styles.title}>LSTM (1997)</h1>
        <div className={styles.metaLine}>
          作者:Sepp Hochreiter · Jürgen Schmidhuber · TU München
        </div>
        <div className={styles.metaLine}>
          论文:Long Short-Term Memory
        </div>
        <p className={styles.keyIdea}>
          用 cell state 高速路 + 三门控制把 vanishing gradient 难题解掉,
          让 RNN 真的能记住远处 token — 序列模型的奠基性结构
        </p>
      </section>

      <section className={styles.stage}>
        <CellHighwayStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <ThreeGatesStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <HiddenCellSeparationStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={LSTM_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={LSTM_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={LSTM_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
