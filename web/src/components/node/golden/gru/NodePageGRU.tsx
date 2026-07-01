import { Link } from "react-router";

import gruMarkdown from "../../../../../../02-rnn-lstm/03-gru.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, GRU_SOURCE_PATH } from "./lib/prose";
import { ResetGateStage } from "./stages/ResetGateStage";
import { UpdateGateStage } from "./stages/UpdateGateStage";
import { SingleStateStage } from "./stages/SingleStateStage";
import { ParamSpeedBars } from "./widgets/ParamSpeedBars";
import styles from "./NodePageGRU.module.css";

const prose = extractProse(gruMarkdown);

export default function NodePageGRU() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/02-rnn-lstm" className={styles.back}>
          ← 返回 RNN / LSTM / GRU 循环网络
        </Link>
        <h1 className={styles.title}>GRU (2014)</h1>
        <div className={styles.metaLine}>
          作者:Kyunghyun Cho · Bart van Merriënboer · Dzmitry Bahdanau · Yoshua Bengio
        </div>
        <div className={styles.metaLine}>
          论文:Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation
        </div>
        <p className={styles.keyIdea}>
          把 LSTM 三道门简成两门、去掉细胞状态,参数减少 25% 而性能基本持平 —
          重置门决定用多少历史,更新门用凸组合替代独立 forget/input,
          单一状态 h 合并长短期记忆,成为 LSTM 的常用轻量替代
        </p>
      </section>

      <section className={styles.stage}>
        <ResetGateStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <UpdateGateStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <SingleStateStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能和实际选择</h2>
          <ParamSpeedBars />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.practice} sourcePath={GRU_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={GRU_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={GRU_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
