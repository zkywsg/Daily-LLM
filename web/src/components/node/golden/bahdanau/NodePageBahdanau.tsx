import { Link } from "react-router";

import bahdanauMarkdown from "../../../../../../02-rnn-lstm/05-attention.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, BAHDANAU_SOURCE_PATH } from "./lib/prose";
import { BottleneckStage } from "./stages/BottleneckStage";
import { AlignmentScoreStage } from "./stages/AlignmentScoreStage";
import { DynamicContextStage } from "./stages/DynamicContextStage";
import { ScoreFamilyTable } from "./widgets/ScoreFamilyTable";
import styles from "./NodePageBahdanau.module.css";

const prose = extractProse(bahdanauMarkdown);

export default function NodePageBahdanau() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/02-rnn-lstm" className={styles.back}>
          ← 返回 RNN / LSTM / GRU 循环网络
        </Link>
        <h1 className={styles.title}>Bahdanau Attention (2015)</h1>
        <div className={styles.metaLine}>
          作者:Dzmitry Bahdanau · Kyunghyun Cho · Yoshua Bengio
        </div>
        <div className={styles.metaLine}>
          论文:Neural Machine Translation by Jointly Learning to Align and Translate
        </div>
        <p className={styles.keyIdea}>
          让 decoder 每步直接回头看 encoder 所有时刻,动态加权汇聚成 context vector —
          绕开 Seq2Seq 固定 c 的信息瓶颈,长句 BLEU 回到与短句平行 ·
          比 Transformer 早 3 年的 attention 起点
        </p>
      </section>

      <section className={styles.stage}>
        <BottleneckStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <AlignmentScoreStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <DynamicContextStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>长句质量数据</h2>
          <MarkdownRenderer markdown={prose.lengthData} sourcePath={BAHDANAU_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>Score 函数演化 — Bahdanau → Luong → Transformer</h2>
          <ScoreFamilyTable />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.luong} sourcePath={BAHDANAU_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={BAHDANAU_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={BAHDANAU_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={BAHDANAU_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
