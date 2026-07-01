import { Link } from "react-router";

import seq2seqMarkdown from "../../../../../../02-rnn-lstm/04-seq2seq.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, SEQ2SEQ_SOURCE_PATH } from "./lib/prose";
import { EncoderStage } from "./stages/EncoderStage";
import { DecoderStage } from "./stages/DecoderStage";
import { EngineeringTricksStage } from "./stages/EngineeringTricksStage";
import styles from "./NodePageSeq2Seq.module.css";

const prose = extractProse(seq2seqMarkdown);

export default function NodePageSeq2Seq() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/02-rnn-lstm" className={styles.back}>
          ← 返回 RNN / LSTM / GRU 循环网络
        </Link>
        <h1 className={styles.title}>Seq2Seq (2014)</h1>
        <div className={styles.metaLine}>
          作者:Ilya Sutskever · Oriol Vinyals · Quoc V. Le · Kyunghyun Cho · Yoshua Bengio
        </div>
        <div className={styles.metaLine}>
          论文:Sequence to Sequence Learning with Neural Networks / Learning Phrase Representations using RNN Encoder–Decoder
        </div>
        <p className={styles.keyIdea}>
          用一个 encoder RNN 把任意长输入压成上下文向量,decoder RNN 从向量生成任意长输出 —
          第一次让端到端神经网络翻译超过统计 SMT,统一了翻译/对话/摘要/语音识别,
          直接催生 Bahdanau attention 与 Transformer 的 encoder-decoder 骨架
        </p>
      </section>

      <section className={styles.stage}>
        <EncoderStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <DecoderStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <EngineeringTricksStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>信息瓶颈</h2>
          <MarkdownRenderer markdown={prose.bottleneck} sourcePath={SEQ2SEQ_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={SEQ2SEQ_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={SEQ2SEQ_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={SEQ2SEQ_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
