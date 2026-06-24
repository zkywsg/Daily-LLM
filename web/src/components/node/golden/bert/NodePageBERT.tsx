import { Link } from "react-router";

import bertMarkdown from "../../../../../../06-bert-family/01-bert.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, BERT_SOURCE_PATH } from "./lib/prose";
import { BidirectionalAttentionStage } from "./stages/BidirectionalAttentionStage";
import { MLMStage } from "./stages/MLMStage";
import { InputInterfaceStage } from "./stages/InputInterfaceStage";
import styles from "./NodePageBERT.module.css";

const prose = extractProse(bertMarkdown);

export default function NodePageBERT() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/06-bert-family" className={styles.back}>
          ← 返回 预训练语言模型 (BERT 系)
        </Link>
        <h1 className={styles.title}>BERT (2018)</h1>
        <div className={styles.metaLine}>
          作者:Jacob Devlin · Ming-Wei Chang · Kenton Lee · Kristina Toutanova · Google AI
        </div>
        <div className={styles.metaLine}>
          论文:Pre-training of Deep Bidirectional Transformers for Language Understanding
        </div>
        <p className={styles.keyIdea}>
          双向 encoder + MLM 自监督让模型从前后文一起反推被 mask 的词,
          再用 [CLS]/[SEP]/Segment 统一输入接口适配所有 NLU 下游任务
        </p>
      </section>

      <section className={styles.stage}>
        <BidirectionalAttentionStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <MLMStage mechanism2Prose={prose.mechanism2} nspProse={prose.nsp} />
      </section>

      <section className={styles.stage}>
        <InputInterfaceStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        {prose.encoderVsDecoder && (
          <div className={styles.footerSection}>
            <h2>Encoder-only vs Decoder-only</h2>
            <MarkdownRenderer markdown={prose.encoderVsDecoder} sourcePath={BERT_SOURCE_PATH} />
          </div>
        )}
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <MarkdownRenderer markdown={prose.performance} sourcePath={BERT_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={BERT_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={BERT_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={BERT_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
