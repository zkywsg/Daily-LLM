import { Link } from "react-router";

import fasttextMarkdown from "../../../../../../03-word-embedding/03-fasttext.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, FASTTEXT_SOURCE_PATH } from "./lib/prose";
import { NgramStage } from "./stages/NgramStage";
import { VectorSumStage } from "./stages/VectorSumStage";
import { HashingStage } from "./stages/HashingStage";
import { MorphologyCompareBars } from "./widgets/MorphologyCompareBars";
import styles from "./NodePageFastText.module.css";

const prose = extractProse(fasttextMarkdown);

export default function NodePageFastText() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/03-word-embedding" className={styles.back}>
          ← 返回 Word Embedding 词嵌入
        </Link>
        <h1 className={styles.title}>FastText (2016)</h1>
        <div className={styles.metaLine}>
          作者:Piotr Bojanowski · Edouard Grave · Armand Joulin · Tomas Mikolov(Facebook AI)
        </div>
        <div className={styles.metaLine}>
          论文:Enriching Word Vectors with Subword Information / Bag of Tricks for Efficient Text Classification
        </div>
        <p className={styles.keyIdea}>
          把词拆成 character n-gram,词向量 = subword 向量之和 —
          处理 OOV / 形态丰富语言 / 罕见词,直接催生了 BPE / WordPiece / SentencePiece,
          今天 GPT-4 / Claude / LLaMA 的 tokenizer 仍是这一思想的延续
        </p>
      </section>

      <section className={styles.stage}>
        <NgramStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <VectorSumStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <HashingStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <MorphologyCompareBars />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={FASTTEXT_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={FASTTEXT_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={FASTTEXT_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
