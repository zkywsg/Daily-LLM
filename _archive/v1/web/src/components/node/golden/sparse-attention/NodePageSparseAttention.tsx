import { Link } from "react-router";

import sparseAttnMarkdown from "../../../../../../05-transformer/03-sparse-attention.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, SPARSE_ATTN_SOURCE_PATH } from "./lib/prose";
import { LocalWindowStage } from "./stages/LocalWindowStage";
import { GlobalTokenStage } from "./stages/GlobalTokenStage";
import { RandomKernelStage } from "./stages/RandomKernelStage";
import { LongDocBenchBars } from "./widgets/LongDocBenchBars";
import styles from "./NodePageSparseAttention.module.css";

const prose = extractProse(sparseAttnMarkdown);

export default function NodePageSparseAttention() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/05-transformer" className={styles.back}>
          ← 返回 Transformer
        </Link>
        <h1 className={styles.title}>Sparse Attention (2020)</h1>
        <div className={styles.metaLine}>
          作者:Iz Beltagy · Matthew Peters · Arman Cohan(Longformer)· Manzil Zaheer et al.(BigBird)
        </div>
        <div className={styles.metaLine}>
          论文:Longformer: The Long-Document Transformer / Big Bird: Transformers for Longer Sequences
        </div>
        <p className={styles.keyIdea}>
          用滑窗局部 attention + 少量全局 token 把 attention 复杂度从 O(N²) 降到 O(N),
          让 Transformer 第一次能在 4K-16K 长上下文上跑训练和推理 ·
          "结构化稀疏"思想被 Swin Transformer / Mixtral MoE 等后续工作继承
        </p>
      </section>

      <section className={styles.stage}>
        <LocalWindowStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <GlobalTokenStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <RandomKernelStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>复杂度对比</h2>
          <LongDocBenchBars />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.complexity} sourcePath={SPARSE_ATTN_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={SPARSE_ATTN_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={SPARSE_ATTN_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={SPARSE_ATTN_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
