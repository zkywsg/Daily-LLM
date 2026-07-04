import { Link } from "react-router";

import flashAttnMarkdown from "../../../../../../05-transformer/05-flash-attention.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, FLASH_ATTN_SOURCE_PATH } from "./lib/prose";
import { TilingStage } from "./stages/TilingStage";
import { OnlineSoftmaxStage } from "./stages/OnlineSoftmaxStage";
import { RecomputeStage } from "./stages/RecomputeStage";
import { SpeedupBars } from "./widgets/SpeedupBars";
import { KvCacheCompareChart } from "./widgets/KvCacheCompareChart";
import styles from "./NodePageFlashAttention.module.css";

const prose = extractProse(flashAttnMarkdown);

export default function NodePageFlashAttention() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/05-transformer" className={styles.back}>
          ← 返回 Transformer
        </Link>
        <h1 className={styles.title}>FlashAttention (2022)</h1>
        <div className={styles.metaLine}>
          作者:Tri Dao · Daniel Y. Fu · Stefano Ermon · Atri Rudra · Christopher Ré(Stanford)
        </div>
        <div className={styles.metaLine}>
          论文:FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
        </div>
        <p className={styles.keyIdea}>
          把 attention 从 HBM 搬到 SRAM 算,分块 + 重计算把 O(N²) 显存压成 O(N)
          而结果完全等价 — 不是近似算法,是 attention 该有的样子,
          两年内成为 PyTorch / HuggingFace / vLLM 的默认 attention backend
        </p>
      </section>

      <section className={styles.stage}>
        <TilingStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <OnlineSoftmaxStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <RecomputeStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <SpeedupBars />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={FLASH_ATTN_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>三个版本的演化</h2>
          <MarkdownRenderer markdown={prose.versions} sourcePath={FLASH_ATTN_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>GQA / MQA:推理时代的多头演化</h2>
          <KvCacheCompareChart />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.gqa} sourcePath={FLASH_ATTN_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={FLASH_ATTN_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={FLASH_ATTN_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
