import { Link } from "react-router";

import transformerXlMarkdown from "../../../../../../05-transformer/02-transformer-xl.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, TRANSFORMER_XL_SOURCE_PATH } from "./lib/prose";
import { RecurrenceStage } from "./stages/RecurrenceStage";
import { RelativePositionStage } from "./stages/RelativePositionStage";
import { EngineeringStage } from "./stages/EngineeringStage";
import { BenchmarkChart } from "./widgets/BenchmarkChart";
import styles from "./NodePageTransformerXL.module.css";

const prose = extractProse(transformerXlMarkdown);

export default function NodePageTransformerXL() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/05-transformer" className={styles.back}>
          ← 返回 Transformer
        </Link>
        <h1 className={styles.title}>Transformer-XL (2019)</h1>
        <div className={styles.metaLine}>
          作者:Zihang Dai · Zhilin Yang · Yiming Yang · Jaime Carbonell · Quoc V. Le · Ruslan Salakhutdinov
        </div>
        <div className={styles.metaLine}>
          论文:Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
        </div>
        <p className={styles.keyIdea}>
          用段级循环把上一段隐状态作为这段的记忆 + 相对位置编码替代绝对 PE,
          让 Transformer 第一次跨越固定窗口处理长上下文
        </p>
      </section>

      <section className={styles.stage}>
        <RecurrenceStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <RelativePositionStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <EngineeringStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <BenchmarkChart />
          <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-2)" }}>
            ↑ WikiText-103 上 perplexity 从 30.0 降到 18.3(降 39%),推理速度比 sliding window 快 1874 倍。
          </p>
          <MarkdownRenderer markdown={prose.perfData} sourcePath={TRANSFORMER_XL_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>训练细节</h2>
          <MarkdownRenderer markdown={prose.trainingDetails} sourcePath={TRANSFORMER_XL_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={TRANSFORMER_XL_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={TRANSFORMER_XL_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
