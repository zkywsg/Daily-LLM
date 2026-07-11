import { Link } from "react-router";

import adapterMarkdown from "../../../../../../11-peft-lora/01-adapter.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, ADAPTER_SOURCE_PATH } from "./lib/prose";
import { BottleneckStructureStage } from "./stages/BottleneckStructureStage";
import { ZeroInitResidualStage } from "./stages/ZeroInitResidualStage";
import { FrozenBaseStage } from "./stages/FrozenBaseStage";
import { StorageCompareChart } from "./widgets/StorageCompareChart";
import styles from "./NodePageAdapter.module.css";

const prose = extractProse(adapterMarkdown);

export default function NodePageAdapter() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/11-peft-lora" className={styles.back}>
          ← 返回 PEFT / LoRA
        </Link>
        <h1 className={styles.title}>Adapter Tuning (2019)</h1>
        <div className={styles.metaLine}>
          作者:Neil Houlsby · Andrei Giurgiu · Stanislaw Jastrzebski · Bruna Morrone ·
          Quentin de Laroussilhe · Andrea Gesmundo · Mona Attariyan · Sylvain Gelly(Google)
        </div>
        <div className={styles.metaLine}>
          论文:Parameter-Efficient Transfer Learning for NLP
        </div>
        <p className={styles.keyIdea}>
          在每层 Transformer 插入 small bottleneck adapter 模块(down → ReLU →
          up + residual),base 模型完全冻结,只训 3% 参数达到全参微调 96%
          性能;PEFT 起源,后续 LoRA / Prefix Tuning 都受其启发
        </p>
      </section>

      <section className={styles.stage}>
        <BottleneckStructureStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <ZeroInitResidualStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <FrozenBaseStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        <div className={styles.footerSection}>
          <h2>性能数据</h2>
          <StorageCompareChart />
          <div style={{ marginTop: "var(--space-4)" }}>
            <MarkdownRenderer markdown={prose.performance} sourcePath={ADAPTER_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={ADAPTER_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={ADAPTER_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
