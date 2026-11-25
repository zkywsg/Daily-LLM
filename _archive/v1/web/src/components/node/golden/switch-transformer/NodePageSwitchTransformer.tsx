import { Link } from "react-router";

import switchTransformerMarkdown from "../../../../../../13-moe-efficient/02-switch-transformer.md?raw";
import { MarkdownRenderer } from "../../MarkdownRenderer";
import { extractProse, SWITCH_TRANSFORMER_SOURCE_PATH } from "./lib/prose";
import { Top1RoutingStage } from "./stages/Top1RoutingStage";
import { LoadBalancingStage } from "./stages/LoadBalancingStage";
import { StabilityStage } from "./stages/StabilityStage";
import { ModelScaleChart } from "./widgets/ModelScaleChart";
import styles from "./NodePageSwitchTransformer.module.css";

const prose = extractProse(switchTransformerMarkdown);

export default function NodePageSwitchTransformer() {
  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <Link to="/families/13-moe-efficient" className={styles.back}>
          ← 返回 MoE / Efficient
        </Link>
        <h1 className={styles.title}>Switch Transformer (2021)</h1>
        <div className={styles.metaLine}>
          作者:William Fedus · Barret Zoph · Noam Shazeer
        </div>
        <div className={styles.metaLine}>
          论文:Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
        </div>
        <p className={styles.keyIdea}>
          把 MoE 移植到 Transformer + 简化为 top-1 gating(每 token 只走一个
          expert,代替 Shazeer top-K),配 load balancing loss 和 selective
          precision;首次做到 1.6T 参数模型,T5-XXL 4× 加速同质量
        </p>
      </section>

      <section className={styles.stage}>
        <Top1RoutingStage
          intuitionProse={prose.intuition}
          mechanism1Prose={prose.mechanism1}
        />
      </section>

      <section className={`${styles.stage} ${styles.stageAlt}`}>
        <LoadBalancingStage mechanism2Prose={prose.mechanism2} />
      </section>

      <section className={styles.stage}>
        <StabilityStage
          mechanism3Prose={prose.mechanism3}
          synergyProse={prose.synergy}
        />
      </section>

      <section className={styles.footer}>
        {prose.performance && (
          <div className={styles.footerSection}>
            <h2>性能数据</h2>
            <ModelScaleChart />
            <MarkdownRenderer markdown={prose.performance} sourcePath={SWITCH_TRANSFORMER_SOURCE_PATH} />
          </div>
        )}
        <div className={styles.footerSection}>
          <h2>关键代码</h2>
          <MarkdownRenderer markdown={prose.keyCode} sourcePath={SWITCH_TRANSFORMER_SOURCE_PATH} />
        </div>
        <div className={styles.footerSection}>
          <h2>影响 / 后续</h2>
          <MarkdownRenderer markdown={prose.aftermath} sourcePath={SWITCH_TRANSFORMER_SOURCE_PATH} />
        </div>
      </section>
    </div>
  );
}
