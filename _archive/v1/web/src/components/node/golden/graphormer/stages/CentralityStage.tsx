import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GRAPHORMER_SOURCE_PATH } from "../lib/prose";
import { CentralityHistogramWidget } from "../widgets/CentralityHistogramWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function CentralityStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:中心性编码
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        标准 Transformer 的全局注意力天然看不见"谁是图里的枢纽节点"。Graphormer 按节点度数查一张 embedding 表,把"这个节点有多重要/多中心"直接加进输入表示里。
      </p>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={GRAPHORMER_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={GRAPHORMER_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <CentralityHistogramWidget />
        </div>
      </div>
    </div>
  );
}
