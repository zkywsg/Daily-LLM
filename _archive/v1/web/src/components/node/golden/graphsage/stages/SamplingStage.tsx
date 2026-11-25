import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GRAPHSAGE_SOURCE_PATH } from "../lib/prose";
import { SampleControlWidget } from "../widgets/SampleControlWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function SamplingStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:固定大小邻域采样
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        度数很高的节点(比如社交网络里的大 V)如果每次都聚合全部邻居,计算量会随度数线性增长且不可预测。GraphSAGE 每次只随机采样固定数量 k 个邻居,把每层计算量的上界锁死。
      </p>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={GRAPHSAGE_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={GRAPHSAGE_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <SampleControlWidget />
          <p className={styles.caption}>↑ 选节点、拖 k、点"重新采样"看不同采样子集</p>
        </div>
      </div>
    </div>
  );
}
