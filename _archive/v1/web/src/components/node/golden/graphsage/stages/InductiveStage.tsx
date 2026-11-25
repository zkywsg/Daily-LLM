import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GRAPHSAGE_SOURCE_PATH } from "../lib/prose";
import { InductiveCompareWidget } from "../widgets/InductiveCompareWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function InductiveStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:逐层采样-聚合-拼接(归纳式泛化)
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        K 层堆叠,每层都是"采样邻居 → 聚合 → 和自身特征拼接 → 非线性变换"。因为聚合函数不绑定具体节点 id,整套流程可以直接搬到训练时从未见过的新节点/新图上。
      </p>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={GRAPHSAGE_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={GRAPHSAGE_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <InductiveCompareWidget />
        </div>
      </div>
    </div>
  );
}
