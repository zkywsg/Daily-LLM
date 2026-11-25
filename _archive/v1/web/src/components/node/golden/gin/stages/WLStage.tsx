import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GIN_SOURCE_PATH } from "../lib/prose";
import { WLColoringWidget } from "../widgets/WLColoringWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function WLStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:图级别读出函数
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        GIN 的 sum+MLP 逐层聚合在理论上等价于 Weisfeiler-Lehman 颜色精细化算法:每一轮都把"自己的颜色 + 邻居颜色多重集"映射成新颜色,收敛后能区分的节点数就是 GNN 表达力的上界。
      </p>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={GIN_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={GIN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <WLColoringWidget />
          <p className={styles.caption}>↑ 点"跑一轮 WL 精细化"看颜色如何逐步收敛出不同的等价类</p>
        </div>
      </div>
    </div>
  );
}
