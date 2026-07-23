import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GIN_SOURCE_PATH } from "../lib/prose";
import { EpsilonSliderWidget } from "../widgets/EpsilonSliderWidget";
import { SelfVsNeighborBarWidget } from "../widgets/SelfVsNeighborBarWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function EpsilonStage({ mechanism2Prose }: Props) {
  const [epsilon, setEpsilon] = useState(0);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:GIN 的求和聚合 + MLP
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        (1+ε) 控制节点在更新时对"自己旧特征"的加权。ε=0 时自身和普通邻居等权;ε 越大,节点越"固执",更新时更依赖自己而不是邻居传来的信息。
      </p>

      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={GIN_SOURCE_PATH} />
        </div>

        <div className={styles.stickyPanel}>
          <EpsilonSliderWidget epsilon={epsilon} onChange={setEpsilon} />
          <SelfVsNeighborBarWidget epsilon={epsilon} />
        </div>
      </div>
    </div>
  );
}
