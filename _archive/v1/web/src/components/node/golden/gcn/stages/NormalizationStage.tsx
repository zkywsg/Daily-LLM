import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GCN_SOURCE_PATH } from "../lib/prose";
import { DegreeSelectorWidget } from "../widgets/DegreeSelectorWidget";
import { NormalizationCompareWidget } from "../widgets/NormalizationCompareWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function NormalizationStage({ mechanism2Prose }: Props) {
  const [center, setCenter] = useState(1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:对称归一化 — D̃^(-1/2) Ã D̃^(-1/2)
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        如果只是把邻居特征简单求和,度数越高的节点会在聚合里贡献越多、数值也越容易爆炸。对称归一化按两端度数的几何平均缩放每条边的权重,压低高度数邻居的影响。
      </p>

      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={GCN_SOURCE_PATH} />
        </div>

        <div className={styles.stickyPanel}>
          <DegreeSelectorWidget selected={center} onSelect={setCenter} />
          <NormalizationCompareWidget center={center} />
          <p className={styles.caption}>
            ↑ 灰色 = 未归一化(每条边权重恒为 1);粉色 = 对称归一化后的实际权重。
            切换中心节点看不同度数组合下权重被压缩的幅度。
          </p>
        </div>
      </div>
    </div>
  );
}
