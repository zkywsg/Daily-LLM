import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GCN_SOURCE_PATH } from "../lib/prose";
import { DegreeSelectorWidget } from "../widgets/DegreeSelectorWidget";
import { LayerToggleWidget } from "../widgets/LayerToggleWidget";
import { ReceptiveFieldWidget } from "../widgets/ReceptiveFieldWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function PropagationStage({ mechanism3Prose, synergyProse }: Props) {
  const [center, setCenter] = useState(3);
  const [hops, setHops] = useState(1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:逐层传播规则
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        单层 GCN 只能看到 1-hop 邻居。堆叠 L 层后,每个节点的特征里累积了 L-hop 内所有节点的信息 —— 感受野随层数线性扩大。
      </p>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={GCN_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={GCN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <DegreeSelectorWidget selected={center} onSelect={setCenter} />
          <LayerToggleWidget hops={hops} onChange={setHops} />
          <ReceptiveFieldWidget center={center} hops={hops} />
          <p className={styles.caption}>↑ 切换 L=1/2 看感受野从直接邻居扩展到二跳邻居</p>
        </div>
      </div>
    </div>
  );
}
