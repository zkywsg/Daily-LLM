import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GAT_SOURCE_PATH } from "../lib/prose";
import { CenterNodeSelectorWidget } from "../widgets/CenterNodeSelectorWidget";
import { CombineModeToggleWidget } from "../widgets/CombineModeToggleWidget";
import { MultiHeadWidget } from "../widgets/MultiHeadWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function MultiHeadStage({ mechanism3Prose, synergyProse }: Props) {
  const [center, setCenter] = useState(1);
  const [mode, setMode] = useState<"concat" | "average">("concat");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:多头注意力
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        每个头独立学一套 attention 权重,关注邻居的不同侧面。中间层用 concat 拼接保留所有头的信息、扩大表示维度;输出层用 average 平均,把多头意见汇总成一个稳定的最终预测。
      </p>

      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={GAT_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={GAT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <CenterNodeSelectorWidget selected={center} onSelect={setCenter} />
          <CombineModeToggleWidget mode={mode} onChange={setMode} />
          <MultiHeadWidget center={center} mode={mode} />
        </div>
      </div>
    </div>
  );
}
