import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GAT_SOURCE_PATH } from "../lib/prose";
import { CenterNodeSelectorWidget } from "../widgets/CenterNodeSelectorWidget";
import { MaskSoftmaxPipelineWidget } from "../widgets/MaskSoftmaxPipelineWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function MaskSoftmaxStage({ mechanism2Prose }: Props) {
  const [center, setCenter] = useState(3);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:softmax 归一化 + 加权聚合
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        GAT 不需要提前知道完整图结构去做矩阵运算 —— 但计算 attention 时仍然只在"真实存在的边"上做 softmax,非邻居的 logit 会被 mask 掉,不参与归一化。
      </p>

      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={GAT_SOURCE_PATH} />
        </div>

        <div className={styles.stickyPanel}>
          <CenterNodeSelectorWidget selected={center} onSelect={setCenter} />
          <MaskSoftmaxPipelineWidget center={center} />
        </div>
      </div>
    </div>
  );
}
