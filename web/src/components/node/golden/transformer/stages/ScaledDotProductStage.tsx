import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { TRANSFORMER_SOURCE_PATH } from "../lib/prose";
import { AttentionHeatmap } from "../widgets/AttentionHeatmap";
import { AttentionControls } from "../widgets/AttentionControls";
import { QKVPipeline } from "../widgets/QKVPipeline";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function ScaledDotProductStage({
  intuitionProse,
  mechanism1Prose,
}: Props) {
  const [tokens, setTokens] = useState<string[]>([
    "The",
    "cat",
    "sat",
    "on",
    "the",
    "mat",
  ]);
  const [dModel, setDModel] = useState(16);
  const [scaled, setScaled] = useState(true);
  const [view, setView] = useState<"raw" | "scaled" | "weights">("weights");
  const stepHighlight = view === "raw" ? 1 : view === "scaled" ? 2 : 3;

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Scaled Dot-Product Attention
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        每个 token 同时当 query / key / value:用自己的 Q 去和所有
        token 的 K 算相似度,softmax 归一,再加权求和别人的 V。
        循环展开换成一次矩阵乘 —— 这是 Transformer 跑得动 GPU 的关键。
      </p>

      <QKVPipeline scaled={scaled} highlightStep={stepHighlight} />
      <p className={styles.caption}>
        ↑ 流程图三步:(1) Q · Kᵀ 得相似度;(2) ÷√dk 防 dk 大时 softmax 饱和;
        (3) softmax 归一,再加权 V。下面的热力图随你切换"看哪一步"实时换内容。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3
            style={{
              fontSize: "var(--fs-xl)",
              marginBottom: "var(--space-4)",
            }}
          >
            直觉
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer
              markdown={intuitionProse}
              sourcePath={TRANSFORMER_SOURCE_PATH}
            />
          </div>

          <h3
            style={{
              fontSize: "var(--fs-xl)",
              marginTop: "var(--space-6)",
              marginBottom: "var(--space-4)",
            }}
          >
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer
              markdown={mechanism1Prose}
              sourcePath={TRANSFORMER_SOURCE_PATH}
            />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <AttentionHeatmap
            tokens={tokens}
            dModel={dModel}
            scaled={scaled}
            view={view}
          />
          <p className={styles.caption}>
            行 = query token,列 = key token。颜色越深表示该 query 把多少
            注意力分配给了对应的 key。
            {view === "weights" ? " 每行加起来 = 1。" : ""}
          </p>
          <div style={{ marginTop: "var(--space-4)" }}>
            <AttentionControls
              tokens={tokens}
              onTokensChange={setTokens}
              dModel={dModel}
              onDModelChange={setDModel}
              scaled={scaled}
              onScaledChange={setScaled}
              view={view}
              onViewChange={setView}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
