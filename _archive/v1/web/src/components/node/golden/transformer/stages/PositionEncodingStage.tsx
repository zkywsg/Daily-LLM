import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { TRANSFORMER_SOURCE_PATH } from "../lib/prose";
import { PositionWavesSVG } from "../widgets/PositionWavesSVG";
import { PositionSimilarityMatrix } from "../widgets/PositionSimilarityMatrix";
import { PEControls } from "../widgets/PEControls";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function PositionEncodingStage({
  mechanism3Prose,
  synergyProse,
}: Props) {
  const [nPos, setNPos] = useState(24);
  const [dModel, setDModel] = useState(32);
  const [highlightDim, setHighlightDim] = useState<number | null>(null);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Position Encoding 补回顺序
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        attention 对 token 顺序完全不变(置换不影响输出)。需要显式给每个
        位置一个独有"指纹"加到 token embedding 上。原版用 sin/cos
        多频率叠加 —— 不需要训练就能编码任意长度。
      </p>

      <PositionWavesSVG
        nPos={nPos}
        dModel={dModel}
        highlightDim={highlightDim ?? undefined}
      />
      <p className={styles.caption}>
        ↑ 每条曲线是 PE 矩阵的一个维度沿 position 的取值。低维短波长,高维长波长 ——
        组合起来每个位置都拿到独一无二的 d_model 维指纹。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3
            style={{
              fontSize: "var(--fs-xl)",
              marginBottom: "var(--space-4)",
            }}
          >
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer
              markdown={mechanism3Prose}
              sourcePath={TRANSFORMER_SOURCE_PATH}
            />
          </div>

          <h3
            style={{
              fontSize: "var(--fs-xl)",
              marginTop: "var(--space-8)",
              marginBottom: "var(--space-4)",
            }}
          >
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer
              markdown={synergyProse}
              sourcePath={TRANSFORMER_SOURCE_PATH}
            />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <PositionSimilarityMatrix nPos={nPos} dModel={dModel} />
          <p className={styles.caption}>
            cos_sim(PE_i, PE_j) 自相似矩阵。对角线必然 = 1(自己跟自己);
            相邻 pos 接近 1,距离越远越往 0 走 —— PE 在用相似度编码"距离"。
          </p>
          <div style={{ marginTop: "var(--space-4)" }}>
            <PEControls
              nPos={nPos}
              onNPosChange={setNPos}
              dModel={dModel}
              onDModelChange={setDModel}
              highlightDim={highlightDim}
              onHighlightDimChange={setHighlightDim}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
