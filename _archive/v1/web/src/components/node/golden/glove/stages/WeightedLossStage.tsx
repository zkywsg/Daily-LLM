import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GLOVE_SOURCE_PATH } from "../lib/prose";
import { WeightFunctionChart } from "../widgets/WeightFunctionChart";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function WeightedLossStage({ mechanism2Prose }: Props) {
  const [xMax, setXMax] = useState(100);
  const [alpha, setAlpha] = useState(0.75);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:加权 loss — 平衡 rare 和 common pair
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        直接 fit log X_ij 有两个问题:rare pair(X=0)是 log(0) 不可算;common pair
        (像 "the/and")共现频次极高会主导 loss。GloVe 加权函数 f(x) = (x/x_max)^α 当
        x&lt;x_max,否则封顶 1.0。rare pair 权重≈0(不学也不亏),common pair 封顶
        不再无限主导,算力集中在携带最多语义信息的中频 pair。
      </p>

      <WeightFunctionChart xMax={xMax} alpha={alpha} />
      <p className={styles.caption}>
        ↑ 拖动 x_max 和 α 看权重曲线形状变化。论文用 x_max=100,α=0.75
        (和 Word2Vec NEG 的 0.75 巧合相同)。
      </p>
      <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginTop: 8 }}>
        <span>x_max</span><strong>{xMax}</strong>
      </label>
      <input type="range" min={20} max={200} step={10} value={xMax}
             onChange={(e) => setXMax(parseInt(e.target.value))} style={{ width: "100%" }} />
      <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginTop: 8 }}>
        <span>α</span><strong>{alpha.toFixed(2)}</strong>
      </label>
      <input type="range" min={0.1} max={1.5} step={0.05} value={alpha}
             onChange={(e) => setAlpha(parseFloat(e.target.value))} style={{ width: "100%" }} />

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={GLOVE_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              完整 loss
            </div>
            <pre style={{ fontSize: 11, lineHeight: 1.6, margin: 0, color: "var(--ink-primary)" }}>{`L = Σ_ij f(X_ij) ·
    (v_i^T v_j + b_i + b_j - log X_ij)²

f(x) = (x/x_max)^α  if x < x_max
     = 1            otherwise`}</pre>
          </div>
        </div>
      </div>
    </div>
  );
}
