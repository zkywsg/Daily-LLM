import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { WORD2VEC_SOURCE_PATH } from "../lib/prose";
import { SoftmaxVsNegFormula } from "../widgets/SoftmaxVsNegFormula";
import { NegSamplingDistribution } from "../widgets/NegSamplingDistribution";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function NegSamplingStage({ mechanism2Prose }: Props) {
  const [alpha, setAlpha] = useState(0.75);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Negative Sampling — 把 O(|V|) 压成 O(K)
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Skip-gram 的 softmax 分母遍历 5 万词,每步 O(V) — 60 亿词训练单机几个月。
        Mikolov 的核心 trick:不算真分母,改判别 "真上下文 vs K 个噪声词",
        把每步从 5 万次 dot 压到 6 次 sigmoid,效率 100×+。
      </p>

      <SoftmaxVsNegFormula />
      <p className={styles.caption}>
        ↑ 上半粉条铺满 |V|=50000;下半绿条 K+1=6 几乎不可见。等量训练样本下,
        NEG 的每步成本是 softmax 的 0.012%。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={WORD2VEC_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <NegSamplingDistribution alpha={alpha} />
          <p className={styles.caption}>
            ↑ 负采样从 P_n(w) ∝ count(w)^α 里采。α=1 偏向高频(stopword 被采爆);
            α=0 退化 uniform(罕见词被过度采);α=0.75 是 Mikolov 实测最佳平衡。
          </p>
          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: 4 }}>
              <span>α (指数)</span><strong>{alpha.toFixed(2)}</strong>
            </label>
            <input type="range" min={0} max={1} step={0.05} value={alpha} onChange={(e) => setAlpha(parseFloat(e.target.value))} style={{ width: "100%" }} />
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 6, lineHeight: 1.4 }}>
              0 = uniform / 0.75 = Mikolov / 1.0 = unigram。
              试试 1.0 — 几乎只采 the/of,所有 embedding 都被 push 远离这些词,语义垮掉。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
