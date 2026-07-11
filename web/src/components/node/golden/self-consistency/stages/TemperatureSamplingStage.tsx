import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SELF_CONSISTENCY_SOURCE_PATH } from "../lib/prose";
import { DiverseSamplingDiagram } from "../widgets/DiverseSamplingDiagram";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function TemperatureSamplingStage({ intuitionProse, mechanism1Prose }: Props) {
  const [numSamples, setNumSamples] = useState(5);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Temperature 采样 — 路径多样性的源头
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        CoT 单次贪婪解码(T=0)遇到模糊路径可能一步走错,后面全错。Self-Consistency
        先用 T=0.7 的温度采样跑出 N 条不同的推理路径 — 路径必须真的多样,
        投票才有意义。拖动下面的滑块看 N 从 1 增加到 5 时,采样出的路径如何逐渐
        把错误路径(168)淹没在正确路径(196)之下。
      </p>

      <DiverseSamplingDiagram numSamples={numSamples} />
      <p className={styles.caption}>
        ↑ 农场羊奶问题(7 只羊 × 4 升/天 × 7 天 = 196 升)。拖动滑块增加采样路径数 N。
      </p>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 8 }}>
        <input
          type="range"
          min={1}
          max={5}
          step={1}
          value={numSamples}
          onChange={(e) => setNumSamples(Number(e.target.value))}
          style={{ flex: 1, maxWidth: 240 }}
        />
        <span style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)" }}>N = {numSamples}</span>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={SELF_CONSISTENCY_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={SELF_CONSISTENCY_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              关键超参
            </div>
            <div style={{ fontSize: "var(--fs-sm)", lineHeight: 1.8, color: "var(--ink-primary)" }}>
              <div>Temperature = 0.7(对比 CoT 默认 greedy T=0)</div>
              <div>Sample count N = 1 ~ 40(论文测试范围)</div>
              <div>Top-p = 0.95(防止极端 token)</div>
            </div>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              温度太低 → 路径过于相似,投票无意义;太高 → 路径质量下降,正确路径不够多。
            </div>
            <div style={{ display: "flex", gap: 6, marginTop: 10 }}>
              {[1, 3, 5].map((n) => (
                <button key={n} type="button" onClick={() => setNumSamples(n)} style={btnStyle(numSamples === n)}>
                  N={n}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
