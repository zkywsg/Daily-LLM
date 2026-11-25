import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SCALING_SOURCE_PATH } from "../lib/prose";
import { PowerLawCurves } from "../widgets/PowerLawCurves";
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

export function PowerLawStage({ intuitionProse, mechanism1Prose }: Props) {
  const [axis, setAxis] = useState<"N" | "D" | "C" | "all">("all");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:三轴 power law — L(N), L(D), L(C) 都是 log-log 直线
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Kaplan 2020 系统拟合一系列 GPT 风格小模型,发现 LM test loss 随
        参数 N / 数据 D / 算力 C 三者都符合幂律 L ∝ x^(-α)。这是经验科学不是理论 —
        但拟合常数足够稳健,让 GPT-3 175B 的设计建立在"小模型外推"的赌注上,赌赢了。
      </p>

      <PowerLawCurves axis={axis} />
      <p className={styles.caption}>
        ↑ 三条 log-log 直线 · scale 一个数量级,loss 下降一个固定 fraction。
        α_N ≈ 0.076 / α_D ≈ 0.095 / α_C ≈ 0.05 — 数据是回报最高的轴。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setAxis("all")} style={btnStyle(axis === "all")}>三轴</button>
        <button type="button" onClick={() => setAxis("N")} style={btnStyle(axis === "N")}>L(N)</button>
        <button type="button" onClick={() => setAxis("D")} style={btnStyle(axis === "D")}>L(D)</button>
        <button type="button" onClick={() => setAxis("C")} style={btnStyle(axis === "C")}>L(C)</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={SCALING_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={SCALING_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              翻倍 vs 收益
            </div>
            <table style={{ width: "100%", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>轴</th>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>α</th>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>翻 2× → loss</th>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>翻 10× → loss</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <td style={{ padding: "8px" }}>N (参数)</td>
                  <td style={{ padding: "8px" }}>0.076</td>
                  <td style={{ padding: "8px" }}>×0.95</td>
                  <td style={{ padding: "8px" }}>×0.84</td>
                </tr>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <td style={{ padding: "8px" }}>D (数据)</td>
                  <td style={{ padding: "8px" }}>0.095</td>
                  <td style={{ padding: "8px" }}>×0.94</td>
                  <td style={{ padding: "8px" }}>×0.80</td>
                </tr>
                <tr>
                  <td style={{ padding: "8px" }}>C (算力)</td>
                  <td style={{ padding: "8px" }}>0.05</td>
                  <td style={{ padding: "8px" }}>×0.97</td>
                  <td style={{ padding: "8px" }}>×0.89</td>
                </tr>
              </tbody>
            </table>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              翻 10× 数据能把 loss 降到 0.80 — 数据是性价比最高的投资轴。
              这就是 LLaMA 路线选 "过训练小模型" 的根因。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
