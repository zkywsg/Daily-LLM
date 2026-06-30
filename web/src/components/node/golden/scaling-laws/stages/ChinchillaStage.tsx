import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SCALING_SOURCE_PATH } from "../lib/prose";
import { NDPlanePlot } from "../widgets/NDPlanePlot";
import { MODEL_POINTS } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
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

export function ChinchillaStage({ mechanism2Prose }: Props) {
  const [showKaplan, setShowKaplan] = useState(true);
  const [showChinchilla, setShowChinchilla] = useState(true);
  const [showLlama, setShowLlama] = useState(true);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Chinchilla 修正 — N : D ≈ 1 : 20
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Chinchilla(DeepMind 2022)用更严谨的实验设计修正 Kaplan:固定算力下,
        参数 N 和数据 D 应等比例扩张,N:D ≈ 1:20。这意味着 GPT-3 (175B / 300B)
        严重训练不足约 1/10,最优搭档应该是 ~63B 模型。
        Chinchilla-70B (1.4T token) 击败 Gopher 280B 验证了这一结论。
      </p>

      <NDPlanePlot showKaplan={showKaplan} showChinchilla={showChinchilla} showLlama={showLlama} />
      <p className={styles.caption}>
        ↑ N-D log-log 平面上三条策略线 + 实际模型点位。
        GPT-3 / Gopher / MT-NLG 在 Kaplan 线上(参数太多);
        Chinchilla-70B 落在 1:20 线;LLaMA 系列远在对角线右上方(D ≫ 20N)。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setShowKaplan(!showKaplan)} style={btnStyle(showKaplan)}>Kaplan 线</button>
        <button type="button" onClick={() => setShowChinchilla(!showChinchilla)} style={btnStyle(showChinchilla)}>Chinchilla 线</button>
        <button type="button" onClick={() => setShowLlama(!showLlama)} style={btnStyle(showLlama)}>LLaMA 线</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={SCALING_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              主要 LLM 的 D/N 比 vs Chinchilla 20
            </div>
            <table style={{ width: "100%", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>模型</th>
                  <th style={{ textAlign: "right", padding: "6px 8px", fontWeight: 600 }}>D/N</th>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>状态</th>
                </tr>
              </thead>
              <tbody>
                {MODEL_POINTS.map((m) => {
                  const status = m.ratio < 5 ? "严重 under" : m.ratio < 25 ? "Chinchilla 区" : "over-trained";
                  const color = m.ratio < 5 ? "#9ca3af" : m.ratio < 25 ? "#ec4899" : "#10b981";
                  return (
                    <tr key={m.name} style={{ borderBottom: "1px solid var(--border)" }}>
                      <td style={{ padding: "6px 8px" }}>{m.name}</td>
                      <td style={{ padding: "6px 8px", textAlign: "right", fontFamily: "ui-monospace, monospace" }}>{m.ratio.toFixed(1)}</td>
                      <td style={{ padding: "6px 8px", color }}>{status}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              LLaMA-3 8B 的 1875:1 远超 Chinchilla 20:1,
              因为推理成本驱动:小模型即使训练浪费些算力,部署时省 10×。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
