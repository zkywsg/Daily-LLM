import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { LDM_SOURCE_PATH } from "../lib/prose";
import { UnetBlockCrossAttn } from "../widgets/UnetBlockCrossAttn";
import { CfgScaleDemo } from "../widgets/CfgScaleDemo";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
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

const PROMPTS = [
  "a cat astronaut",
  "ancient temple at sunset",
  "watercolor of a sleeping fox",
];

export function CrossAttentionStage({ mechanism3Prose, synergyProse }: Props) {
  const [highlight, setHighlight] = useState<"self" | "cross" | "ffn">("cross");
  const [cfg, setCfg] = useState(7.5);
  const [promptIdx, setPromptIdx] = useState(0);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Cross-Attention — 一套接口接所有 condition
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        DDPM 原版只支持无条件 / 类条件。LDM 在 U-Net 每个 block 内加一层
        cross-attention,Q 来自 latent、K/V 来自 condition encoder。文本走 CLIP、
        类别走 embedding lookup、语义图走 conv —— 所有 encoder 都输出 [N, 768]
        喂同一个 cross-attention。这是 SD / ControlNet / inpainting 共用的架构基础。
      </p>

      <UnetBlockCrossAttn highlight={highlight} />
      <p className={styles.caption}>
        ↑ 点按钮聚焦三件之一。Cross-Attn 是关键:让 latent 被 condition 调制。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setHighlight("self")} style={btnStyle(highlight === "self")}>Self-Attn</button>
        <button type="button" onClick={() => setHighlight("cross")} style={btnStyle(highlight === "cross")}>Cross-Attn</button>
        <button type="button" onClick={() => setHighlight("ffn")} style={btnStyle(highlight === "ffn")}>FFN</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={LDM_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={LDM_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <CfgScaleDemo guidanceScale={cfg} prompt={PROMPTS[promptIdx]} />
          <p className={styles.caption}>
            ↑ Classifier-Free Guidance:同时算 cond 和 uncond 预测,
            按 scale s 推到 cond 方向。s=7.5 是 SD 默认甜蜜点。
          </p>
          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <label style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: 4 }}>
              <span>guidance scale</span><strong>{cfg.toFixed(1)}</strong>
            </label>
            <input type="range" min={0} max={20} step={0.5} value={cfg} onChange={(e) => setCfg(parseFloat(e.target.value))} style={{ width: "100%" }} />
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 6 }}>
              0 = 无 prompt · 7.5 = SD 默认 · &gt;12 过曝 + artifacts
            </div>

            <div style={{ marginTop: 12, paddingTop: 10, borderTop: "1px dashed var(--border)" }}>
              <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                prompt
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {PROMPTS.map((p, i) => (
                  <button key={i} type="button" onClick={() => setPromptIdx(i)} style={btnStyle(i === promptIdx)}>
                    {p}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
