import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DIT_SOURCE_PATH } from "../lib/prose";
import { AdaLNZeroBlock } from "../widgets/AdaLNZeroBlock";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

type Highlight = "condition" | "norm" | "scale" | "gate" | null;

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function AdaLNStage({ mechanism2Prose }: Props) {
  const [hl, setHl] = useState<Highlight>(null);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:adaLN-Zero — 把 timestep / class 条件注入每个 block
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Diffusion denoising 需要知道 timestep t (噪声强度) + class c (条件目标)。
        DiT 把 (t + c) 编码成 c_total,通过一个小 MLP 映射成 6 个调节参数
        (γ₁ β₁ α₁ γ₂ β₂ α₂),用 adaLN 替代标准 LN — 既 scale/shift 又 gate residual。
        MLP 最后一层 Zero init,初始 forward 退化成恒等,训练第 1 步就稳定。
      </p>

      <AdaLNZeroBlock highlight={hl} />
      <p className={styles.caption}>
        ↑ 左侧 condition path:(t, c) → MLP → 6 个 chunk 参数;
        右侧 forward path:attention + FFN 双 sub-block,每个里 LN → scale/shift → 运算 → gate residual。
        点按钮聚焦各部分。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setHl(null)} style={btnStyle(hl === null)}>全部</button>
        <button type="button" onClick={() => setHl("condition")} style={btnStyle(hl === "condition")}>① condition MLP</button>
        <button type="button" onClick={() => setHl("norm")} style={btnStyle(hl === "norm")}>② LayerNorm</button>
        <button type="button" onClick={() => setHl("scale")} style={btnStyle(hl === "scale")}>③ scale/shift (γ, β)</button>
        <button type="button" onClick={() => setHl("gate")} style={btnStyle(hl === "gate")}>④ gate (α)</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={DIT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              DiT Block 核心代码 (PyTorch)
            </div>
            <pre style={{ fontSize: 10, lineHeight: 1.5, margin: 0, color: "var(--ink-primary)", overflow: "auto" }}>{`def forward(self, x, c):
    # c = t_embed(t) + y_embed(class)
    shift1, scale1, gate1, shift2, scale2, gate2 = \\
        self.adaLN_modulation(c).chunk(6, dim=-1)

    # attention sub-block
    h = self.norm1(x) * (1 + scale1) + shift1
    x = x + gate1 * self.attn(h, h, h)

    # FFN sub-block
    h = self.norm2(x) * (1 + scale2) + shift2
    x = x + gate2 * self.mlp(h)
    return x

# Zero init 关键
nn.init.zeros_(self.adaLN_modulation[-1].weight)
nn.init.zeros_(self.adaLN_modulation[-1].bias)
# → 初始 γ=0 β=0 α=0 → adaLN 退化 identity
# → 训练第 1 步就稳定 · 不需 LR warmup`}</pre>
          </div>

          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              对比其他条件注入方式
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>Cross-attention</strong>:参数多 · 训练慢 · GLIDE/Imagen 用</li>
              <li><strong>通道相加</strong>:简单但表达力弱 · DDPM 用</li>
              <li><strong>adaLN-Zero</strong>:参数高效 + 强表达 + 训练稳定 — DiT/SD3/Sora 用</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
