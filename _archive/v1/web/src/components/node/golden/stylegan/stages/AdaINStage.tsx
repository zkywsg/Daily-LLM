import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { STYLEGAN_SOURCE_PATH } from "../lib/prose";
import { AdaINDiagram } from "../widgets/AdaINDiagram";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

type Highlight = "instnorm" | "affine" | "modulate" | null;

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function AdaINStage({ mechanism2Prose }: Props) {
  const [hl, setHl] = useState<Highlight>(null);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:AdaIN — 用 style 控制每层 feature 的统计量
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        每个 Conv layer 后:先 Instance Normalize 擦掉原统计信息,再用 w 通过
        learned linear 变成 (scale, bias) 重新调制。style 改变 feature 的
        均值方差 → 改变整体外观(颜色/纹理)但不改变空间结构(位置/形状)—
        这就是 "换 style 不换姿态" 的物理基础。
      </p>

      <AdaINDiagram highlight={hl} />
      <p className={styles.caption}>
        ↑ 三步:x → Instance Normalize → 用 w 生成的 (scale, bias) 重新调制 → 输出。
        点按钮聚焦其中一步。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setHl(null)} style={btnStyle(hl === null)}>全部</button>
        <button type="button" onClick={() => setHl("instnorm")} style={btnStyle(hl === "instnorm")}>① Instance Norm</button>
        <button type="button" onClick={() => setHl("affine")} style={btnStyle(hl === "affine")}>② w → (scale, bias)</button>
        <button type="button" onClick={() => setHl("modulate")} style={btnStyle(hl === "modulate")}>③ modulate</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={STYLEGAN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              AdaIN 代码(简化)
            </div>
            <pre style={{ fontSize: 10, lineHeight: 1.5, margin: 0, color: "var(--ink-primary)", overflow: "auto" }}>{`class AdaIN(nn.Module):
    def __init__(self, w_dim, channels):
        super().__init__()
        self.affine = nn.Linear(w_dim, 2 * channels)

    def forward(self, x, w):
        style = self.affine(w)
        scale, bias = style.chunk(2, dim=1)
        scale = scale[:, :, None, None]
        bias  = bias[:, :, None, None]
        # Instance normalize
        mu = x.mean([2, 3], keepdim=True)
        sd = x.std([2, 3], keepdim=True) + 1e-8
        x  = (x - mu) / sd
        # Modulate
        return scale * x + bias`}</pre>
          </div>

          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              妙处
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li>借自 Huang 2017 neural style transfer</li>
              <li>每层用不同 w 注入 — 就是 StyleGAN 的关键创新</li>
              <li>style 改统计量 = 改外观;不动空间 = 保姿态</li>
              <li>StyleGAN2 后来去 AdaIN 用 weight modulation 消除水滴伪影</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
