import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SWIN_SOURCE_PATH } from "../lib/prose";
import { PatchMergingPyramid } from "../widgets/PatchMergingPyramid";
import { SWIN_STAGES } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 10px",
  fontSize: "var(--fs-xs)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function PatchMergingStage({ mechanism3Prose, synergyProse }: Props) {
  const [idx, setIdx] = useState(-1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Patch Merging — 层级化产出多尺度特征
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        每个 stage 结束后做一次 Patch Merging:把 2×2 邻居 patch 沿 channel 拼接,
        再过 linear 把 channel 减半。空间 2× 降、channel 2× 增,这和 CNN(ResNet stage 间)
        完全一样。最终产出 4 个尺度的特征图(56/28/14/7),可以直接接 FPN / 检测 / 分割 head —
        这是 ViT 单一分辨率做不到的。
      </p>

      <PatchMergingPyramid highlightIdx={idx} />
      <p className={styles.caption}>
        ↑ 4 个 stage,分辨率逐级减半,channel 逐级翻倍。点按钮聚焦某一 stage。
      </p>
      <div style={{ display: "flex", gap: 4, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setIdx(-1)} style={btnStyle(idx === -1)}>全部</button>
        {SWIN_STAGES.map((s, i) => (
          <button key={i} type="button" onClick={() => setIdx(i)} style={btnStyle(idx === i)}>{s.name}</button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={SWIN_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={SWIN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              PatchMerging 代码
            </div>
            <pre style={{ fontSize: 10, lineHeight: 1.5, margin: 0, color: "var(--ink-primary)", overflow: "auto" }}>{`class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm = nn.LayerNorm(4*dim)

    def forward(self, x, H, W):
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # 偶偶
        x1 = x[:, 1::2, 0::2, :]  # 奇偶
        x2 = x[:, 0::2, 1::2, :]  # 偶奇
        x3 = x[:, 1::2, 1::2, :]  # 奇奇
        x = torch.cat([x0,x1,x2,x3], -1)  # [B,H/2,W/2,4C]
        return self.reduction(self.norm(x))`}</pre>
          </div>
        </div>
      </div>
    </div>
  );
}
