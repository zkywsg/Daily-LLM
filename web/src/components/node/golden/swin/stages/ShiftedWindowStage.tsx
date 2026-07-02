import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SWIN_SOURCE_PATH } from "../lib/prose";
import { ShiftedWindowDiagram } from "../widgets/ShiftedWindowDiagram";
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

export function ShiftedWindowStage({ mechanism2Prose }: Props) {
  const [showShift, setShowShift] = useState(true);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Shifted Window — 让信息跨窗口流动
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        交替使用两种窗口划分:W-MSA 标准窗口 / SW-MSA 窗口整体偏移 (M/2, M/2)。
        第 n 层窗口边界在第 n+1 层窗口中央,两个 block 配合就完成窗口间信息混合。
        工程上用 cyclic shift + attention mask 巧妙实现,和标准窗口开销完全等价,
        零额外计算。
      </p>

      <ShiftedWindowDiagram showShift={showShift} />
      <p className={styles.caption}>
        ↑ 左标准窗口,右偏移窗口(相差 M/2)。注意色块边界完全错开 —
        原本在标准窗口边界处的 patch,在偏移窗口里被合并到同一色块。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setShowShift(!showShift)} style={btnStyle(showShift)}>
          {showShift ? "隐藏偏移窗口" : "显示偏移窗口"}
        </button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={SWIN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              Cyclic Shift 实现
            </div>
            <pre style={{ fontSize: 10, lineHeight: 1.5, margin: 0, color: "var(--ink-primary)", overflow: "auto" }}>{`if self.shift_size > 0:
    x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))

x_windows = window_partition(x, window_size)
attn = self.attn(x_windows, mask=self.attn_mask)  # SW-MSA 需要 mask
x = window_reverse(attn, window_size, H, W)

if self.shift_size > 0:
    x = torch.roll(x, shifts=(shift, shift), dims=(1, 2))  # 逆移位`}</pre>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              整个 feature map 循环移位,再切等大窗口;attention mask 掉跨边界 patch 对,
              防止图像左边突然 attend 到图像右边。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
