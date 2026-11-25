import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SWIN_SOURCE_PATH } from "../lib/prose";
import { WindowPartitionDiagram } from "../widgets/WindowPartitionDiagram";
import { ComplexityChart } from "../widgets/ComplexityChart";
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

export function WindowedAttnStage({ intuitionProse, mechanism1Prose }: Props) {
  const [mode, setMode] = useState<"full" | "windowed">("windowed");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Windowed Attention
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        ViT/DeiT 所有层同一分辨率,高分辨率(800×800 detection)attention 直接 OOM。
        Swin 把 attention 限制在不重叠的 M×M 局部窗口(典型 M=7)里,复杂度从 O(N²)
        降到 O(N·M²)=O(N)。但窗口之间没有信息交换 — 跨窗口边界的物体会被切割,
        这是下一机制要解决的问题。
      </p>

      <WindowPartitionDiagram mode={mode} />
      <p className={styles.caption}>
        ↑ 全局 attention 每个 patch 连所有 patch;windowed 只连窗口内。
        点按钮切换看连接密度差异。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setMode("full")} style={btnStyle(mode === "full")}>全局 Attention</button>
        <button type="button" onClick={() => setMode("windowed")} style={btnStyle(mode === "windowed")}>Windowed Attention</button>
      </div>

      <ComplexityChart />
      <p className={styles.caption}>
        ↑ 4 个分辨率下的复杂度对比。800² 时全局 attention 6.25M 次操作,
        windowed 只需 245K 次 — 差距随分辨率平方增长。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={SWIN_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={SWIN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              56×56 feature map 举例
            </div>
            <pre style={{ fontSize: 11, lineHeight: 1.6, margin: 0, color: "var(--ink-primary)" }}>{`切成 8×8 = 64 个窗口
每窗口 7×7 = 49 patches

窗口内 attention: 49² = 2401 次/窗口
总计算: 64 × 2401 = 154K 次

对比全图 attention: (56×56)² = 9.8M 次`}</pre>
          </div>
        </div>
      </div>
    </div>
  );
}
