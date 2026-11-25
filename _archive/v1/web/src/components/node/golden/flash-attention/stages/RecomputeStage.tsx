import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { FLASH_ATTN_SOURCE_PATH } from "../lib/prose";
import { RecomputeTradeoffDiagram } from "../widgets/RecomputeTradeoffDiagram";
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

export function RecomputeStage({ mechanism3Prose, synergyProse }: Props) {
  const [mode, setMode] = useState<"store" | "recompute">("store");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Recomputation in Backward — 反向不存中间,直接重算
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        反向传播需要 forward 算过的 S 和 P 来算梯度。标准做法把它们存在 HBM,
        但这又把 O(N²) 显存吃回去 — 机制一二的努力被反向吃光。FlashAttention 选择
        不存 S/P,反向时按 forward 同样的 tiling 重算一遍,只在 HBM 存最终输出 O、
        softmax 统计量 (m, ℓ) 和原始 Q/K/V。
      </p>

      <RecomputeTradeoffDiagram mode={mode} />
      <p className={styles.caption}>
        ↑ 切换看"存储"vs"重计算"两种反向策略的显存/算力/wall-clock 权衡。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setMode("store")} style={btnStyle(mode === "store")}>标准:存储 S/P</button>
        <button type="button" onClick={() => setMode("recompute")} style={btnStyle(mode === "recompute")}>FlashAttention:重计算</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={FLASH_ATTN_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={FLASH_ATTN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              抽掉任何一件,FlashAttention 都不成立
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>只有 tiling,没有 online softmax</strong>:softmax 仍要看整行,必须先物化 N×N 到 HBM</li>
              <li><strong>只有 online softmax,没有 tiling</strong>:SRAM 装不下整行,算法成立但 GPU 上跑不动</li>
              <li><strong>没有 recomputation</strong>:反向时 N²·b 的 S/P 还是要写 HBM,前两个机制白做</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
