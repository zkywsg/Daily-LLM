import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { FLASH_ATTN_SOURCE_PATH } from "../lib/prose";
import { MemoryHierarchyDiagram } from "../widgets/MemoryHierarchyDiagram";
import { TilingDiagram } from "../widgets/TilingDiagram";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function TilingStage({ intuitionProse, mechanism1Prose }: Props) {
  const [step, setStep] = useState(0);
  const totalSteps = 16; // 4x4 grid
  const activeI = Math.floor(step / 4);
  const activeJ = step % 4;

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Tiling — 把 Q/K/V 切块在 SRAM 里 blockwise 算
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        标准 attention 在 GPU 上的瓶颈不是 FLOPs,而是 HBM ↔ SRAM 的数据搬运 —
        S = QKᵀ 写 HBM、softmax 读写 HBM、P@V 读写 HBM,N×N 矩阵被物化三次。
        FlashAttention 把 Q 横切成行块、K/V 竖切成列块,每次只把一对小块加载到
        SRAM 里算,S_ij 用完即弃,整个 N×N 矩阵从不在 HBM 上完整存在。
      </p>

      <MemoryHierarchyDiagram />
      <p className={styles.caption}>
        ↑ SRAM 带宽是 HBM 的 10× 但容量小 2000× — attention 慢的根因是反复搬运 N×N 矩阵。
      </p>

      <TilingDiagram activeI={activeI} activeJ={activeJ} />
      <p className={styles.caption}>
        ↑ 4×4 分块演示,拖动进度条看 tiling 遍历顺序(Q 外层循环、K/V 内层循环)。
      </p>
      <input
        type="range"
        min={0}
        max={totalSteps - 1}
        step={1}
        value={step}
        onChange={(e) => setStep(parseInt(e.target.value))}
        style={{ width: "100%", marginTop: 8 }}
      />

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={FLASH_ATTN_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={FLASH_ATTN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              朴素实现的三次 HBM 往返
            </div>
            <pre style={{ fontSize: 11, lineHeight: 1.6, margin: 0, color: "var(--ink-primary)" }}>{`S = Q @ K.T     # 写 HBM
P = softmax(S)  # 读 S,写 P
O = P @ V       # 读 P,V,写 O
# N=4096: ~100MB / head / layer`}</pre>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              tiling 单独还不够 — softmax 要看整行才能算 max+sum,这正是机制二 online softmax 要解决的。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
