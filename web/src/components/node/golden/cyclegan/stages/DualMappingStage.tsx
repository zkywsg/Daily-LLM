import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { CYCLEGAN_SOURCE_PATH } from "../lib/prose";
import { DualGeneratorDiscriminatorDiagram } from "../widgets/DualGeneratorDiscriminatorDiagram";
import { PairedVsUnpairedDiagram } from "../widgets/PairedVsUnpairedDiagram";
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

export function DualMappingStage({ intuitionProse, mechanism1Prose }: Props) {
  const [highlight, setHighlight] = useState<"xy" | "yx">("xy");
  const [pairMode, setPairMode] = useState<"paired" | "unpaired">("paired");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:双向 G + 双向 D — 形成 cycle 的结构前提
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        pix2pix 能学 X→Y,但需要严格配对数据。现实中马↔斑马、莫奈画↔照片这类任务根本拿不到配对。
        CycleGAN 的第一步是同时维护 G: X→Y、F: Y→X 两个 generator,以及 D_X、D_Y 两个 discriminator —
        没有反向 mapping F,就没法形成后面机制二的 cycle。
      </p>

      <DualGeneratorDiscriminatorDiagram highlight={highlight} />
      <p className={styles.caption}>
        ↑ 切换看 G: X→Y(马变斑马,D_Y 判别)与 F: Y→X(斑马变马,D_X 判别)两条独立方向。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setHighlight("xy")} style={btnStyle(highlight === "xy")}>G: X → Y</button>
        <button type="button" onClick={() => setHighlight("yx")} style={btnStyle(highlight === "yx")}>F: Y → X</button>
      </div>

      <div style={{ marginTop: "var(--space-8)" }}>
        <PairedVsUnpairedDiagram mode={pairMode} />
        <p className={styles.caption}>
          ↑ 对比 pix2pix 的配对数据要求 vs CycleGAN 的无配对域集合。
        </p>
        <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
          <button type="button" onClick={() => setPairMode("paired")} style={btnStyle(pairMode === "paired")}>pix2pix(配对)</button>
          <button type="button" onClick={() => setPairMode("unpaired")} style={btnStyle(pairMode === "unpaired")}>CycleGAN(无配对)</button>
        </div>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={CYCLEGAN_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={CYCLEGAN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              四个网络分工
            </div>
            <div style={{ fontSize: "var(--fs-sm)", lineHeight: 1.8, color: "var(--ink-secondary)" }}>
              <div><strong style={{ color: "var(--ink-primary)" }}>G: X→Y</strong> — 马 → 斑马</div>
              <div><strong style={{ color: "var(--ink-primary)" }}>F: Y→X</strong> — 斑马 → 马</div>
              <div><strong style={{ color: "var(--ink-primary)" }}>D_Y</strong> — 判别是否真斑马</div>
              <div><strong style={{ color: "var(--ink-primary)" }}>D_X</strong> — 判别是否真马</div>
            </div>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              对抗 loss 只保证"像目标域",不保证内容保留 — 这正是机制二 cycle loss 要解决的问题。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
