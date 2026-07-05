import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DEIT_SOURCE_PATH } from "../lib/prose";
import { DistillationTokenDiagram } from "../widgets/DistillationTokenDiagram";
import { HardVsSoftLabelDiagram } from "../widgets/HardVsSoftLabelDiagram";
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

export function DistillationStage({ mechanism2Prose }: Props) {
  const [mode, setMode] = useState<"soft" | "hard">("hard");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Distillation Token + CNN Teacher
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        DeiT 加一个和 [CLS] 并列的 <strong>distillation token</strong>——[CLS] 学 ground truth
        标签,distill token 学 CNN teacher(RegNet-Y 16GF, 84.2%)的预测。两路损失
        平均相加,让 CNN 的 locality 归纳偏置通过蒸馏"补"给 ViT,而不需要改架构。
      </p>

      <DistillationTokenDiagram />
      <p className={styles.caption}>
        ↑ token 序列里插入 [DIST],和 [CLS] 各自接一个独立 head,分别监督真实标签与
        teacher 预测——distill token 贡献 1.6 分(81.8% → 83.4%)。
      </p>

      <div style={{ marginTop: "var(--space-8)" }}>
        <HardVsSoftLabelDiagram mode={mode} />
        <p className={styles.caption}>
          ↑ 切换看 soft distillation(KL 散度,Hinton 经典 KD)vs hard distillation
          (argmax one-hot,DeiT 采用)——DeiT 消融显示 hard 效果更好更简单。
        </p>
        <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
          <button type="button" onClick={() => setMode("hard")} style={btnStyle(mode === "hard")}>Hard(DeiT 采用)</button>
          <button type="button" onClick={() => setMode("soft")} style={btnStyle(mode === "soft")}>Soft(经典 KD)</button>
        </div>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={DEIT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              为什么用 CNN 而不是 ViT 当 teacher
            </div>
            <div style={{ fontSize: "var(--fs-sm)", lineHeight: 1.6, color: "var(--ink-primary)" }}>
              直觉上应该用更强的 ViT 互相蒸馏,但实验发现 CNN(RegNet)teacher 效果更好——
              CNN 内置的 locality / 平移不变性正是 ViT 缺的那块,跨架构蒸馏把这种"先验"
              传递给 student,而同架构蒸馏传不出新的归纳偏置。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
