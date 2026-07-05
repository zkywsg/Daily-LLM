import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { CYCLEGAN_SOURCE_PATH } from "../lib/prose";
import { CycleConsistencyDiagram } from "../widgets/CycleConsistencyDiagram";
import { CYCLE_TRAINING_CURVE, LOSS_TERMS } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function CycleConsistencyStage({ mechanism2Prose }: Props) {
  const [stepIdx, setStepIdx] = useState(0);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Cycle Consistency L1 Loss — 强迫保留内容
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        对抗 loss 只保证 G(x) 看起来像 Y 域,不保证内容保留。CycleGAN 的核心创新是加一条约束:
        如果 G 把马变成斑马,F 把斑马变回去,应该还是原来那只马(同样姿态、背景、光照)。
        这条 ‖F(G(x))-x‖₁ 的重建约束,是 CycleGAN 绕过配对数据的真正核心。
      </p>

      <CycleConsistencyDiagram step={stepIdx} />
      <p className={styles.caption}>
        ↑ 拖动滑块模拟训练步数增加,看 F(G(x)) 逐渐逼近原图 x,重建误差如何收缩。
      </p>
      <input
        type="range"
        min={0}
        max={CYCLE_TRAINING_CURVE.length - 1}
        step={1}
        value={stepIdx}
        onChange={(e) => setStepIdx(Number(e.target.value))}
        style={{ width: "100%", marginTop: 8 }}
      />
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "var(--fs-xs)", color: "var(--ink-muted)" }}>
        <span>训练步 0</span>
        <span>训练步 {CYCLE_TRAINING_CURVE[CYCLE_TRAINING_CURVE.length - 1].step}</span>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={CYCLEGAN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              总 loss 组成
            </div>
            {LOSS_TERMS.map((row) => (
              <div key={row.name} style={{ marginBottom: 10 }}>
                <div style={{ fontSize: "var(--fs-sm)", fontWeight: 700, color: "var(--ink-primary)" }}>{row.name}</div>
                <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", lineHeight: 1.5, fontFamily: "monospace" }}>
                  {row.formula}
                </div>
                <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-secondary)", lineHeight: 1.5 }}>
                  {row.role}
                </div>
              </div>
            ))}
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 6, lineHeight: 1.5, fontStyle: "italic" }}>
              总 loss = L_GAN + λ·L_cyc,λ=10(L1 而非 L2,重建更 sharp)
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
