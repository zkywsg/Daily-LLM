import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DCGAN_SOURCE_PATH } from "../lib/prose";
import { BatchNormEffectDiagram } from "../widgets/BatchNormEffectDiagram";
import { STABILIZERS } from "../lib/data";
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

export function StabilizersStage({ mechanism2Prose }: Props) {
  const [withBn, setWithBn] = useState(true);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:BatchNorm + LeakyReLU + Adam(β₁=0.5) — 三个"必装"的训练稳定剂
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        单独把 MLP 换成 CNN 还不够 —— GAN 训练对归一化、激活函数、优化器超参数极其敏感。
        DCGAN 把 BatchNorm(除 G 输出层和 D 输入层)、D 用 LeakyReLU(0.2)、
        Adam(lr=2e-4, β₁=0.5)钉死成默认值,后来几年所有 GAN 工作几乎都照抄这套配置。
      </p>

      <BatchNormEffectDiagram withBn={withBn} />
      <p className={styles.caption}>
        ↑ 切换看有 / 没有 BatchNorm 时训练损失曲线的差异,以及 DCGAN 前后 GAN 复现成功率的对比。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setWithBn(true)} style={btnStyle(withBn)}>有 BatchNorm</button>
        <button type="button" onClick={() => setWithBn(false)} style={btnStyle(!withBn)}>没有 BatchNorm</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={DCGAN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              三个稳定剂各管什么
            </div>
            {STABILIZERS.map((row) => (
              <div key={row.name} style={{ marginBottom: 10 }}>
                <div style={{ fontSize: "var(--fs-sm)", fontWeight: 700, color: "var(--ink-primary)" }}>{row.name}</div>
                <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", lineHeight: 1.5 }}>
                  应用范围:{row.appliesTo}
                </div>
                <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-secondary)", lineHeight: 1.5 }}>
                  作用:{row.effect}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
