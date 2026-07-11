import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DEEPSEEK_V3_SOURCE_PATH } from "../lib/prose";
import { AuxLossFreeBalancingDiagram } from "../widgets/AuxLossFreeBalancingDiagram";
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

export function AuxLossFreeStage({ mechanism2Prose }: Props) {
  const [progress, setProgress] = useState(0);
  const [showAuxLoss, setShowAuxLoss] = useState(false);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Auxiliary-Loss-Free Load Balancing — 用 bias 替代辅助损失
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Switch / Mixtral 用 auxiliary loss 强制 expert 负载均衡,但这个额外 loss
        会和主任务 loss 打架 —— 为了 balance,router 不能完全自由选最优 expert。
        V3 给每个 expert 加一个 learnable bias:训练时监控负载,过载就调小 bias、
        欠载就调大,这个调整不参与梯度,只是在线动态平衡。
      </p>

      <AuxLossFreeBalancingDiagram progress={progress} showAuxLoss={showAuxLoss} />
      <p className={styles.caption}>
        ↑ 拖动进度条看 bias 如何随训练把负载拉平;勾选对比传统 aux loss 方案。
      </p>
      <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", minWidth: 70 }}>训练进度</span>
          <input
            type="range"
            min={0}
            max={100}
            value={Math.round(progress * 100)}
            onChange={(e) => setProgress(Number(e.target.value) / 100)}
            style={{ flex: 1, maxWidth: 320 }}
          />
          <span style={{ fontSize: "var(--fs-sm)", color: "var(--ink-primary)", minWidth: 40 }}>{Math.round(progress * 100)}%</span>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          <button type="button" onClick={() => setShowAuxLoss((v) => !v)} style={btnStyle(showAuxLoss)}>
            {showAuxLoss ? "隐藏" : "对比"} 传统 aux loss(Switch/Mixtral)
          </button>
        </div>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={DEEPSEEK_V3_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              score = sigmoid(x·W_g) + b
            </div>
            <div style={{ fontSize: "var(--fs-sm)", lineHeight: 1.7, color: "var(--ink-primary)" }}>
              b_i 是每个 expert 的 learnable bias,只在训练循环里按负载手工调整
              (过载调小、欠载调大),<strong>不参与反向传播</strong>。
            </div>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              主任务 loss 只看不含 bias 的原始 routing weight,负载均衡完全是"旁路"操作
              —— 这是 V3 比 Mixtral 训练更稳定的关键。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
