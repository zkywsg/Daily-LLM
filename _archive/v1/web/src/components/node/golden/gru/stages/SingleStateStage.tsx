import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GRU_SOURCE_PATH } from "../lib/prose";
import { CellArchitectureCompare } from "../widgets/CellArchitectureCompare";
import { TrajectoryCompare } from "../widgets/TrajectoryCompare";
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

export function SingleStateStage({ mechanism3Prose, synergyProse }: Props) {
  const [side, setSide] = useState<"lstm" | "gru" | "both">("both");
  const [resetLow, setResetLow] = useState(false);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:单一状态 h — 不再分长期 / 短期记忆
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        GRU 去掉 LSTM 的 cell state C,只保留 hidden state h。LSTM 用 C 作长期记忆 highway、
        h 作短期对外接口;GRU 把两者合并,所有信息都在 h 里,参数省 25%(3 组矩阵 vs 4 组)。
        但训练只快 15-20% —— 因为候选 h̃ 用 r⊙h_{"{t-1}"},要等 r 算出来才能算,至少要 2 次大矩阵乘。
      </p>

      <CellArchitectureCompare side={side} />
      <p className={styles.caption}>
        ↑ LSTM 4 组矩阵 + 双状态 vs GRU 3 组矩阵 + 单一状态。点按钮聚焦其中一边。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setSide("lstm")} style={btnStyle(side === "lstm")}>聚焦 LSTM</button>
        <button type="button" onClick={() => setSide("gru")} style={btnStyle(side === "gru")}>聚焦 GRU</button>
        <button type="button" onClick={() => setSide("both")} style={btnStyle(side === "both")}>对比</button>
      </div>

      <TrajectoryCompare resetLow={resetLow} />
      <p className={styles.caption}>
        ↑ 同一模拟序列上 LSTM(C 粉虚线 / h 蓝线)vs GRU(h 绿线)走势。
        GRU 的 h 是凸组合,z_t 接近 0 时 h_t≈h_{"{t-1}"},梯度近似 1 直通 — 和 LSTM cell highway 本质一样。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setResetLow(false)} style={btnStyle(!resetLow)}>均匀 reset</button>
        <button type="button" onClick={() => setResetLow(true)} style={btnStyle(resetLow)}>周期性重启(句子边界)</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={GRU_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={GRU_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              后续影响
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li>Highway Networks(2015)直接用 GRU 风格 T⊙F(x)+(1-T)⊙x 做前馈深度门</li>
              <li>Transformer residual + LayerNorm 可看作 GRU (1-z,z) 退化成 (1,1) — 没有门但保留加法主干</li>
              <li>Bahdanau 原始 attention 论文用 GRU 实现</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
