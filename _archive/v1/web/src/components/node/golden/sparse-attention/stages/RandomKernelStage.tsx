import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SPARSE_ATTN_SOURCE_PATH } from "../lib/prose";
import { ComplexityCurve } from "../widgets/ComplexityCurve";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 10px",
  fontSize: "var(--fs-xs)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

const MODEL_NAMES = ["Dense", "Sparse Transformer", "Reformer", "Longformer/BigBird"];

export function RandomKernelStage({ mechanism3Prose, synergyProse }: Props) {
  const [idx, setIdx] = useState(-1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Random 连接 + CUDA Kernel — 理论保证 + 工程落地
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Local + Global 构成的图直径仍可能很大,加少量随机边(r≈3)后根据 small-world
        network 理论,图的有效直径降到 O(log N)。BigBird 论文证明:堆 O(N) 层可逼近
        任意 seq-to-seq 函数,与 dense 表达力等价。但 CUDA Kernel 是工程关键 —
        dense mask 实现仍是 O(N²),必须手写 kernel 只算非零位置才能真正达到 O(N)。
      </p>

      <ComplexityCurve highlightModel={idx} />
      <p className={styles.caption}>
        ↑ log-log 复杂度对比曲线。点按钮聚焦某个模型。N=8K 时 dense 需 64GB 超出 A100,
        O(N) 方案仅需约 2GB。
      </p>
      <div style={{ display: "flex", gap: 4, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setIdx(-1)} style={btnStyle(idx === -1)}>全部</button>
        {MODEL_NAMES.map((n, i) => (
          <button key={i} type="button" onClick={() => setIdx(i)} style={btnStyle(idx === i)}>{n}</button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={SPARSE_ATTN_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={SPARSE_ATTN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              为什么 dense mask 实现不够?
            </div>
            <pre style={{ fontSize: 10, lineHeight: 1.5, margin: 0, color: "var(--ink-primary)", overflow: "auto" }}>{`# 语义正确但仍是 O(N²)
scores = torch.matmul(q, k.transpose(-2,-1))
mask = make_sliding_window_mask(N, w, g_idx)
scores = scores.masked_fill(~mask, -inf)
# ↑ matmul 已经算了完整 N×N 矩阵！

# 真正 O(N) 需要手写 CUDA kernel
# 只计算非零位置(diagonaled_mm.cu)`}</pre>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              没有 CUDA kernel,"稀疏"只在数学定义上稀疏,不会成为生产可用方案。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
