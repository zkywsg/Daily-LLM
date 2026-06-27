import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GPT3_SOURCE_PATH } from "../lib/prose";
import type { AttentionPattern } from "../lib/scaling";
import { SparseAttentionPattern } from "../widgets/SparseAttentionPattern";
import { ComputeBudgetBar } from "../widgets/ComputeBudgetBar";
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

const PATTERNS: { v: AttentionPattern; label: string }[] = [
  { v: "dense", label: "Dense" },
  { v: "strided", label: "Strided" },
  { v: "fixed", label: "Fixed" },
];

export function SparseAttentionStage({ mechanism3Prose, synergyProse }: Props) {
  const [pattern, setPattern] = useState<AttentionPattern>("strided");
  const [seqLen, setSeqLen] = useState(2048);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Sparse Attention + 工程实现
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        175B 参数本身已经是工程挑战 —— 单 GPU 装不下,attention 又是 O(n²)。
        GPT-3 用 Sparse Transformer 的 attention 模式让 50% 的 layer 走 O(n√n),
        同时配 model parallelism、激活重计算等工程基础设施才真的能训出来。
      </p>

      <SparseAttentionPattern pattern={pattern} n={32} />
      <p className={styles.caption}>
        ↑ 切换三种 attention pattern 看 mask 形状变化。Dense 是全因果三角;
        Strided / Fixed 只算 token 之间的稀疏子集 —— 长 context 时省的算力很可观。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={GPT3_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={GPT3_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <ComputeBudgetBar seqLen={seqLen} />
          <div
            style={{
              marginTop: "var(--space-4)",
              padding: "var(--space-3)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-md)",
              background: "var(--bg-surface)",
            }}
          >
            <div
              style={{
                fontSize: "var(--fs-xs)",
                color: "var(--ink-muted)",
                marginBottom: 6,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
              }}
            >
              attention pattern
            </div>
            <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-3)" }}>
              {PATTERNS.map((p) => (
                <button
                  key={p.v}
                  type="button"
                  onClick={() => setPattern(p.v)}
                  style={btnStyle(pattern === p.v)}
                >
                  {p.label}
                </button>
              ))}
            </div>
            <label
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "var(--fs-sm)",
                color: "var(--ink-secondary)",
                marginBottom: 4,
              }}
            >
              <span>seq_len</span>
              <strong>{seqLen}</strong>
            </label>
            <input
              type="range"
              min={128}
              max={8192}
              step={128}
              value={seqLen}
              onChange={(e) => setSeqLen(parseInt(e.target.value, 10))}
              style={{ width: "100%" }}
            />
            <div
              style={{
                fontSize: "var(--fs-xs)",
                color: "var(--ink-muted)",
                marginTop: 4,
                lineHeight: 1.4,
              }}
            >
              GPT-3 原始 context = 2048,后来 GPT-3.5 / GPT-4 扩到 4K / 8K /
              32K / 128K,sparse attention + KV cache + RoPE 都是其中的工程支撑。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
