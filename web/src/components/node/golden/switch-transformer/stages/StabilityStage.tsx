import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { SWITCH_TRANSFORMER_SOURCE_PATH } from "../lib/prose";
import { CapacityFactorDiagram } from "../widgets/CapacityFactorDiagram";
import { PrecisionCompareDiagram } from "../widgets/PrecisionCompareDiagram";
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

export function StabilityStage({ mechanism3Prose, synergyProse }: Props) {
  const [capacityFactor, setCapacityFactor] = useState(1.25);
  const [precisionMode, setPrecisionMode] = useState<"fp32-all" | "bf16-all" | "selective">("selective");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:稳定性兜底 — Capacity Factor + Selective Precision
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        即使有 f·P loss,某个 batch 内还是可能瞬时偏向某几个 expert —— capacity
        factor 给每个 expert 设一个 token 容量上限,撑爆的 token 直接 drop
        (走 residual 跳过这一层 MoE)。另一个坑是精度:MoE 的 softmax/log
        计算在纯 bf16 下经常发散,Switch 提出 selective precision —— 主体用
        bf16 省显存,只有 router 用 fp32 保精度。
      </p>

      <CapacityFactorDiagram capacityFactor={capacityFactor} />
      <p className={styles.caption}>
        ↑ 拖动 capacity factor 看 token 撑爆/被 drop 的行为。1.0 = 理想均匀时刚好装下,
        实际论文用 1.25-2.0 留缓冲。
      </p>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 10 }}>
        <input
          type="range"
          min={0.8}
          max={2.5}
          step={0.05}
          value={capacityFactor}
          onChange={(e) => setCapacityFactor(Number(e.target.value))}
          style={{ flex: 1 }}
        />
        <span style={{ fontSize: "var(--fs-sm)", fontFamily: "ui-monospace", minWidth: 40 }}>
          {capacityFactor.toFixed(2)}
        </span>
      </div>

      <div style={{ marginTop: "var(--space-8)" }}>
        <PrecisionCompareDiagram mode={precisionMode} />
        <p className={styles.caption}>
          ↑ 切换看三种精度方案:全 fp32 稳但贵、全 bf16 便宜但易发散、
          selective precision(router 用 fp32)两者兼得。
        </p>
        <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
          <button type="button" onClick={() => setPrecisionMode("fp32-all")} style={btnStyle(precisionMode === "fp32-all")}>全 fp32</button>
          <button type="button" onClick={() => setPrecisionMode("bf16-all")} style={btnStyle(precisionMode === "bf16-all")}>全 bf16</button>
          <button type="button" onClick={() => setPrecisionMode("selective")} style={btnStyle(precisionMode === "selective")}>Selective Precision</button>
        </div>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={SWITCH_TRANSFORMER_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={SWITCH_TRANSFORMER_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div
            style={{
              padding: "var(--space-4)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-md)",
              background: "var(--bg-surface)",
            }}
          >
            <div
              style={{
                fontSize: "var(--fs-xs)",
                color: "var(--ink-muted)",
                marginBottom: 8,
                fontWeight: 600,
                textTransform: "uppercase",
              }}
            >
              三件套缺一不可
            </div>
            <div style={{ fontSize: "var(--fs-sm)", lineHeight: 1.7, color: "var(--ink-primary)" }}>
              只有 Top-1:省了路由,但没 load balancing → expert 立刻塌缩<br />
              只有 f·P loss:训练均衡,但没 capacity 兜底 → 推理时撑爆 OOM<br />
              只有 capacity + precision:训练不挂,但 top-K 还是 4 → 通信量 4×
            </div>
            <div
              style={{
                marginTop: 12,
                fontSize: "var(--fs-xs)",
                color: "var(--ink-muted)",
                fontStyle: "italic",
              }}
            >
              三件套首次组合 → 1.57T 总参 / 11B 激活 / T5-XXL 同质量 4× 提速
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
