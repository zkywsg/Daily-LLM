import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { LORA_SOURCE_PATH } from "../lib/prose";
import { InferenceMergeFlow } from "../widgets/InferenceMergeFlow";
import { MethodComparisonTable } from "../widgets/MethodComparisonTable";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function InferenceMergeStage({ mechanism3Prose, synergyProse }: Props) {
  const [merged, setMerged] = useState(true);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:推理时合并(零延迟)
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        Adapter 把新层串到 transformer 里 → 推理多一层运算。LoRA 并联在
        W₀ 旁边,部署前可直接计算 W' = W₀ + (α/r)·BA 合并回单个矩阵
        —— 推理图谱跟原模型一模一样,零额外延迟。
      </p>

      <InferenceMergeFlow merged={merged} />
      <p className={styles.caption}>
        ↑ 切换"是否合并"看推理图谱变化。未合并时两路并联;合并后
        只剩一次矩阵乘,跟原模型完全等价。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={LORA_SOURCE_PATH} />
          </div>

          <h3
            style={{
              fontSize: "var(--fs-xl)",
              marginTop: "var(--space-8)",
              marginBottom: "var(--space-4)",
            }}
          >
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={LORA_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <MethodComparisonTable />
          <p className={styles.caption}>
            LoRA 是唯一"参数少 + 推理零延迟 + GLUE 几乎不掉"的三角形组合
            —— 这是它成为工业 PEFT 默认方案的原因。
          </p>
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
              推理模式
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              {[
                { v: false, label: "未合并(可切 task)" },
                { v: true, label: "合并 W' = W₀+BA(零延迟)" },
              ].map((opt) => {
                const active = merged === opt.v;
                return (
                  <button
                    key={String(opt.v)}
                    type="button"
                    onClick={() => setMerged(opt.v)}
                    style={{
                      padding: "4px 12px",
                      fontSize: "var(--fs-sm)",
                      borderRadius: "var(--radius-sm)",
                      border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
                      background: active ? "var(--accent-link)" : "var(--bg-surface)",
                      color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
                      cursor: "pointer",
                    }}
                  >
                    {opt.label}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
