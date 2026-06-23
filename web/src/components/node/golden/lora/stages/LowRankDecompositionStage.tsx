import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { LORA_SOURCE_PATH } from "../lib/prose";
import type { LoraConfig } from "../lib/math";
import { WeightDecompositionSVG } from "../widgets/WeightDecompositionSVG";
import { ParamAccountBar } from "../widgets/ParamAccountBar";
import { LoraControls } from "../widgets/LoraControls";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function LowRankDecompositionStage({
  intuitionProse,
  mechanism1Prose,
}: Props) {
  const [config, setConfig] = useState<LoraConfig>({
    d: 1024,
    k: 1024,
    r: 8,
    alpha: 8,
  });

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Low-Rank Decomposition
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        全量微调要更新 d×k 的整个 W,大模型上参数量爆炸。LoRA 的赌注是:
        微调阶段 ΔW 的"有效秩"其实很低,所以用 B(d×r) 和 A(r×k)
        两个细矩阵相乘近似 ΔW —— 只学 r·(d+k) 个参数。
      </p>

      <WeightDecompositionSVG d={config.d} k={config.k} r={config.r} />
      <p className={styles.caption}>
        ↑ W₀(灰)冻结不动。ΔW 拆成 B·A,只有蓝色两块是 trainable。
        拖右侧 rank slider 看 B、A 的"瘦身"程度。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            直觉
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={LORA_SOURCE_PATH} />
          </div>

          <h3
            style={{
              fontSize: "var(--fs-xl)",
              marginTop: "var(--space-6)",
              marginBottom: "var(--space-4)",
            }}
          >
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={LORA_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <ParamAccountBar config={config} />
          <p className={styles.caption}>
            log scale 横向对比:全量微调 vs LoRA 的可训练参数量。
            典型 d=k=1024、r=8 时 LoRA 只占 1.5%(下面百分比是单层比例)。
          </p>
          <div style={{ marginTop: "var(--space-4)" }}>
            <LoraControls config={config} onChange={setConfig} />
          </div>
        </div>
      </div>
    </div>
  );
}
