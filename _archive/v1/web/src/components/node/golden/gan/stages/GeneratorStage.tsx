import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GAN_SOURCE_PATH } from "../lib/prose";
import { GeneratorFlow } from "../widgets/GeneratorFlow";
import { DistributionFittingScatter } from "../widgets/DistributionFittingScatter";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

const TOTAL_ITER = 100;

export function GeneratorStage({ intuitionProse, mechanism1Prose }: Props) {
  const [iter, setIter] = useState(60);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Generator G — 从噪声 z 映射到图像
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        G 不学 p_data 的显式密度,只学一个"把噪声 z 映射到看起来像真图的 x̂"
        的采样器。这是 GAN 跟 VAE / Normalizing Flow 的关键区别 ——
        implicit 模型,不可计算 likelihood,但生成质量更高。
      </p>

      <GeneratorFlow />

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            直觉
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={GAN_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={GAN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <DistributionFittingScatter iter={iter} totalIter={TOTAL_ITER} />
          <p className={styles.caption}>
            ↑ 2D 玩具数据,4 模真分布(粉色)+ G 生成分布(蓝色)。
            拖 iter slider:开始 G 撒得很散,后期 G 在 4 个真模周围聚出 4 个蓝团 ——
            这就是"学到了真分布"。
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
            <label
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "var(--fs-sm)",
                color: "var(--ink-secondary)",
                marginBottom: 4,
              }}
            >
              <span>训练 iter</span>
              <strong>{iter} / {TOTAL_ITER}</strong>
            </label>
            <input
              type="range"
              min={0}
              max={TOTAL_ITER}
              step={1}
              value={iter}
              onChange={(e) => setIter(parseInt(e.target.value, 10))}
              style={{ width: "100%" }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
