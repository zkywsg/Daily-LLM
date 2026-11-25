import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GAN_SOURCE_PATH } from "../lib/prose";
import { DiscriminatorBoundary } from "../widgets/DiscriminatorBoundary";
import { DiscriminatorScoreBars } from "../widgets/DiscriminatorScoreBars";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

const TOTAL_ITER = 100;

export function DiscriminatorStage({ mechanism2Prose }: Props) {
  const [iter, setIter] = useState(50);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Discriminator D — 区分真图 vs G 生成图
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        D 是个二分类器 —— 输入一个样本,输出"是真图"的概率。训练时给它真图标 1、
        给 G 生成图标 0,做 BCE loss。D 的"看错率"就是 G 的"造假成功率",
        D 和 G 你死我活构成了 GAN 的核心博弈。
      </p>

      <DiscriminatorBoundary iter={iter} totalIter={TOTAL_ITER} />
      <p className={styles.caption}>
        ↑ 背景色 = D 对该位置的判断概率(粉=真 / 蓝=假 / 灰=不确定)。
        训练初期到处是灰(D 还没学到),后期 4 个真模附近变粉、远离的变蓝
        —— D 学到了真分布的支撑集。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={GAN_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <DiscriminatorScoreBars iter={iter} totalIter={TOTAL_ITER} />
          <p className={styles.caption}>
            16 个样本(8 真 R + 8 假 F)的 D 输出分数。iter=0 全都 ≈0.5;
            iter→T 时 R 升到 ~1、F 降到 ~0。这是 D 训练有效的直接证据。
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
            <div
              style={{
                fontSize: "var(--fs-xs)",
                color: "var(--ink-muted)",
                marginTop: 4,
                lineHeight: 1.4,
              }}
            >
              iter 拉到 0 看"D 还没学会"的混乱;拉到 100 看"D 几乎完美"
              —— 此时 G 必须使尽全力骗它。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
