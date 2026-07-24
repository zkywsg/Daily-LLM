import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GAMENGEN_SOURCE_PATH } from "../lib/prose";
import { NextFramePredictWidget } from "../widgets/NextFramePredictWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function DiffusionPredictStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:条件 diffusion 模型预测下一帧
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        游戏引擎的渲染循环本质上是"给定历史状态和玩家输入,画出下一帧"的函数——diffusion 模型直接学会了这个函数。
      </p>
      <div className={styles.grid}>
        <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
          <MarkdownRenderer markdown={mechanism2Prose} sourcePath={GAMENGEN_SOURCE_PATH} />
        </div>
        <div className={styles.stickyPanel}>
          <NextFramePredictWidget />
        </div>
      </div>
    </div>
  );
}
