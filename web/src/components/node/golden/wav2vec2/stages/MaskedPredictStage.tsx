import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { WAV2VEC2_SOURCE_PATH } from "../lib/prose";
import { MaskedPredictWidget } from "../widgets/MaskedPredictWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function MaskedPredictStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Transformer 上下文编码 + 掩码预测
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        随机 mask 掉若干帧,Transformer 需要从上下文预测被 mask 位置对应的量化目标——这是自监督训练的"完形填空"任务。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={WAV2VEC2_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={WAV2VEC2_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <MaskedPredictWidget />
        </div>
      </div>
    </div>
  );
}
