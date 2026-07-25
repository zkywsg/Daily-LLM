import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { WAV2VEC2_SOURCE_PATH } from "../lib/prose";
import { DownsampleWidget } from "../widgets/DownsampleWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function FeatureEncoderStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:CNN 特征编码器
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        多层一维卷积把原始波形压缩成较低频率的潜在特征序列,压缩掉冗余的高频细节,保留语音相关的结构信息。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={WAV2VEC2_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={WAV2VEC2_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <DownsampleWidget />
        </div>
      </div>
    </div>
  );
}
