import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { MUSICGEN_SOURCE_PATH } from "../lib/prose";
import { ConditionToggleWidget } from "../widgets/ConditionToggleWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function ConditionStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:文本 + 旋律双重条件控制
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        文本条件控制生成音乐的风格/描述内容,旋律条件让模型按指定旋律生成不同编曲风格的音乐,两种条件可单独或组合使用。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={MUSICGEN_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={MUSICGEN_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <ConditionToggleWidget />
        </div>
      </div>
    </div>
  );
}
