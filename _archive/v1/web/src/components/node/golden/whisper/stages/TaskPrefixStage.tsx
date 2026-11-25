import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { WHISPER_SOURCE_PATH } from "../lib/prose";
import { TaskPrefixWidget } from "../widgets/TaskPrefixWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function TaskPrefixStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:多任务统一格式
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用特殊 token 把转写、翻译、语言识别、时间戳预测统一编码进同一个 sequence-to-sequence 格式,一个模型同时具备多种能力。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={WHISPER_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={WHISPER_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <TaskPrefixWidget />
        </div>
      </div>
    </div>
  );
}
