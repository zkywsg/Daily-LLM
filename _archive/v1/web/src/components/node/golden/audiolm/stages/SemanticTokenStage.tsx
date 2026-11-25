import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { AUDIOLM_SOURCE_PATH } from "../lib/prose";
import { SemanticTokenWidget } from "../widgets/SemanticTokenWidget";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function SemanticTokenStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:语义 token
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用自监督模型的中间层表征提取粗粒度语义 token,负责"接下来该说/演奏什么内容、是谁在说/演奏"。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={AUDIOLM_SOURCE_PATH} />
          </div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7, marginTop: "var(--space-6)" }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={AUDIOLM_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <SemanticTokenWidget />
        </div>
      </div>
    </div>
  );
}
