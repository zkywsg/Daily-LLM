import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { AUDIOLM_SOURCE_PATH } from "../lib/prose";
import { CascadeWidget } from "../widgets/CascadeWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function CascadeStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:层级式级联生成
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        语义 token → 粗声学 token → 细声学 token 三阶段级联,粗细粒度的 token 之间有清晰的条件依赖关系。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={AUDIOLM_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={AUDIOLM_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <CascadeWidget />
        </div>
      </div>
    </div>
  );
}
