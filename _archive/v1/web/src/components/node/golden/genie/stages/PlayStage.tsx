import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GENIE_SOURCE_PATH } from "../lib/prose";
import { PlayWidget } from "../widgets/PlayWidget";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function PlayStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:动态模型 —— 给定 latent action 自回归生成下一帧
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        给定当前 token 序列和选择的离散动作,动态模型自回归预测下一帧的 token —— 用户可以用学到的动作逐帧"玩"生成出的世界。
      </p>
      <div className={styles.grid}>
        <div>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={GENIE_SOURCE_PATH} />
          </div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-6)", marginBottom: "var(--space-4)" }}>
            三件套协同
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={GENIE_SOURCE_PATH} />
          </div>
        </div>
        <div className={styles.stickyPanel}>
          <PlayWidget />
        </div>
      </div>
    </div>
  );
}
