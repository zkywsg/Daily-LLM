import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { TRANSFORMER_SOURCE_PATH } from "../lib/prose";
import { HeadSplitSVG } from "../widgets/HeadSplitSVG";
import { HeadCountSlider } from "../widgets/HeadCountSlider";
import { PerHeadHeatmapGrid } from "../widgets/PerHeadHeatmapGrid";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

// 复用 Stage 1 的 demo 句子,让 viewer 直接看到"同一个句子,h 个 head 关注模式不一样"。
const DEMO_TOKENS = ["The", "cat", "sat", "on", "the", "mat"];
const D_MODEL = 32;

export function MultiHeadStage({ mechanism2Prose }: Props) {
  const [numHeads, setNumHeads] = useState(4);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Multi-Head 多角度切片
      </h2>
      <p
        style={{
          fontSize: "var(--fs-md)",
          color: "var(--ink-secondary)",
          marginBottom: "var(--space-8)",
        }}
      >
        单个注意力头只能学一种关系(比如"verb 看主语")。把 d_model 切成
        h 个低维子空间,让每个 head 独立学一种,再 concat 回去 —— 一次
        forward 学多种语言关系。
      </p>

      <HeadSplitSVG dModel={D_MODEL} numHeads={numHeads} />
      <p className={styles.caption}>
        ↑ 拖下面的 head 数 slider,看 d_model 怎么被切成更多 / 更少子空间。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3
            style={{
              fontSize: "var(--fs-xl)",
              marginBottom: "var(--space-4)",
            }}
          >
            机制
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer
              markdown={mechanism2Prose}
              sourcePath={TRANSFORMER_SOURCE_PATH}
            />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <PerHeadHeatmapGrid
            tokens={DEMO_TOKENS}
            dModel={D_MODEL}
            numHeads={numHeads}
          />
          <p className={styles.caption}>
            每个 head 一张迷你热力图(行 query / 列 key)。注意不同 head 的
            高亮位置不一样 —— 这是 Multi-Head 想要的"多视角"。
          </p>
          <div style={{ marginTop: "var(--space-4)" }}>
            <HeadCountSlider
              value={numHeads}
              onChange={setNumHeads}
              dModel={D_MODEL}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
