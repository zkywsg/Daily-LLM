import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { ROBERTA_SOURCE_PATH } from "../lib/prose";
import { AblationStepChart } from "../widgets/AblationStepChart";
import { ABLATION_STEPS } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function AblationStage({ intuitionProse, mechanism1Prose }: Props) {
  const [visibleSteps, setVisibleSteps] = useState(ABLATION_STEPS.length);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:10× 数据 + 4× 训练步 — 真正起决定作用的
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        2018-2019 年学界一窝蜂改 BERT 架构,但 Facebook AI 复现后发现:许多"BERT
        改进"其实不是架构胜出,是训练 recipe 胜出。RoBERTa 架构一行都不改,只把
        训练 recipe 调对 — 论文 Table 4 的系统消融显示,五项改动里"更多数据 + 更长
        训练"贡献了绝大部分提升,其他四项加起来涨不到 1 分。
      </p>

      <AblationStepChart visibleSteps={visibleSteps} />
      <p className={styles.caption}>
        ↑ 拖动逐步叠加五项改动,观察 SQuAD F1 / MNLI 的累积涨幅 — 最后一步(数据+训练步)贡献最大。
      </p>
      <input
        type="range"
        min={1}
        max={ABLATION_STEPS.length}
        step={1}
        value={visibleSteps}
        onChange={(e) => setVisibleSteps(parseInt(e.target.value))}
        style={{ width: "100%", marginTop: 8 }}
      />

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={ROBERTA_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={ROBERTA_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              方法论意义
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              RoBERTa 把"架构创新 vs 训练充分"两个变量第一次解开,确立"做架构改动
              之前先确认 baseline 是充分训练的"研究规范 — 这一原则在 Chinchilla
              修正 Kaplan scaling law 时再次体现。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
