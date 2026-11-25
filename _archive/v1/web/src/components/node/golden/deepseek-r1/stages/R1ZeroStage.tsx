import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DEEPSEEK_R1_SOURCE_PATH } from "../lib/prose";
import { PipelineCompareDiagram } from "../widgets/PipelineCompareDiagram";
import { ThinkingGrowthChart } from "../widgets/ThinkingGrowthChart";
import { TRAINING_STEP_GROWTH } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function R1ZeroStage({ intuitionProse, mechanism1Prose }: Props) {
  const [visibleSteps, setVisibleSteps] = useState(TRAINING_STEP_GROWTH.length);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:R1-Zero — 纯 RL 让推理涌现
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        reasoning 不是教出来的,是逼出来的。R1-Zero 的训练 pipeline 极简到几乎反直觉 —
        没有 SFT cold start、没有 PRM、没有 MCTS、没有 search,直接在 DeepSeek-V3 base
        上做 GRPO(rule-based reward)。训练过程中模型自己学到反思、回溯、自验证,
        思考长度自然增长,没有人为加 length reward。
      </p>

      <PipelineCompareDiagram />
      <p className={styles.caption}>
        ↑ R1-Zero(上)完全跳过 SFT;R1(下)在其基础上加 4 阶段训练。两条路径共享同一个 GRPO 内核。
      </p>

      <div style={{ marginTop: "var(--space-8)" }}>
        <ThinkingGrowthChart visibleSteps={visibleSteps} />
        <p className={styles.caption}>
          ↑ 拖动看训练 step 增加时思考长度(蓝)与 AIME 准确率(绿虚线)如何联合增长,💡 标记 "Aha moment" 涌现点。
        </p>
        <input
          type="range"
          min={1}
          max={TRAINING_STEP_GROWTH.length}
          step={1}
          value={visibleSteps}
          onChange={(e) => setVisibleSteps(parseInt(e.target.value))}
          style={{ width: "100%", marginTop: 8 }}
        />
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={DEEPSEEK_R1_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={DEEPSEEK_R1_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              reward 设计(极简)
            </div>
            <pre style={{ fontSize: 11, lineHeight: 1.6, margin: 0, color: "var(--ink-primary)" }}>{`def reward(question, response):
    acc = 1.0 if check_correct(...) else 0.0
    fmt = 1.0 if has_think_tags(response) else 0.0
    return acc + 0.5 * fmt
# 没有 PRM,没有"步骤3错了 -0.1"`}</pre>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              rule-based outcome reward 避免了 PRM 被 reward hacking 的问题 — DeepSeek 论文明确记录了尝试 PRM 失败的经历。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
