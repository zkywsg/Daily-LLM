import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { DISTILBERT_SOURCE_PATH } from "../lib/prose";
import { TeacherStudentDistillationDiagram } from "../widgets/TeacherStudentDistillationDiagram";
import { TripleLossDiagram } from "../widgets/TripleLossDiagram";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function TripleLossStage({ intuitionProse, mechanism1Prose }: Props) {
  const [temperature, setTemperature] = useState(2);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:三损失联合训练
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Hinton 2015 的知识蒸馏洞察:让小 student 模仿大 teacher 的"软分布"——teacher
        的 softmax 输出包含类别间的相对关系,比硬标签信息量大得多。DistilBERT 把
        蒸馏损失(KL)、原始 MLM 损失、隐状态 cosine 对齐损失联合训练,三者按
        α=0.5 / β=0.2 / γ=0.1 加权。
      </p>

      <TeacherStudentDistillationDiagram temperature={temperature} />
      <p className={styles.caption}>
        ↑ 拖动温度滑块看软标签分布如何随 τ 变化——τ 越大分布越平滑,类别间相对关系越明显。
      </p>
      <input
        type="range"
        min={1}
        max={10}
        step={0.5}
        value={temperature}
        onChange={(evt) => setTemperature(Number(evt.target.value))}
        style={{ width: "100%", marginTop: 12 }}
        aria-label="蒸馏温度滑块"
      />

      <div style={{ marginTop: "var(--space-8)" }}>
        <TripleLossDiagram />
        <p className={styles.caption}>
          ↑ 三损失按权重合并成总训练信号——蒸馏权重最大,MLM 与 cosine 分别提供硬标签约束和中间表征对齐。
        </p>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={DISTILBERT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={DISTILBERT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              为什么 × t² 抵消梯度缩放?
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              温度软化后 logit 变小,梯度按 1/t² 缩放,标准 KD 公式要乘 t² 抵消——
              否则蒸馏损失的梯度贡献会被温度稀释,student 学不到 teacher 的软信号。
              DistilBERT 训练用 τ=2,推理时换回 τ=1(硬 softmax)。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
