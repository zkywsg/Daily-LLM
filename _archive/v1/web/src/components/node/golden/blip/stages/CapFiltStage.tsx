import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { BLIP_SOURCE_PATH } from "../lib/prose";
import { CapFiltPipelineDiagram } from "../widgets/CapFiltPipelineDiagram";
import { CapFiltImprovementChart } from "../widgets/CapFiltImprovementChart";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function CapFiltStage({ mechanism2Prose }: Props) {
  const [step, setStep] = useState(0);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:CapFilt — 用模型自己清理和扩增数据
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        原始网络图文对很 noisy,直接训练效果有限。CapFilt(Captioner + Filter)
        用 BLIP 自己的 LM head 生成新 caption,用 ITM head 过滤不匹配的样本,
        把噪声数据变成更干净、更大规模的合成数据集。
      </p>

      <CapFiltPipelineDiagram step={step} />
      <p className={styles.caption}>
        ↑ 拖动查看 CapFilt 完整流程:filter 判断 → captioner 生成 → filter 再检查 → 扩增数据集。
      </p>
      <input
        type="range"
        min={0}
        max={4}
        step={1}
        value={step}
        onChange={(e) => setStep(parseInt(e.target.value))}
        style={{ width: "100%", marginTop: 8 }}
      />

      <div style={{ marginTop: "var(--space-8)" }}>
        <CapFiltImprovementChart />
        <p className={styles.caption}>
          ↑ 数据 bootstrap 后 COCO CIDEr / VQA 分数同步提升 — 数据质量的价值有时比加模型大小还大。
        </p>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={BLIP_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              synthetic data 成为标配
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              "用模型生成 + 过滤训练数据"这一思路后被 GPT-4 / Phi-3 / SD3 等
              多个工作沿用 — synthetic data + 自动过滤成为现代大模型训练的
              标配做法,而不只是 BLIP 的一个小 trick。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
