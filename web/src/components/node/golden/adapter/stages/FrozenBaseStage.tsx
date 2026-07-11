import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { ADAPTER_SOURCE_PATH } from "../lib/prose";
import { FrozenVsTrainableDiagram } from "../widgets/FrozenVsTrainableDiagram";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function FrozenBaseStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:冻结 base + 只训 Adapter
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        Bottleneck 结构 + 零初始化 residual 保证 adapter 能安全插入、平滑训练,
        但要真正省参数、防遗忘,还差最后一步:训练时把 BERT 原参数完全冻结,
        只让梯度流过 Adapter₁、Adapter₂、LayerNorm 和任务 head —— 这是 PEFT
        "冻结 base + 训小模块"范式的核心。
      </p>

      <FrozenVsTrainableDiagram />
      <p className={styles.caption}>
        ↑ 灰色 = 冻结的 BERT 权重(99% 参数,保留通用知识);粉色 = 可训练的
        Adapter + LayerNorm(&lt;1% 参数,承担任务特化)。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={ADAPTER_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={ADAPTER_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              抽掉任何一件,PEFT 范式都不成立
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>只有 Bottleneck</strong>:adapter 设计好了但 base 也训 → 退回全参微调,丢了存储和遗忘的好处</li>
              <li><strong>只有零初始化 Residual</strong>:没 bottleneck 用满秩 adapter → 参数等同 FFN(36M/层),压缩失败</li>
              <li><strong>只有冻结 base</strong>:adapter 无残差或非零初始化 → 训练初期 adapter 随机扰动 base 表征 → 训练崩</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
