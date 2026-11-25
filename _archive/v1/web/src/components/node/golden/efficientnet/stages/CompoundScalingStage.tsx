import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { EFFICIENTNET_SOURCE_PATH } from "../lib/prose";
import { SingleVsCompoundScalingChart } from "../widgets/SingleVsCompoundScalingChart";
import { CompoundFormulaDiagram } from "../widgets/CompoundFormulaDiagram";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function CompoundScalingStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Compound Coefficient φ — 单参数控制整网规模
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        只加深 / 只加宽 / 只加分辨率各自都较快出现边际收益递减。EfficientNet 把"放大网络"
        重新表述成一个带约束的优化问题:引入单一复合缩放系数 φ,depth=α^φ、width=β^φ、
        resolution=γ^φ 同步放大,α·β²·γ²≈2 钉住 FLOPs 翻倍节奏,φ=0..7 依次得到 B0-B7。
      </p>

      <SingleVsCompoundScalingChart />
      <p className={styles.caption}>
        ↑ 点击图例可切换显示单轴 / 复合缩放曲线(示意,还原论文 Figure 5 的定性形状)——
        相同 FLOPs 预算下,三轴等比联合缩放的 Top-1 准确率明显高于任一单轴独自放大。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={EFFICIENTNET_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={EFFICIENTNET_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              拖动 φ 看三轴同步放大
            </div>
            <CompoundFormulaDiagram />
          </div>
        </div>
      </div>
    </div>
  );
}
