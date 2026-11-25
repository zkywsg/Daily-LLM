import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { VGG_SOURCE_PATH } from "../lib/prose";
import { KernelStackingDiagram } from "../widgets/KernelStackingDiagram";
import { KernelParamSavingsChart } from "../widgets/KernelParamSavingsChart";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function SmallKernelStackingStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:3×3 砖头堆叠 — 用最小卷积单元拿等价感受野 + 更少参数
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        AlexNet / ZFNet / OverFeat 之前的共识是前几层要用大 kernel 才能拿到足够感受野。
        VGG 反过来把所有 conv 统一成 3×3,靠堆叠拿到等价感受野——同样的感受野,参数更少、
        深度更深、非线性也更多。
      </p>

      <KernelStackingDiagram />
      <p className={styles.caption}>
        ↑ 切换看 1 层 7×7 卷积与 3 层 3×3 堆叠卷积:原图上的等效感受野完全相同,但堆叠版本
        深度 +2、非线性 +2、参数少 45%。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={VGG_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={VGG_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              等价感受野,参数量差一截
            </div>
            <KernelParamSavingsChart />
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: "var(--space-2) 0 0", lineHeight: 1.6 }}>
              用 3 个 3×3 替换 1 个 7×7,参数降 45% 的同时深度多 2 层、非线性多 2 次——
              参数 / 深度 / 非线性三个维度同时改善。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
