import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { LENET_SOURCE_PATH } from "../lib/prose";
import { LeNetPipelineDiagram } from "../widgets/LeNetPipelineDiagram";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function FullyConnectedStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:全连接分类头 + 端到端反传 — 三件套协同
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        C5 用 5×5 卷积把空间维度压成 1×1 共 120 维,F6 是 84 维全连接,最后输出 10 类。
        整条网络从输入图到输出 logits 完全可微,误差通过反向传播一次性送回所有
        卷积 / 池化 / 全连接层的权重——这是 LeNet 区别于早期"非端到端"工作的关键。
      </p>

      <LeNetPipelineDiagram />
      <p className={styles.caption}>
        ↑ 点击或用下方按钮切换高亮层,看 C1 → S2 → C3 → S4 → C5 → F6 → output 的形状变化。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={LENET_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>
            三件套协同 — 缺一不可
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={LENET_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              少了任何一件都不成立
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, paddingLeft: 18, lineHeight: 1.7 }}>
              <li>只有卷积:没池化,感受野不够,学不到 high-level 结构</li>
              <li>只有池化:没卷积特征,池化谁也没意义</li>
              <li>只有 FC + 反传:就是 MLP,二维结构信息丢光</li>
            </ul>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: "var(--space-3) 0 0", lineHeight: 1.6 }}>
              三件套首次组合,LeNet 60K 参数在 MNIST 上达到 0.95% 错误率,定义后续 20 年 CNN 范式。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
