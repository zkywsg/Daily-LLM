import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { LENET_SOURCE_PATH } from "../lib/prose";
import { LocalVsGlobalConnectionDiagram } from "../widgets/LocalVsGlobalConnectionDiagram";
import { ParamCountCompareChart } from "../widgets/ParamCountCompareChart";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function ConvolutionStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:局部连接卷积层 — 参数与图像尺寸解耦
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        MLP 把图像拉平成一维向量,每个像素都要连到每个隐藏单元,参数量随图像尺寸平方膨胀。
        卷积层把"局部相关性"这条先验直接写进结构——一小块滤波器在整张图上滑动共享,
        参数数量只取决于卷积核大小和通道数,和输入图像的空间尺寸完全解耦。
      </p>

      <LocalVsGlobalConnectionDiagram />
      <p className={styles.caption}>
        ↑ 切换看卷积的局部连接 + 权值共享,与 MLP 把每个像素连到每个隐藏单元的全连接方式。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={LENET_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={LENET_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              参数量差一个量级
            </div>
            <ParamCountCompareChart />
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: "var(--space-2) 0 0", lineHeight: 1.6 }}>
              LeNet-5 全网约 60K 参数,比同样输入接一层 MLP 隐层(约 100 万参数)还少一个量级——
              卷积把"局部+共享"先验写进结构,参数不再随图像尺寸平方膨胀。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
