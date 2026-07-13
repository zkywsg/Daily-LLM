import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { INCEPTION_SOURCE_PATH } from "../lib/prose";
import { InceptionModuleDiagram } from "../widgets/InceptionModuleDiagram";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

export function MultiScaleModuleStage({ intuitionProse, mechanism1Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:多分支并行 — 同一层同时看多个尺度
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        VGG 用统一 3×3 卷积堆深来扩大感受野,浅层只能看局部、深层才能看全局。
        Inception 反其道而行:在同一层并行使用 1×1 / 3×3 / 5×5 / pool 多个尺度,
        让网络自己学如何加权这些通道 —— 信息流更扁更宽。
      </p>

      <InceptionModuleDiagram />
      <p className={styles.caption}>
        ↑ 输入并行走 4 条分支:纯 1×1、1×1→3×3、1×1→5×5、MaxPool→1×1,最后在通道维 concat。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={INCEPTION_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={INCEPTION_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              为什么不堆深,而是并行
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, paddingLeft: 18, lineHeight: 1.7 }}>
              <li>视觉模式没有单一尺度 —— 一张脸既有整体轮廓也有像素级细节</li>
              <li>VGG 堆 3 层 3×3 才等价 1 个 7×7 感受野,深浅层看的尺度不同</li>
              <li>Inception 在每层都同时看多尺度,由下一层 1×1 自适应加权</li>
            </ul>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: "var(--space-3) 0 0", lineHeight: 1.6 }}>
              这种"多分支自适应"思想后来被 ResNeXt、Xception、ViT 的 multi-head attention 各自借鉴。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
