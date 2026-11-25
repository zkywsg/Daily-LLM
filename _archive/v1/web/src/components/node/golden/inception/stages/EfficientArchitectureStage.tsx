import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { INCEPTION_SOURCE_PATH } from "../lib/prose";
import { GoogLeNetPipelineDiagram } from "../widgets/GoogLeNetPipelineDiagram";
import { HeadParamCompareChart } from "../widgets/HeadParamCompareChart";
import { ModelParamCompareChart } from "../widgets/ModelParamCompareChart";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function EfficientArchitectureStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:GAP 替代大 FC — 三件套协同
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        GoogLeNet 去掉 VGG/AlexNet 那两层 4096 维的 FC,最后一个 Inception block 输出直接做
        Global Average Pooling,再接一层 FC 到 1000 类 —— 这一步贡献了 5M 整网参数预算的主要来源。
      </p>

      <GoogLeNetPipelineDiagram />
      <p className={styles.caption}>
        ↑ GoogLeNet 整网:stem + 9 个 Inception block + GAP + 单层 FC,2 个 aux head 训练时挂在 4a/4d,推理时丢弃。
      </p>

      <HeadParamCompareChart />
      <p className={styles.caption}>
        ↑ VGG-16 的 fc6 一层就占整网 138M 参数的 74%,GoogLeNet 用 GAP 把这部分压到约 1M。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={INCEPTION_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>
            三件套协同 — 缺一不可
          </h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={INCEPTION_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              5M 参数拿下 ImageNet 冠军
            </div>
            <ModelParamCompareChart />
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: "var(--space-2) 0 0", lineHeight: 1.6 }}>
              GoogLeNet 整网仅 5M 参数,比 VGG-16 的 138M 少 28 倍、比 AlexNet 的 60M 少 12 倍 ——
              多分支 + 1×1 瓶颈 + GAP 三件套缺一不可,任何一个抽掉,"参数效率"这条独立优化轴都不成立。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
