import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { ROPE_SOURCE_PATH } from "../lib/prose";
import { QkvOnlyDiagram } from "../widgets/QkvOnlyDiagram";
import { ExtensionMethodsChart } from "../widgets/ExtensionMethodsChart";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

export function EngineeringStage({ mechanism3Prose, synergyProse }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:只对 Q/K 应用 + cos/sin 预计算 — 工程零开销
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        只对 Q 和 K 应用 RoPE,V 不动 — V 承载内容信息,不该被位置污染。
        cos/sin 表只依赖位置和频率,不依赖输入,可以一次预计算并 buffer 缓存。
        完整 RoPE attention(LLaMA 风格)只多约 10 行代码,推理无额外计算开销 —
        这是 RoPE 战胜所有相对位置方案的核心工程优势。
      </p>

      <QkvOnlyDiagram />
      <p className={styles.caption}>
        ↑ RoPE 只作用在 Q/K 分支,V 直接传入最终加权求和。
      </p>

      <ExtensionMethodsChart />
      <p className={styles.caption}>
        ↑ Position Interpolation / NTK-aware / YaRN 都建立在 RoPE 旋转频率结构上 —
        LLaMA 从 2K 训练长度扩到 128K 推理长度。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={ROPE_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={ROPE_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              为什么是"基础设施"而非单纯位置编码
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              绝对 PE / learned PE 上几乎不存在等效的"上下文扩展"方法 —
              RoPE 的旋转数学结构让 PI / NTK / YaRN 这类简单缩放 trick 成为可能,
              这是 2023 年长上下文 LLM 能快速迭代的根本原因。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
