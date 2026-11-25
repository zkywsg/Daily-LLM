import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { QLORA_SOURCE_PATH } from "../lib/prose";
import { DoubleQuantDiagram } from "../widgets/DoubleQuantDiagram";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

export function DoubleQuantStage({ mechanism2Prose }: Props) {
  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:Double Quantization
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        NF4 量化时每 64 个值一个 block,需要存一个 fp32 scale factor。256 个 block
        就要 8KB 额外开销。Double Quantization 把这些 scale factor 自身再做一次
        8-bit 量化,进一步省 ~30% 显存 — 量化元数据本身也不是免费的。
      </p>

      <DoubleQuantDiagram />
      <p className={styles.caption}>
        ↑ 每个 block 先算出自己的 fp32 scale factor,再把这些 scale factor 打包做第二次量化。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={QLORA_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              为什么元数据也要量化?
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              65B 模型有海量 block,即使每个 block 只多花几十 bit 的 scale factor,
              累加起来也是数 GB 级别的开销。Double Quantization 把这部分"隐藏成本"
              也压缩,是 NF4 精度不变前提下继续省显存的关键一步。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
