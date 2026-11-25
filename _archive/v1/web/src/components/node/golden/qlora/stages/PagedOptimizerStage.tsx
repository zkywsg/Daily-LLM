import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { QLORA_SOURCE_PATH } from "../lib/prose";
import { PagedOptimizerChart } from "../widgets/PagedOptimizerChart";
import styles from "./Stage.module.css";

interface Props {
  mechanism3Prose: string;
  synergyProse: string;
}

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function PagedOptimizerStage({ mechanism3Prose, synergyProse }: Props) {
  const [showPaging, setShowPaging] = useState(true);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Paged Optimizer
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        训练时 AdamW 优化器状态(m, v)显存占用大,且梯度检查点重算 / batch 边界
        会造成显存 spike。QLoRA 用 NVIDIA Unified Memory 把优化器状态"分页"—
        大部分留在 CPU 内存,GPU 训练时按需 paging 到 GPU,把显存 spike 削平在
        GPU 容量以内。
      </p>

      <PagedOptimizerChart showPaging={showPaging} />
      <p className={styles.caption}>
        ↑ 切换看有无 paging 时显存曲线的差异 — 无 paging 时红点标记的 spike 会超出 GPU 容量导致 OOM。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setShowPaging(true)} style={btnStyle(showPaging)}>Paged Optimizer</button>
        <button type="button" onClick={() => setShowPaging(false)} style={btnStyle(!showPaging)}>无 Paging</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={QLORA_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={QLORA_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              抽掉任何一件,65B 都跑不动
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>只有 NF4</strong>:base 33GB 但 optimizer 仍 15GB,总 65GB,单卡 48GB 装不下</li>
              <li><strong>只有 Double Quantization(用 INT4)</strong>:精度损失大,Guanaco-65B 质量崩</li>
              <li><strong>只有 Paged Optimizer</strong>:base 仍 fp16 130GB,不管 optimizer 多省也跑不动 65B</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
