import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { ALBERT_SOURCE_PATH } from "../lib/prose";
import { SopVsNspDiagram } from "../widgets/SopVsNspDiagram";
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

export function SopStage({ mechanism3Prose, synergyProse }: Props) {
  const [mode, setMode] = useState<"nsp" | "sop">("sop");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:NSP → SOP(句子顺序预测)
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        RoBERTa 已经证明 NSP 几乎无用 —— negative sample 是随机不相关的句子,
        模型靠浅层主题信号就能区分。ALBERT 不简单删掉 NSP,而是改成 SOP:
        正负例都是同一对句子,负例只是把顺序调换,模型必须学到真正的句间连贯性。
      </p>

      <SopVsNspDiagram mode={mode} />
      <p className={styles.caption}>
        ↑ 切换看 NSP 与 SOP 的负例构造方式差异 —— SOP 的负例和正例主题完全一样,只是顺序反了。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setMode("sop")} style={btnStyle(mode === "sop")}>SOP(句子顺序预测)</button>
        <button type="button" onClick={() => setMode("nsp")} style={btnStyle(mode === "nsp")}>NSP(下一句预测)</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={ALBERT_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={ALBERT_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              抽掉任何一件,都达不到目标
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>只有因式分解</strong>:embedding 30M → 4M,但 attention+FFN 仍占 75% → 压缩不到位</li>
              <li><strong>只有跨层共享</strong>:attention+FFN 288M → 12M,但 embedding 仍随 H 膨胀(xxlarge 时爆)</li>
              <li><strong>只有 SOP</strong>:不动架构,只涨 0.6–1.0 分,完全达不到"18× 压缩"目标</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
