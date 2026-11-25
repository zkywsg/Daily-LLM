import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { FLAMINGO_SOURCE_PATH } from "../lib/prose";
import { FlamingoArchitecture } from "../widgets/FlamingoArchitecture";
import { LlmScaleIclChart } from "../widgets/LlmScaleIclChart";
import { LLM_SCALE_ICL } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

type Hl = "vision" | "perceiver" | "llm" | "xattn" | null;

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function FrozenLlmStage({ intuitionProse, mechanism1Prose }: Props) {
  const [hl, setHl] = useState<Hl>(null);
  const [llmIdx, setLlmIdx] = useState(-1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:冻结 70B LLM + Vision Encoder — 保留 LLM 全部能力
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        GPT-3 的 in-context learning 是 175B 大 LLM 涌现的能力,本质上和模态无关。
        前作 ViLBERT / Florence 从零联合训练,LLM 部分根本起不到这个规模。
        Flamingo 冻结一个 70B Chinchilla LLM,只训视觉适配 + cross-attention 注入层 —
        保留语言能力 + 训练成本可控 + 模块化复用。
      </p>

      <FlamingoArchitecture highlight={hl} />
      <p className={styles.caption}>
        ↑ 冻结视觉编码器 + 冻结 70B LLM,中间用可训练桥接层连接。点按钮聚焦各组件。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setHl(null)} style={btnStyle(hl === null)}>全部</button>
        <button type="button" onClick={() => setHl("vision")} style={btnStyle(hl === "vision")}>Vision Encoder</button>
        <button type="button" onClick={() => setHl("perceiver")} style={btnStyle(hl === "perceiver")}>Perceiver Resampler</button>
        <button type="button" onClick={() => setHl("llm")} style={btnStyle(hl === "llm")}>LLM(冻结)</button>
        <button type="button" onClick={() => setHl("xattn")} style={btnStyle(hl === "xattn")}>Cross-Attn</button>
      </div>

      <LlmScaleIclChart highlightIdx={llmIdx} />
      <p className={styles.caption}>
        ↑ 70B 是 ICL 涌现的关键阈值。BLIP-2 的 11B Flan-T5 4-shot 几乎不提升,
        Chinchilla 70B 从 49.2 涨到 56.3。
      </p>
      <div style={{ display: "flex", gap: 4, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setLlmIdx(-1)} style={btnStyle(llmIdx === -1)}>全部</button>
        {LLM_SCALE_ICL.map((r, i) => (
          <button key={i} type="button" onClick={() => setLlmIdx(i)} style={btnStyle(llmIdx === i)}>{r.name}</button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={FLAMINGO_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={FLAMINGO_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              为什么必须冻结 LLM?
            </div>
            <ol style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 18, lineHeight: 1.7, margin: 0 }}>
              <li>保留 LLM 全部语言能力 — 微调会破坏 zero-shot / ICL / 推理</li>
              <li>训练成本可控 — 70B 全微调几百万美元,冻结只训 ~10B 新层,约 $1M</li>
              <li>模块化复用 — 同一 LLM 可接不同视觉模块做图像/视频/音频扩展</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
}
