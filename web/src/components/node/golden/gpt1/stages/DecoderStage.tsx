import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GPT1_SOURCE_PATH } from "../lib/prose";
import { ParadigmShift } from "../widgets/ParadigmShift";
import { DecoderBlockDiagram } from "../widgets/DecoderBlockDiagram";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

type Hl = "attn" | "ffn" | "gelu" | "tying" | null;

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function DecoderStage({ intuitionProse, mechanism1Prose }: Props) {
  const [side, setSide] = useState<"old" | "new" | "both">("both");
  const [hl, setHl] = useState<Hl>(null);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Decoder-only Transformer — masked self-attn + GELU + Weight Tying
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        2018 年初 NLP 是"每任务一个专门模型"的范式,标注数据稀缺 + 模型不通用 + 浪费海量无标注文本。
        GPT-1 用原版 Transformer 的 decoder 部分(去掉 cross-attention)构建 backbone,
        12 层 d_model=768 h=12,117M 参数,配 GELU 激活 + weight tying 两个沿用至今的工程细节。
      </p>

      <ParadigmShift side={side} />
      <p className={styles.caption}>
        ↑ 旧范式每任务从头训专门网络;GPT-1 预训练一次 + 微调适配所有任务,
        只换最后一层 head。点按钮聚焦其中一边。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setSide("old")} style={btnStyle(side === "old")}>聚焦 旧范式</button>
        <button type="button" onClick={() => setSide("new")} style={btnStyle(side === "new")}>聚焦 GPT-1</button>
        <button type="button" onClick={() => setSide("both")} style={btnStyle(side === "both")}>对比</button>
      </div>

      <DecoderBlockDiagram highlight={hl} />
      <p className={styles.caption}>
        ↑ Post-LN + masked self-attn + GELU FFN,右侧展示 weight tying(LM head 与 token embedding 共享权重)。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setHl(null)} style={btnStyle(hl === null)}>全部</button>
        <button type="button" onClick={() => setHl("attn")} style={btnStyle(hl === "attn")}>Masked Attn</button>
        <button type="button" onClick={() => setHl("ffn")} style={btnStyle(hl === "ffn")}>FFN</button>
        <button type="button" onClick={() => setHl("gelu")} style={btnStyle(hl === "gelu")}>GELU</button>
        <button type="button" onClick={() => setHl("tying")} style={btnStyle(hl === "tying")}>Weight Tying</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={GPT1_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={GPT1_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              为什么选 decoder-only?
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li>抛弃 ULMFiT / ELMo 的 LSTM,选 Transformer decoder</li>
              <li>并行训练 + 长依赖建模 + 可 scale</li>
              <li>2018 年是非主流选择(BERT 双向更亮眼)</li>
              <li>GPT-2/3 后被全面验证,2023+ LLaMA/Mistral/GPT-4 全部 decoder-only</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
