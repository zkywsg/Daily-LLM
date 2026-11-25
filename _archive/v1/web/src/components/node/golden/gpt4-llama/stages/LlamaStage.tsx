import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GPT4_LLAMA_SOURCE_PATH } from "../lib/prose";
import { LlamaModelSizesDiagram } from "../widgets/LlamaModelSizesDiagram";
import { OverTrainRatioChart } from "../widgets/OverTrainRatioChart";
import { LLAMA_MODELS } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
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

export function LlamaStage({ mechanism2Prose }: Props) {
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [ratioIdx, setRatioIdx] = useState(-1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:LLaMA — 开源现代 LLM 的样板(可复现 + over-train 小模型)
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        LLaMA-1 给社区一个可复现、高质量、架构现代化的基础模型 — 所有训练细节、
        架构选择、数据组成全部公开。7B 模型在 1T token 上训练,数据/参数比达
        143:1,远超 Chinchilla 的 20:1 — 这是"故意 over-train 小模型"的实证:
        loss 略高但模型小 10 倍,推理时省 10× 算力,部署成本完全胜出。
      </p>

      <LlamaModelSizesDiagram selectedIdx={selectedIdx} />
      <p className={styles.caption}>
        ↑ 点按钮查看 LLaMA-1 四档模型的具体规格与特点。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        {LLAMA_MODELS.map((m, i) => (
          <button key={m.name} type="button" onClick={() => setSelectedIdx(i)} style={btnStyle(selectedIdx === i)}>{m.name}</button>
        ))}
      </div>

      <div style={{ marginTop: "var(--space-8)" }}>
        <OverTrainRatioChart highlightIdx={ratioIdx} />
        <p className={styles.caption}>
          ↑ Chinchilla 最优比 vs LLaMA 系列持续推高的 over-train 比例,点按钮聚焦某一行。
        </p>
        <div style={{ display: "flex", gap: 4, marginTop: 8, flexWrap: "wrap" }}>
          <button type="button" onClick={() => setRatioIdx(-1)} style={btnStyle(ratioIdx === -1)}>全部</button>
          {["Chinchilla", "LLaMA-1", "LLaMA-2", "LLaMA-3"].map((n, i) => (
            <button key={i} type="button" onClick={() => setRatioIdx(i)} style={btnStyle(ratioIdx === i)}>{n}</button>
          ))}
        </div>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={GPT4_LLAMA_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              为什么"推理优先"划算?
            </div>
            <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", margin: 0, lineHeight: 1.6 }}>
              训练是一次性成本,推理是长期反复发生的成本。小模型多训一些 token
              换来更低的推理算力,这笔账在大规模部署场景下远比"训练时省一点"
              划算 — LLaMA-2 推到 286:1、LLaMA-3 8B 推到 1875:1,持续验证这一路线。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
