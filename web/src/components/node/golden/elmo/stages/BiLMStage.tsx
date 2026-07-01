import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { ELMO_SOURCE_PATH } from "../lib/prose";
import { BankContextCompare } from "../widgets/BankContextCompare";
import { BiLMDiagram } from "../widgets/BiLMDiagram";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
}

type Hl = "char" | "fwd" | "bwd" | "concat" | null;

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function BiLMStage({ intuitionProse, mechanism1Prose }: Props) {
  const [mode, setMode] = useState<"static" | "elmo">("elmo");
  const [hl, setHl] = useState<Hl>(null);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:Deep biLM 预训练 — 正向 + 反向独立 LSTM
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        用大语料无监督训 2 层双向 LSTM 语言模型 · 正向从左预测下一词,反向从右预测前一词。
        "双向"是两个独立 LSTM 训完拼接,不像 BERT self-attention 真双向(BERT 论文指出这是 ELMo 局限)。
        输入用 char-CNN,让 OOV 词也能编码。
      </p>

      <BankContextCompare mode={mode} />
      <p className={styles.caption}>
        ↑ 静态查表 vs ELMo contextualized 对比。静态 cosine=1.0;
        ELMo 走两条不同 LSTM 路径 → cosine=0.42,清晰区分"河岸"和"银行"。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setMode("static")} style={btnStyle(mode === "static")}>静态词向量</button>
        <button type="button" onClick={() => setMode("elmo")} style={btnStyle(mode === "elmo")}>ELMo</button>
      </div>

      <BiLMDiagram highlight={hl} />
      <p className={styles.caption}>
        ↑ char-CNN → 2 层 biLSTM(粉色 forward · 蓝色 backward,独立参数)→ 每位置 5 个 hidden(char + L1×2 + L2×2)。
        点按钮聚焦各层。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setHl(null)} style={btnStyle(hl === null)}>全部</button>
        <button type="button" onClick={() => setHl("char")} style={btnStyle(hl === "char")}>① char-CNN</button>
        <button type="button" onClick={() => setHl("fwd")} style={btnStyle(hl === "fwd")}>② forward LSTM</button>
        <button type="button" onClick={() => setHl("bwd")} style={btnStyle(hl === "bwd")}>③ backward LSTM</button>
        <button type="button" onClick={() => setHl("concat")} style={btnStyle(hl === "concat")}>④ 输出</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={ELMO_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={ELMO_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              网络结构
            </div>
            <table style={{ width: "100%", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>层</th>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>配置</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <td style={{ padding: "8px" }}>Input</td>
                  <td style={{ padding: "8px" }}>char-CNN(FastText 思想 · 处理 OOV)</td>
                </tr>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <td style={{ padding: "8px" }}>biLSTM L1</td>
                  <td style={{ padding: "8px" }}>4096 hidden + 512 projected · 双向</td>
                </tr>
                <tr>
                  <td style={{ padding: "8px" }}>biLSTM L2</td>
                  <td style={{ padding: "8px" }}>4096 hidden + 512 projected · 双向</td>
                </tr>
              </tbody>
            </table>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              训练数据:1B Word Benchmark(10 亿词)· 94M 参数 · 无监督语言建模目标
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
