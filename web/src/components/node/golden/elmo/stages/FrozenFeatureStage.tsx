import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { ELMO_SOURCE_PATH } from "../lib/prose";
import { FrozenVsFineTune } from "../widgets/FrozenVsFineTune";
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

export function FrozenFeatureStage({ mechanism3Prose, synergyProse }: Props) {
  const [mode, setMode] = useState<"elmo" | "bert">("elmo");

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制三:Frozen Feature 拼接 — 不修改 ELMo 参数
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        ELMo 不替换原 word embedding,而是作为额外 feature 拼接到下游 input:
        <code> concat(glove_emb, elmo_emb) </code>。下游模型(典型 BiLSTM-CRF)在拼接后 input 上训练,
        ELMo 参数完全冻结(只学下游 + 每任务的 s_j / γ 权重)。8 个月后 BERT 用端到端 fine-tune 替代这套。
      </p>

      <FrozenVsFineTune mode={mode} />
      <p className={styles.caption}>
        ↑ 切换看 ELMo frozen vs BERT fine-tune 两种范式对比。
        ELMo 训练快 · 多任务共享;BERT 表达力强 · 部署成本高。
      </p>
      <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
        <button type="button" onClick={() => setMode("elmo")} style={btnStyle(mode === "elmo")}>ELMo(frozen)</button>
        <button type="button" onClick={() => setMode("bert")} style={btnStyle(mode === "bert")}>BERT(fine-tune)</button>
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism3Prose} sourcePath={ELMO_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>三件套协同</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={synergyProse} sourcePath={ELMO_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              使用代码
            </div>
            <pre style={{ fontSize: 10, lineHeight: 1.5, margin: 0, color: "var(--ink-primary)", overflow: "auto" }}>{`class NER_with_ELMo(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.glove = nn.Embedding(V, 300)     # 静态
        self.elmo = Elmo(opts, weights,
                          1, dropout=0.5)   # frozen 🔒
        self.lstm = nn.LSTM(300 + 1024, 200,
                            bidirectional=True)
        self.classifier = nn.Linear(400, num_tags)

    def forward(self, word_ids, char_ids):
        g = self.glove(word_ids)               # 300
        e = self.elmo(char_ids)["elmo_representations"][0]  # 1024
        x = torch.cat([g, e], dim=-1)          # 1324
        x, _ = self.lstm(x)
        return self.classifier(x)`}</pre>
          </div>

          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              ELMo → BERT 演进
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li>LSTM → Transformer(更并行)</li>
              <li>两个单向拼接 → 真双向 masked LM</li>
              <li>frozen feature → 端到端 fine-tune</li>
              <li>94M → 340M(BERT-large)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
