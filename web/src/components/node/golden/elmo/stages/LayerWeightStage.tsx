import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { ELMO_SOURCE_PATH } from "../lib/prose";
import { LayerWeightsChart } from "../widgets/LayerWeightsChart";
import { LayerVectorSpace } from "../widgets/LayerVectorSpace";
import { LAYER_WEIGHTS } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  mechanism2Prose: string;
}

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 10px",
  fontSize: "var(--fs-xs)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function LayerWeightStage({ mechanism2Prose }: Props) {
  const [task, setTask] = useState<string | null>(null);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:多层 hidden 加权组合 — 不同任务偏好不同层
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        ELMo 不固定用哪一层,而是对每个下游任务学一组 softmax 权重 s_j + 全局 γ scale。
        POS Tagging 偏 char-CNN(语法浅)· WSD 偏 LSTM-L2(语义深)· 大多数任务平均使用各层。
        这是 ELMo 最具洞察力的实证 — LSTM 不同层学到不同抽象层次,后来被 BERT/Tenney 2019 反复验证。
      </p>

      <LayerWeightsChart highlightTask={task} />
      <p className={styles.caption}>
        ↑ 5 个任务的层权重堆叠。POS Tagging 49% 权重在 char-CNN;
        WSD 45% 权重在 LSTM-L2。选中一个任务对比。
      </p>
      <div style={{ display: "flex", gap: 4, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setTask(null)} style={btnStyle(task === null)}>全部</button>
        {LAYER_WEIGHTS.map((r) => (
          <button key={r.task} type="button" onClick={() => setTask(r.task)} style={btnStyle(task === r.task)}>{r.task}</button>
        ))}
      </div>

      <LayerVectorSpace />
      <p className={styles.caption}>
        ↑ 3 层 hidden state 上 "river bank" vs "money bank" 的距离。
        char-CNN 距离小(只看字面) · L1 开始分开 · L2 完全分开(语义完全区分)。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={ELMO_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              加权公式
            </div>
            <pre style={{ fontSize: 11, lineHeight: 1.5, margin: 0, color: "var(--ink-primary)", overflow: "auto" }}>{`ELMo_k^{task} =
    γ^{task} · Σ_{j=0}^L
        s_j^{task} · h_k^{LM, j}

- s_j: softmax-normalized 权重
        (每任务学一组,∑ s_j = 1)
- γ:   全局 scaling factor
- h_k^{LM, j}: 第 j 层第 k 位置 hidden`}</pre>
          </div>

          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              为什么这个设计有远见?
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li>底层 = 语法 / 顶层 = 语义 · 这个分层特性 BERT 也有</li>
              <li>Tenney 2019 <em>BERT Rediscovers Classical NLP Pipeline</em> 直接借用 ELMo 的方法论</li>
              <li>今天所有 probing 研究都在 ELMo 之上继承</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
