import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { ROBERTA_SOURCE_PATH } from "../lib/prose";
import { NspCompareChart } from "../widgets/NspCompareChart";
import { NSP_FORMAT_COMPARE } from "../lib/data";
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

export function NspStage({ mechanism2Prose }: Props) {
  const [idx, setIdx] = useState(-1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:去掉 NSP + 单序列输入
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        NSP(Next Sentence Prediction)是 BERT 的第二个预训练任务,Devlin 团队设计
        它是希望帮助句子级下游任务。但 RoBERTa 的对照实验发现:去 NSP 反而略好,
        而且改成单序列输入比双句子还好 — NSP 任务太简单、句对输入还浪费了一半上下文。
      </p>

      <NspCompareChart highlightIdx={idx} />
      <p className={styles.caption}>
        ↑ 三种输入格式对比,点按钮聚焦某一行 — 单序列无 NSP 在 MNLI / SQuAD 上都最好。
      </p>
      <div style={{ display: "flex", gap: 4, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setIdx(-1)} style={btnStyle(idx === -1)}>全部</button>
        {NSP_FORMAT_COMPARE.map((f, i) => (
          <button key={f.format} type="button" onClick={() => setIdx(i)} style={btnStyle(idx === i)}>{f.format}</button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={ROBERTA_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px dashed var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-canvas)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, fontWeight: 600, textTransform: "uppercase" }}>
              为什么去 NSP 反而更好?
            </div>
            <ul style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", paddingLeft: 16, lineHeight: 1.6, margin: 0 }}>
              <li><strong>NSP 太简单</strong> — 随机负样本主题差异巨大,模型靠浅层信号就能区分</li>
              <li><strong>句对浪费上下文</strong> — 512 token 被分成两段,单序列能看完整长依赖</li>
              <li><strong>数据收集成本高</strong> — NSP 需要标记句子边界,扩展到 web 数据麻烦</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
