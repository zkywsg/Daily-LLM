import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GPT1_SOURCE_PATH } from "../lib/prose";
import { AutoregressiveDemo } from "../widgets/AutoregressiveDemo";
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

export function PretrainStage({ mechanism2Prose }: Props) {
  const [step, setStep] = useState(3);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制二:自回归预训练 — 用海量无标注文本
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        预训练目标是标准语言模型最大似然,k=512 上下文窗口。关键数据选择是 BookCorpus
        而非 Wikipedia — 7000 本未发表小说提供更丰富的长程依赖,迫使模型学"看到远处语义结构"的能力。
        8×P6000×30天,117M 参数,今天单卡几小时就能训完。
      </p>

      <AutoregressiveDemo step={step} />
      <p className={styles.caption}>
        ↑ 拖动切换看预测第几个词。绿色是可见上文,灰色 "?" 是被 causal mask 遮住的未来词。
      </p>
      <input type="range" min={0} max={6} step={1} value={step}
             onChange={(e) => setStep(parseInt(e.target.value))}
             style={{ width: "100%", marginTop: 8 }} />
      <div style={{ display: "flex", gap: 4, marginTop: 8, flexWrap: "wrap" }}>
        {[0, 1, 2, 3, 4, 5, 6].map((s) => (
          <button key={s} type="button" onClick={() => setStep(s)} style={btnStyle(step === s)}>第 {s + 1} 词</button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism2Prose} sourcePath={GPT1_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              为什么用 BookCorpus 而非 Wikipedia?
            </div>
            <table style={{ width: "100%", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>数据源</th>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>长程依赖</th>
                  <th style={{ textAlign: "left", padding: "6px 8px", fontWeight: 600 }}>规模</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <td style={{ padding: "8px" }}>BookCorpus(选用)</td>
                  <td style={{ padding: "8px" }}>强 — 伏笔隔几页才回应</td>
                  <td style={{ padding: "8px" }}>800M token</td>
                </tr>
                <tr>
                  <td style={{ padding: "8px" }}>Wikipedia</td>
                  <td style={{ padding: "8px" }}>弱 — 文章短、独立</td>
                  <td style={{ padding: "8px" }}>~2.5B token</td>
                </tr>
              </tbody>
            </table>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 10, lineHeight: 1.5, fontStyle: "italic" }}>
              "用数据形态驱动模型能力" 的思想后被 GPT-2/3 推到极致(WebText / Common Crawl)。
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
