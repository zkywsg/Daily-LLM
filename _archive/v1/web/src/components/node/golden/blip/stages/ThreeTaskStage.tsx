import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { BLIP_SOURCE_PATH } from "../lib/prose";
import { ThreeTaskDiagram } from "../widgets/ThreeTaskDiagram";
import { THREE_TASKS } from "../lib/data";
import styles from "./Stage.module.css";

interface Props {
  intuitionProse: string;
  mechanism1Prose: string;
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

export function ThreeTaskStage({ intuitionProse, mechanism1Prose }: Props) {
  const [idx, setIdx] = useState(-1);

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:BLIP — 三任务联合预训练 + CapFilt 数据自举
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        CLIP 解决了图像和文本对齐,但只能做判别式匹配,不能生成 caption /
        答 VQA。BLIP 用共享 vision encoder + text encoder,同时训练三个任务:
        ITC(对比,同 CLIP)、ITM(匹配,二分类)、LM(生成,causal decoder
        生成 caption)。三任务互补,让模型既能对齐又能生成。
      </p>

      <ThreeTaskDiagram highlightIdx={idx} />
      <p className={styles.caption}>
        ↑ 点按钮聚焦某个任务头,查看它如何从共享 encoder 分支出来。
      </p>
      <div style={{ display: "flex", gap: 4, marginTop: 8, flexWrap: "wrap" }}>
        <button type="button" onClick={() => setIdx(-1)} style={btnStyle(idx === -1)}>全部</button>
        {THREE_TASKS.map((t, i) => (
          <button key={t.name} type="button" onClick={() => setIdx(i)} style={btnStyle(idx === i)}>{t.name}</button>
        ))}
      </div>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={BLIP_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={BLIP_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 8, fontWeight: 600, textTransform: "uppercase" }}>
              三个任务头的分工
            </div>
            {THREE_TASKS.map((t) => (
              <div key={t.name} style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: 8, lineHeight: 1.5 }}>
                <strong>{t.name}</strong>({t.full}):{t.desc}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
