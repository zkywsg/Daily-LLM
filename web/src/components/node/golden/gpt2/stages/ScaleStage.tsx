import { useState } from "react";
import { MarkdownRenderer } from "../../../MarkdownRenderer";
import { GPT2_SOURCE_PATH } from "../lib/prose";
import { ScaleComparisonBars } from "../widgets/ScaleComparisonBars";
import { ZeroShotScalingCurves } from "../widgets/ZeroShotScalingCurves";
import { ZS_SCALING } from "../lib/data";
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

export function ScaleStage({ intuitionProse, mechanism1Prose }: Props) {
  const [hIdx, setHIdx] = useState(4); // 默认高亮 GPT-2 XL
  const [taskIdx, setTaskIdx] = useState(-1); // -1 显示全部

  return (
    <div>
      <h2 style={{ fontSize: "var(--fs-3xl)", marginBottom: "var(--space-2)" }}>
        机制一:1.5B + 40GB WebText — 10× scale 引出涌现
      </h2>
      <p style={{ fontSize: "var(--fs-md)", color: "var(--ink-secondary)", marginBottom: "var(--space-8)" }}>
        GPT-2 架构相对 GPT-1 几乎不变,核心变量是规模 — 参数 117M → 1.5B (13×),
        数据 BookCorpus 5GB → WebText 40GB (8×)。WebText 用 Reddit karma ≥ 3 过滤,
        天然多样 + 高质量 + 多领域。"高质量数据 + 大模型" 是 GPT-2 涌现的物理基础。
      </p>

      <ScaleComparisonBars highlightIdx={hIdx} />
      <p className={styles.caption}>
        ↑ log 横轴,从 GPT-1 117M 到 GPT-3 175B 跨 3 个数量级。
        粉色四条 = GPT-2 系列,绿色 GPT-2 XL 是当时(2019)发表的最大公开 LM。
      </p>

      <div className={styles.grid} style={{ marginTop: "var(--space-8)" }}>
        <div>
          <h3 style={{ fontSize: "var(--fs-xl)", marginBottom: "var(--space-4)" }}>直觉</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={intuitionProse} sourcePath={GPT2_SOURCE_PATH} />
          </div>

          <h3 style={{ fontSize: "var(--fs-xl)", marginTop: "var(--space-8)", marginBottom: "var(--space-4)" }}>机制</h3>
          <div style={{ fontFamily: "var(--font-serif)", lineHeight: 1.7 }}>
            <MarkdownRenderer markdown={mechanism1Prose} sourcePath={GPT2_SOURCE_PATH} />
          </div>
        </div>

        <div className={styles.stickyPanel}>
          <ZeroShotScalingCurves selectedTask={taskIdx} />
          <p className={styles.caption}>
            ↑ 4 个 GPT-2 size 上的 zero-shot 性能。LAMBADA / WikiText 平滑提升;
            WMT 翻译是"~1B 才稳定 work"的涌现型 — 117M 几乎为 0,1.5B 拿到 11.5 BLEU。
          </p>
          <div style={{ marginTop: "var(--space-4)", padding: "var(--space-3)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
            <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              聚焦任务
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <button type="button" onClick={() => setTaskIdx(-1)} style={btnStyle(taskIdx === -1)}>全部</button>
              {ZS_SCALING.map((t, i) => (
                <button key={i} type="button" onClick={() => setTaskIdx(i)} style={btnStyle(taskIdx === i)}>
                  {t.task} {t.isEmergent && "·涌现"}
                </button>
              ))}
            </div>

            <div style={{ marginTop: 12, paddingTop: 10, borderTop: "1px dashed var(--border)" }}>
              <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                左图高亮
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                <button type="button" onClick={() => setHIdx(-1)} style={btnStyle(hIdx === -1)}>全部</button>
                <button type="button" onClick={() => setHIdx(0)} style={btnStyle(hIdx === 0)}>GPT-1</button>
                <button type="button" onClick={() => setHIdx(4)} style={btnStyle(hIdx === 4)}>GPT-2 XL</button>
                <button type="button" onClick={() => setHIdx(5)} style={btnStyle(hIdx === 5)}>GPT-3</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
