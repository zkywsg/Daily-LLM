import { useState } from "react";
import { CODEBOOK_SIZE } from "../lib/data";

const SEQ_LEN = 10;
// 用 (i*2+1) % CODEBOOK_SIZE(周期 3,而非 i*3+1 的周期 2)构造序列,避免严格
// 交替模式——严格周期 2 的序列下,任意单点 mask 后用左右邻居平均预测必然
// 恒错(两个邻居总是同一类、且总是"对面"那一类)。这里的周期 3 序列保证默认
// mask 的两个位置(3、7)一个预测对、一个预测错,真实展示"有时对有时错"。
const TOKEN_SEQUENCE: number[] = Array.from({ length: SEQ_LEN }, (_, i) => (i * 2 + 1) % CODEBOOK_SIZE);

/** 用左右相邻未 mask 位置的平均(四舍五入)预测被 mask 位置——
 * 这是"用上下文预测"的简化示意,不保证每次都对,和真实掩码预测任务一样。 */
function predictMasked(seq: number[], maskedIdx: number): number {
  const left = seq[(maskedIdx - 1 + seq.length) % seq.length];
  const right = seq[(maskedIdx + 1) % seq.length];
  return Math.round((left + right) / 2);
}

export function MaskedPredictWidget() {
  const [maskedSet, setMaskedSet] = useState<Set<number>>(new Set([3, 7]));

  const toggle = (i: number) => {
    setMaskedSet((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  };

  return (
    <div>
      <div style={{ display: "flex", gap: 4, marginBottom: "var(--space-4)", flexWrap: "wrap" }}>
        {TOKEN_SEQUENCE.map((v, i) => {
          const isMasked = maskedSet.has(i);
          const predicted = isMasked ? predictMasked(TOKEN_SEQUENCE, i) : null;
          const correct = predicted === v;
          return (
            <button
              key={i} type="button" onClick={() => toggle(i)} aria-pressed={isMasked}
              style={{
                width: 56, height: 56, borderRadius: "var(--radius-md)",
                border: `2px solid ${isMasked ? (correct ? "#059669" : "#dc2626") : "var(--border)"}`,
                background: isMasked ? "var(--bg-surface)" : "var(--bg-subtle)",
                display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                cursor: "pointer", fontSize: "var(--fs-xs)",
              }}
            >
              {isMasked ? (
                <>
                  <span style={{ fontSize: 9, color: "var(--ink-muted)" }}>预测</span>
                  <span style={{ fontWeight: 700, color: correct ? "#059669" : "#dc2626" }}>{predicted}</span>
                </>
              ) : (
                <span style={{ fontWeight: 700, color: "var(--ink-primary)" }}>{v}</span>
              )}
            </button>
          );
        })}
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        点击方块切换是否 mask。被 mask 的位置(粗边框)显示 Transformer 用左右上下文预测出的量化目标——绿色表示预测正确,红色表示预测错误(和真实模型一样,上下文预测不保证每次都对)。
      </p>
    </div>
  );
}
