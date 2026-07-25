import { useState } from "react";
import { ConditionMode, CONDITION_LABELS } from "../lib/data";

export function ConditionToggleWidget() {
  const [textOn, setTextOn] = useState(true);
  const [melodyOn, setMelodyOn] = useState(false);

  const mode: ConditionMode = textOn && melodyOn ? "both" : textOn ? "text" : melodyOn ? "melody" : "none";

  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: "var(--space-4)" }}>
        <button
          type="button" onClick={() => setTextOn((v) => !v)} aria-pressed={textOn}
          style={{
            padding: "6px 16px", borderRadius: "var(--radius-sm)",
            border: `1px solid ${textOn ? "#fb7185" : "var(--border)"}`,
            background: textOn ? "#fb7185" : "var(--bg-surface)",
            color: textOn ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-sm)",
          }}
        >
          文本条件(T5 编码)
        </button>
        <button
          type="button" onClick={() => setMelodyOn((v) => !v)} aria-pressed={melodyOn}
          style={{
            padding: "6px 16px", borderRadius: "var(--radius-sm)",
            border: `1px solid ${melodyOn ? "#fb7185" : "var(--border)"}`,
            background: melodyOn ? "#fb7185" : "var(--bg-surface)",
            color: melodyOn ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-sm)",
          }}
        >
          旋律条件(chromagram)
        </button>
      </div>
      <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
        <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 4 }}>当前生成模式</div>
        <div style={{ fontSize: "var(--fs-lg)", fontWeight: 700, color: "#9d174d" }}>{CONDITION_LABELS[mode]}</div>
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-3)" }}>
        文本条件通过 T5 文本编码器以 cross-attention 方式注入;旋律条件从参考音频提取色度图(音高/和声走向)。两种条件可以单独或组合使用。
      </p>
    </div>
  );
}
