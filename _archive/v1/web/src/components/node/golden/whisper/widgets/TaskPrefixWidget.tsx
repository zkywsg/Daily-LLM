import { useState } from "react";
import { TASK_PREFIXES } from "../lib/data";

export function TaskPrefixWidget() {
  const [taskIdx, setTaskIdx] = useState(0);
  const task = TASK_PREFIXES[taskIdx];

  return (
    <div>
      <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-4)", flexWrap: "wrap" }}>
        {TASK_PREFIXES.map((t, i) => (
          <button
            key={t.id} type="button" onClick={() => setTaskIdx(i)} aria-pressed={i === taskIdx}
            style={{
              padding: "4px 12px", borderRadius: "var(--radius-sm)",
              border: `1px solid ${i === taskIdx ? "#fb7185" : "var(--border)"}`,
              background: i === taskIdx ? "#fb7185" : "var(--bg-surface)",
              color: i === taskIdx ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-sm)",
            }}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
        <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 4 }}>decoder 输入的第一个特殊 token</div>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: "var(--fs-md)", color: "#9d174d", marginBottom: "var(--space-3)" }}>{task.prefix}</div>
        <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 4 }}>模型输出示例</div>
        <div style={{ fontSize: "var(--fs-md)", color: "var(--ink-primary)" }}>{task.outputExample}</div>
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-3)" }}>
        同一个模型、同一套权重,仅靠切换 decoder 起始的特殊 token,就能在转写/翻译/语言识别/时间戳预测之间切换。
      </p>
    </div>
  );
}
