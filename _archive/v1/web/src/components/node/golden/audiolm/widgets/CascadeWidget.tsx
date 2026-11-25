import { useState } from "react";
import { CASCADE_STAGES } from "../lib/data";

export function CascadeWidget() {
  const [completed, setCompleted] = useState(0);

  return (
    <div>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {CASCADE_STAGES.map((s, i) => {
          const done = i < completed;
          const active = i === completed;
          return (
            <div
              key={s.label}
              style={{
                padding: "var(--space-3)", borderRadius: "var(--radius-md)",
                border: `1px solid ${done ? "#059669" : active ? "#fb7185" : "var(--border)"}`,
                background: done ? "#ecfdf5" : active ? "#fff1f2" : "var(--bg-surface)",
                opacity: i > completed ? 0.5 : 1,
              }}
            >
              <div style={{ fontSize: "var(--fs-sm)", fontWeight: 700, color: done ? "#065f46" : active ? "#9d174d" : "var(--ink-secondary)" }}>
                {done ? "✓ " : ""}{s.label}
              </div>
              <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginTop: 4 }}>{s.detail}</div>
            </div>
          );
        })}
      </div>
      <div style={{ display: "flex", gap: 8, marginTop: "var(--space-4)" }}>
        <button
          type="button" onClick={() => setCompleted((c) => Math.min(c + 1, CASCADE_STAGES.length))}
          disabled={completed >= CASCADE_STAGES.length}
          style={{ padding: "4px 14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: completed >= CASCADE_STAGES.length ? "not-allowed" : "pointer", fontSize: "var(--fs-sm)", opacity: completed >= CASCADE_STAGES.length ? 0.5 : 1 }}
        >
          执行下一阶段
        </button>
        <button
          type="button" onClick={() => setCompleted(0)}
          style={{ padding: "4px 14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: "pointer", fontSize: "var(--fs-sm)" }}
        >
          重置
        </button>
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-2)" }}>
        三个阶段依次级联执行,每个阶段都由独立的 Transformer decoder 训练,后一阶段以前一阶段的输出为条件。
      </p>
    </div>
  );
}
