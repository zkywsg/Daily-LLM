interface Props {
  mode: "concat" | "average";
  onChange: (m: "concat" | "average") => void;
}

export function CombineModeToggleWidget({ mode, onChange }: Props) {
  return (
    <div style={{ display: "flex", gap: 8, marginBottom: "var(--space-3)" }}>
      {(["concat", "average"] as const).map((m) => (
        <button
          key={m} type="button" onClick={() => onChange(m)} aria-pressed={mode === m}
          style={{
            padding: "4px 14px", borderRadius: "var(--radius-sm)",
            border: `1px solid ${mode === m ? "#ec4899" : "var(--border)"}`,
            background: mode === m ? "#ec4899" : "var(--bg-surface)",
            color: mode === m ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-sm)",
          }}
        >
          {m === "concat" ? "concat(中间层)" : "average(输出层)"}
        </button>
      ))}
    </div>
  );
}
