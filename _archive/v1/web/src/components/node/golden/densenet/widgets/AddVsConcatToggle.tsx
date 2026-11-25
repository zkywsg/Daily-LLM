interface Props {
  value: "add" | "concat"; // "add" = ResNet y = F(x) + x, "concat" = DenseNet [x, F(x)]
  onChange: (mode: "add" | "concat") => void;
}

export function AddVsConcatToggle({ value, onChange }: Props) {
  return (
    <div
      style={{
        display: "inline-flex",
        background: "var(--bg-subtle)",
        borderRadius: "var(--radius-full)",
        padding: "var(--space-1)",
        margin: "var(--space-4) 0",
      }}
    >
      <button
        onClick={() => onChange("add")}
        style={{
          padding: "var(--space-2) var(--space-4)",
          borderRadius: "var(--radius-full)",
          fontSize: "var(--fs-sm)",
          fontWeight: 500,
          color: value === "add" ? "var(--ink-primary)" : "var(--ink-secondary)",
          background: value === "add" ? "var(--bg-surface)" : "transparent",
          boxShadow: value === "add" ? "var(--shadow-sm)" : "none",
          fontStyle: "italic",
        }}
      >
        ResNet: F(x) + x (add)
      </button>
      <button
        onClick={() => onChange("concat")}
        style={{
          padding: "var(--space-2) var(--space-4)",
          borderRadius: "var(--radius-full)",
          fontSize: "var(--fs-sm)",
          fontWeight: 500,
          color: value === "concat" ? "var(--ink-primary)" : "var(--ink-secondary)",
          background: value === "concat" ? "var(--bg-surface)" : "transparent",
          boxShadow: value === "concat" ? "var(--shadow-sm)" : "none",
          fontStyle: "italic",
        }}
      >
        DenseNet: [x, F(x)] (concat)
      </button>
    </div>
  );
}
