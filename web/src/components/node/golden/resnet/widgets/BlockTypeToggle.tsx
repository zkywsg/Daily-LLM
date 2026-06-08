interface Props {
  value: "basic" | "bottleneck";
  onChange: (blockType: "basic" | "bottleneck") => void;
}

export function BlockTypeToggle({ value, onChange }: Props) {
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
        onClick={() => onChange("basic")}
        style={{
          padding: "var(--space-2) var(--space-4)",
          borderRadius: "var(--radius-full)",
          fontSize: "var(--fs-sm)",
          fontWeight: 500,
          color:
            value === "basic" ? "var(--ink-primary)" : "var(--ink-secondary)",
          background: value === "basic" ? "var(--bg-surface)" : "transparent",
          boxShadow: value === "basic" ? "var(--shadow-sm)" : "none",
        }}
      >
        BasicBlock
      </button>
      <button
        onClick={() => onChange("bottleneck")}
        style={{
          padding: "var(--space-2) var(--space-4)",
          borderRadius: "var(--radius-full)",
          fontSize: "var(--fs-sm)",
          fontWeight: 500,
          color:
            value === "bottleneck"
              ? "var(--ink-primary)"
              : "var(--ink-secondary)",
          background:
            value === "bottleneck" ? "var(--bg-surface)" : "transparent",
          boxShadow: value === "bottleneck" ? "var(--shadow-sm)" : "none",
        }}
      >
        Bottleneck
      </button>
    </div>
  );
}
