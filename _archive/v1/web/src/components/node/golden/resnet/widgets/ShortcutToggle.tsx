interface Props {
  value: boolean; // true = 显示 F(x)+x（含 shortcut）
  onChange: (showShortcut: boolean) => void;
}

export function ShortcutToggle({ value, onChange }: Props) {
  return (
    <div
      style={{
        display: "inline-flex",
        background: "var(--bg-subtle)",
        borderRadius: "var(--radius-full)",
        padding: "var(--space-1)",
        margin: "var(--space-4) 0",
        marginLeft: "var(--space-3)",
      }}
    >
      <button
        onClick={() => onChange(false)}
        style={{
          padding: "var(--space-2) var(--space-4)",
          borderRadius: "var(--radius-full)",
          fontSize: "var(--fs-sm)",
          fontWeight: 500,
          color: !value ? "var(--ink-primary)" : "var(--ink-secondary)",
          background: !value ? "var(--bg-surface)" : "transparent",
          boxShadow: !value ? "var(--shadow-sm)" : "none",
          fontStyle: "italic",
        }}
      >
        F(x)
      </button>
      <button
        onClick={() => onChange(true)}
        style={{
          padding: "var(--space-2) var(--space-4)",
          borderRadius: "var(--radius-full)",
          fontSize: "var(--fs-sm)",
          fontWeight: 500,
          color: value ? "var(--ink-primary)" : "var(--ink-secondary)",
          background: value ? "var(--bg-surface)" : "transparent",
          boxShadow: value ? "var(--shadow-sm)" : "none",
          fontStyle: "italic",
        }}
      >
        F(x) + x
      </button>
    </div>
  );
}
