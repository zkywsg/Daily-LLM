interface Props {
  value: boolean; // true = shortcut ON
  onChange: (showShortcut: boolean) => void;
}

export function HighwayShortcutToggle({ value, onChange }: Props) {
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
        }}
      >
        无 shortcut
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
        }}
      >
        有 shortcut
      </button>
    </div>
  );
}
