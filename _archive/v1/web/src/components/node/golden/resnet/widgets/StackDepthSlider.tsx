interface Props {
  value: number;
  onChange: (depth: number) => void;
}

const STACK_OPTIONS = [2, 6, 20, 50];

export function StackDepthSlider({ value, onChange }: Props) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "var(--space-3)",
        padding: "var(--space-3) var(--space-4)",
        background: "var(--bg-subtle)",
        borderRadius: "var(--radius-md)",
        margin: "var(--space-4) 0",
      }}
    >
      <span
        style={{
          fontSize: "var(--fs-sm)",
          color: "var(--ink-secondary)",
          fontWeight: 500,
        }}
      >
        堆叠块数：
      </span>
      {STACK_OPTIONS.map((d) => (
        <button
          key={d}
          onClick={() => onChange(d)}
          style={{
            padding: "var(--space-1) var(--space-3)",
            borderRadius: "var(--radius-full)",
            fontSize: "var(--fs-sm)",
            fontWeight: 500,
            background:
              value === d ? "var(--accent-link)" : "var(--bg-surface)",
            color:
              value === d ? "var(--bg-surface)" : "var(--ink-secondary)",
            border: `1px solid ${
              value === d ? "var(--accent-link)" : "var(--border)"
            }`,
            transition: "all var(--dur-fast) var(--ease-out)",
          }}
        >
          {d}
        </button>
      ))}
    </div>
  );
}
