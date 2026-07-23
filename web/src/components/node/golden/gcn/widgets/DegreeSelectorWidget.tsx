import { NODES } from "../lib/data";

interface Props {
  selected: number;
  onSelect: (n: number) => void;
}

export function DegreeSelectorWidget({ selected, onSelect }: Props) {
  return (
    <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-3)" }}>
      {NODES.map((n) => (
        <button
          key={n}
          type="button"
          onClick={() => onSelect(n)}
          style={{
            width: 32, height: 32, borderRadius: "var(--radius-sm)",
            border: `1px solid ${n === selected ? "#ec4899" : "var(--border)"}`,
            background: n === selected ? "#ec4899" : "var(--bg-surface)",
            color: n === selected ? "#fff" : "var(--ink-secondary)",
            cursor: "pointer", fontSize: "var(--fs-sm)",
          }}
        >
          {n}
        </button>
      ))}
    </div>
  );
}
