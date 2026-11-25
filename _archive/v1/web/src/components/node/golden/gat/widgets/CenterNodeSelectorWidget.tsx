import { NODES, rawNeighbors } from "../lib/data";

interface Props {
  selected: number;
  onSelect: (n: number) => void;
}

export function CenterNodeSelectorWidget({ selected, onSelect }: Props) {
  return (
    <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-3)" }}>
      {NODES.filter((n) => rawNeighbors(n).length > 0).map((n) => (
        <button
          key={n} type="button" onClick={() => onSelect(n)} aria-pressed={n === selected}
          style={{
            width: 30, height: 30, borderRadius: "var(--radius-sm)",
            border: `1px solid ${n === selected ? "#ec4899" : "var(--border)"}`,
            background: n === selected ? "#ec4899" : "var(--bg-surface)",
            color: n === selected ? "#fff" : "var(--ink-secondary)", cursor: "pointer",
          }}
        >
          {n}
        </button>
      ))}
    </div>
  );
}
