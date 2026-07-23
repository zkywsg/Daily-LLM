import { NODES } from "../lib/data";

interface Props {
  i: number;
  j: number;
  onSelectI: (n: number) => void;
  onSelectJ: (n: number) => void;
}

export function NodePairSelectorWidget({ i, j, onSelectI, onSelectJ }: Props) {
  const row = (label: string, current: number, onSelect: (n: number) => void) => (
    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: "var(--space-2)" }}>
      <span style={{ width: 60, fontSize: "var(--fs-sm)", color: "var(--ink-secondary)" }}>{label}</span>
      {NODES.map((n) => (
        <button
          key={n} type="button" onClick={() => onSelect(n)} aria-pressed={n === current}
          style={{
            width: 28, height: 28, borderRadius: "var(--radius-sm)",
            border: `1px solid ${n === current ? "#ec4899" : "var(--border)"}`,
            background: n === current ? "#ec4899" : "var(--bg-surface)",
            color: n === current ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-xs)",
          }}
        >
          {n}
        </button>
      ))}
    </div>
  );

  return (
    <div style={{ marginBottom: "var(--space-4)" }}>
      {row("节点 i", i, onSelectI)}
      {row("节点 j", j, onSelectJ)}
    </div>
  );
}
