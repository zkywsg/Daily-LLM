interface Props {
  hops: number;
  onChange: (h: number) => void;
}

export function LayerToggleWidget({ hops, onChange }: Props) {
  return (
    <div style={{ display: "flex", gap: 8, marginBottom: "var(--space-3)" }}>
      {[1, 2].map((h) => (
        <button
          key={h}
          type="button"
          onClick={() => onChange(h)}
          style={{
            padding: "4px 14px", borderRadius: "var(--radius-sm)",
            border: `1px solid ${hops === h ? "#ec4899" : "var(--border)"}`,
            background: hops === h ? "#ec4899" : "var(--bg-surface)",
            color: hops === h ? "#fff" : "var(--ink-secondary)",
            cursor: "pointer", fontSize: "var(--fs-sm)",
          }}
        >
          L = {h} 层
        </button>
      ))}
    </div>
  );
}
