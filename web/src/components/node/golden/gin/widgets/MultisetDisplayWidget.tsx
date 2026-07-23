interface Props {
  label: string;
  values: number[];
  color: string;
}

export function MultisetDisplayWidget({ label, values, color }: Props) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: "var(--space-3)" }}>
      <span style={{ width: 90, fontSize: "var(--fs-sm)", fontWeight: 600 }}>{label}</span>
      <div style={{ display: "flex", gap: 4 }}>
        {values.map((v, idx) => (
          <div
            key={idx}
            style={{
              width: 28, height: 28, borderRadius: "50%", background: color, color: "#fff",
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: "var(--fs-xs)", fontWeight: 700,
            }}
          >
            {v}
          </div>
        ))}
      </div>
    </div>
  );
}
