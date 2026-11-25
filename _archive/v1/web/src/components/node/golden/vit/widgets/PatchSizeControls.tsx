interface Props {
  patchSize: number;
  onPatchSizeChange: (p: number) => void;
}

const OPTIONS = [
  { v: 1, label: "1×1 patch (196 token)" },
  { v: 2, label: "2×2 patch (49 token)" },
  { v: 7, label: "7×7 patch (4 token)" },
  { v: 14, label: "14×14 (1 token)" },
];

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function PatchSizeControls({ patchSize, onPatchSizeChange }: Props) {
  return (
    <div
      style={{
        padding: "var(--space-3)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-md)",
        background: "var(--bg-surface)",
      }}
    >
      <div
        style={{
          fontSize: "var(--fs-xs)",
          color: "var(--ink-muted)",
          marginBottom: 6,
          textTransform: "uppercase",
          letterSpacing: "0.05em",
        }}
      >
        patch 大小(14×14 demo image)
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {OPTIONS.map((o) => (
          <button key={o.v} type="button" onClick={() => onPatchSizeChange(o.v)} style={btnStyle(patchSize === o.v)}>
            {o.label}
          </button>
        ))}
      </div>
      <div
        style={{
          fontSize: "var(--fs-xs)",
          color: "var(--ink-muted)",
          marginTop: 8,
          lineHeight: 1.4,
        }}
      >
        真 ViT-B/16:224×224 image · 16×16 patch · 196 token + 1 CLS。
        patch 越小细粒度越高,但 token 数 ↑² → attention 算力爆炸。
      </div>
    </div>
  );
}
