interface Props {
  value: number;
  onChange: (n: number) => void;
  dModel: number;
}

export function HeadCountSlider({ value, onChange, dModel }: Props) {
  const dk = Math.floor(dModel / value);
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "var(--space-2)",
        padding: "var(--space-3)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-md)",
        background: "var(--bg-surface)",
      }}
    >
      <label
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: "var(--fs-sm)",
          color: "var(--ink-secondary)",
        }}
      >
        <span>head 数 h</span>
        <strong>
          {value} (每 head dk = {dk})
        </strong>
      </label>
      <input
        type="range"
        min={1}
        max={8}
        step={1}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value, 10))}
        style={{ width: "100%" }}
      />
      <div
        style={{
          fontSize: "var(--fs-xs)",
          color: "var(--ink-muted)",
          lineHeight: 1.4,
        }}
      >
        Vaswani 2017 原版 d_model=512、h=8、dk=64。注意算力大致不变:
        h 个 head × dk² ≈ 单个 head × d_model²。
      </div>
    </div>
  );
}
