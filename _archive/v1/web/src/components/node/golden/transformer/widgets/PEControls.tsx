interface Props {
  nPos: number;
  onNPosChange: (n: number) => void;
  dModel: number;
  onDModelChange: (d: number) => void;
  highlightDim: number | null;
  onHighlightDimChange: (d: number | null) => void;
}

export function PEControls({
  nPos,
  onNPosChange,
  dModel,
  onDModelChange,
  highlightDim,
  onHighlightDimChange,
}: Props) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "var(--space-3)",
        padding: "var(--space-3)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-md)",
        background: "var(--bg-surface)",
      }}
    >
      <div>
        <label
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: "var(--fs-sm)",
            color: "var(--ink-secondary)",
            marginBottom: 4,
          }}
        >
          <span>序列长度 n_pos</span>
          <strong>{nPos}</strong>
        </label>
        <input
          type="range"
          min={8}
          max={64}
          step={4}
          value={nPos}
          onChange={(e) => onNPosChange(parseInt(e.target.value, 10))}
          style={{ width: "100%" }}
        />
      </div>

      <div>
        <label
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: "var(--fs-sm)",
            color: "var(--ink-secondary)",
            marginBottom: 4,
          }}
        >
          <span>d_model</span>
          <strong>{dModel}</strong>
        </label>
        <input
          type="range"
          min={8}
          max={64}
          step={2}
          value={dModel}
          onChange={(e) => onDModelChange(parseInt(e.target.value, 10))}
          style={{ width: "100%" }}
        />
      </div>

      <div>
        <label
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: "var(--fs-sm)",
            color: "var(--ink-secondary)",
            marginBottom: 4,
          }}
        >
          <span>高亮单维 (空=画前 8 维)</span>
          <strong>{highlightDim ?? "—"}</strong>
        </label>
        <input
          type="range"
          min={-1}
          max={dModel - 1}
          step={1}
          value={highlightDim ?? -1}
          onChange={(e) => {
            const v = parseInt(e.target.value, 10);
            onHighlightDimChange(v < 0 ? null : v);
          }}
          style={{ width: "100%" }}
        />
        <div
          style={{
            fontSize: "var(--fs-xs)",
            color: "var(--ink-muted)",
            marginTop: 4,
            lineHeight: 1.4,
          }}
        >
          低维(0, 1)波长短 → 编码"几步以内"的相对位置;高维波长指数级
          变长 → 编码"段落以上"的粗粒度。
        </div>
      </div>
    </div>
  );
}
