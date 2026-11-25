interface Props {
  tokens: string[];
  onTokensChange: (t: string[]) => void;
  dModel: number;
  onDModelChange: (d: number) => void;
  scaled: boolean;
  onScaledChange: (s: boolean) => void;
  view: "raw" | "scaled" | "weights";
  onViewChange: (v: "raw" | "scaled" | "weights") => void;
}

const PRESETS: Array<{ label: string; tokens: string[] }> = [
  { label: "The cat sat", tokens: ["The", "cat", "sat", "on", "the", "mat"] },
  { label: "I love NLP", tokens: ["I", "love", "NLP", "so", "much"] },
  { label: "猫坐垫子上", tokens: ["猫", "坐", "在", "垫子", "上"] },
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

export function AttentionControls({
  tokens,
  onTokensChange,
  dModel,
  onDModelChange,
  scaled,
  onScaledChange,
  view,
  onViewChange,
}: Props) {
  const presetActive = PRESETS.findIndex(
    (p) => p.tokens.join("|") === tokens.join("|"),
  );
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
        <div
          style={{
            fontSize: "var(--fs-xs)",
            color: "var(--ink-muted)",
            marginBottom: 4,
            textTransform: "uppercase",
            letterSpacing: "0.05em",
          }}
        >
          句子
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {PRESETS.map((p, i) => (
            <button
              key={p.label}
              type="button"
              onClick={() => onTokensChange(p.tokens)}
              style={btnStyle(i === presetActive)}
            >
              {p.label}
            </button>
          ))}
        </div>
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
          <span>d_model (向量维度)</span>
          <strong>{dModel}</strong>
        </label>
        <input
          type="range"
          min={4}
          max={64}
          step={4}
          value={dModel}
          onChange={(e) => onDModelChange(parseInt(e.target.value, 10))}
          style={{ width: "100%" }}
        />
      </div>

      <div>
        <div
          style={{
            fontSize: "var(--fs-xs)",
            color: "var(--ink-muted)",
            marginBottom: 4,
            textTransform: "uppercase",
            letterSpacing: "0.05em",
          }}
        >
          看哪一步
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          <button type="button" onClick={() => onViewChange("raw")} style={btnStyle(view === "raw")}>
            raw QKᵀ
          </button>
          <button type="button" onClick={() => onViewChange("scaled")} style={btnStyle(view === "scaled")}>
            {scaled ? "÷√dk 后" : "(未 scale)"}
          </button>
          <button type="button" onClick={() => onViewChange("weights")} style={btnStyle(view === "weights")}>
            softmax 权重
          </button>
        </div>
      </div>

      <label
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          fontSize: "var(--fs-sm)",
          color: "var(--ink-secondary)",
          cursor: "pointer",
        }}
      >
        <input
          type="checkbox"
          checked={scaled}
          onChange={(e) => onScaledChange(e.target.checked)}
        />
        <span>
          除以 √dk(关掉看 dk 大时 softmax 饱和成 one-hot 的失败模式)
        </span>
      </label>
    </div>
  );
}
