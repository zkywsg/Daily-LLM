interface Props {
  t: number;
  onTimeChange: (v: number) => void;
  T: number;
  epsilonNoise: number;
  onEpsilonNoiseChange: (v: number) => void;
}

export function ReverseControls({
  t,
  onTimeChange,
  T,
  epsilonNoise,
  onEpsilonNoiseChange,
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
          <span>reverse 起点 t (网络要从这一步推回 t-1)</span>
          <strong>{t}</strong>
        </label>
        <input
          type="range"
          min={1}
          max={T - 1}
          step={1}
          value={t}
          onChange={(e) => onTimeChange(parseInt(e.target.value, 10))}
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
          <span>ε_θ 预测误差 δ (0 = 完美)</span>
          <strong>{epsilonNoise.toFixed(2)}</strong>
        </label>
        <input
          type="range"
          min={0}
          max={1.5}
          step={0.05}
          value={epsilonNoise}
          onChange={(e) => onEpsilonNoiseChange(parseFloat(e.target.value))}
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
          训练好的 U-Net 让 δ 很小,reverse 才能逐步收敛回有效图像。
          δ=0 时 μ_{"ₜ₋₁"} 几乎跟 x_0 重合(完美去噪)。
        </div>
      </div>
    </div>
  );
}
