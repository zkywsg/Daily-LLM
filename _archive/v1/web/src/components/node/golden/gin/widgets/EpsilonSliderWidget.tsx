interface Props {
  epsilon: number;
  onChange: (e: number) => void;
}

export function EpsilonSliderWidget({ epsilon, onChange }: Props) {
  return (
    <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
      ε = {epsilon.toFixed(2)}
      <input
        type="range" min={0} max={2} step={0.05} value={epsilon}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ display: "block", width: "100%", marginTop: 6 }}
      />
    </label>
  );
}
