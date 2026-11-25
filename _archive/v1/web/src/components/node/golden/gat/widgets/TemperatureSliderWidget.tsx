interface Props {
  temperature: number;
  onChange: (t: number) => void;
}

export function TemperatureSliderWidget({ temperature, onChange }: Props) {
  return (
    <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
      注意力锐化程度 = {temperature.toFixed(1)}
      <input
        type="range" min={0.2} max={3} step={0.1} value={temperature}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ display: "block", width: "100%", marginTop: 6 }}
      />
    </label>
  );
}
