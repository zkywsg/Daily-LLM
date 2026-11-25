interface Props {
  step: number;
  onChange: (s: number) => void;
}

export function TrainStepSlider({ step, onChange }: Props) {
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
        <span>训练进度</span>
        <strong>{step === 0 ? "step 0(挂上 adapter 瞬间)" : `${step}%`}</strong>
      </label>
      <input
        type="range"
        min={0}
        max={100}
        step={5}
        value={step}
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
        把 slider 拖到 0:B 全 0,ΔW=0,模型行为完全等于 W₀。
        往右拖 B 渐填充,ΔW 渐生效。这就是 LoRA 工业可挂的关键
        —— 任意时刻"加"和"撤"adapter 都不会立即把模型搞坏。
      </div>
    </div>
  );
}
