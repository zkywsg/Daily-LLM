import type { LoraConfig } from "../lib/math";

interface Props {
  config: LoraConfig;
  onChange: (c: LoraConfig) => void;
}

export function LoraControls({ config, onChange }: Props) {
  const update = (patch: Partial<LoraConfig>) => onChange({ ...config, ...patch });
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
      <SliderRow
        label="d (输出维度)"
        value={config.d}
        min={64}
        max={4096}
        step={64}
        onChange={(v) => update({ d: v })}
      />
      <SliderRow
        label="k (输入维度)"
        value={config.k}
        min={64}
        max={4096}
        step={64}
        onChange={(v) => update({ k: v })}
      />
      <SliderRow
        label="rank r (LoRA 秩)"
        value={config.r}
        min={1}
        max={64}
        step={1}
        onChange={(v) => update({ r: v })}
        hint={`r 越小越省参数,但学得到的 ΔW 越受秩约束。原论文常用 r=8。`}
      />
      <SliderRow
        label="α (scaling)"
        value={config.alpha}
        min={1}
        max={64}
        step={1}
        onChange={(v) => update({ alpha: v })}
        hint="实际系数 = α/r,一般直接令 α=r 让 scaling=1。"
      />
    </div>
  );
}

function SliderRow({
  label,
  value,
  min,
  max,
  step,
  onChange,
  hint,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  hint?: string;
}) {
  return (
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
        <span>{label}</span>
        <strong>{value}</strong>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value, 10))}
        style={{ width: "100%" }}
      />
      {hint && (
        <div
          style={{
            fontSize: "var(--fs-xs)",
            color: "var(--ink-muted)",
            marginTop: 4,
            lineHeight: 1.4,
          }}
        >
          {hint}
        </div>
      )}
    </div>
  );
}
