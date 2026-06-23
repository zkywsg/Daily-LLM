import type { Schedule } from "../lib/math";

interface Props {
  T: number;
  onTChange: (v: number) => void;
  t: number;
  onTimeChange: (v: number) => void;
  schedule: Schedule;
  onScheduleChange: (s: Schedule) => void;
}

const btnStyle = (active: boolean): React.CSSProperties => ({
  padding: "4px 12px",
  fontSize: "var(--fs-sm)",
  borderRadius: "var(--radius-sm)",
  border: `1px solid ${active ? "var(--accent-link)" : "var(--border)"}`,
  background: active ? "var(--accent-link)" : "var(--bg-surface)",
  color: active ? "var(--bg-surface)" : "var(--ink-secondary)",
  cursor: "pointer",
});

export function ForwardControls({
  T,
  onTChange,
  t,
  onTimeChange,
  schedule,
  onScheduleChange,
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
        <div
          style={{
            fontSize: "var(--fs-xs)",
            color: "var(--ink-muted)",
            marginBottom: 4,
            textTransform: "uppercase",
            letterSpacing: "0.05em",
          }}
        >
          噪声 schedule
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          <button
            type="button"
            onClick={() => onScheduleChange("linear")}
            style={btnStyle(schedule === "linear")}
          >
            linear (原版 DDPM)
          </button>
          <button
            type="button"
            onClick={() => onScheduleChange("cosine")}
            style={btnStyle(schedule === "cosine")}
          >
            cosine (Improved DDPM)
          </button>
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
          <span>当前步 t</span>
          <strong>
            {t} / {T - 1}
          </strong>
        </label>
        <input
          type="range"
          min={0}
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
          <span>总步数 T</span>
          <strong>{T}</strong>
        </label>
        <input
          type="range"
          min={50}
          max={1000}
          step={50}
          value={T}
          onChange={(e) => onTChange(parseInt(e.target.value, 10))}
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
          原版 T=1000(图像够细)。T 小则每步噪声大、ᾱ_t 衰减快;
          T 大则反向链路长 → 采样慢但质量更稳。
        </div>
      </div>
    </div>
  );
}
