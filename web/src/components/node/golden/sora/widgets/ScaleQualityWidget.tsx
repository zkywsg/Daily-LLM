import { useState } from "react";
import { SCALE_PRESETS, scaleToQuality } from "../lib/data";

const W = 500;
const H = 260;

export function ScaleQualityWidget() {
  const [scaleIdx, setScaleIdx] = useState(2);
  const scale = SCALE_PRESETS[scaleIdx];
  const quality = scaleToQuality(scale);

  const toX = (s: number) => 40 + (s / 16) * (W - 80);
  const toY = (q: number) => H - 40 - q * (H - 80);
  const curvePoints = Array.from({ length: 50 }, (_, i) => (i / 49) * 16);
  const path = curvePoints.map((s, i) => `${i === 0 ? "M" : "L"} ${toX(s)} ${toY(scaleToQuality(s))}`).join(" ");

  return (
    <div>
      <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-3)", flexWrap: "wrap" }}>
        {SCALE_PRESETS.map((s, i) => (
          <button
            key={s} type="button" onClick={() => setScaleIdx(i)} aria-pressed={i === scaleIdx}
            style={{
              padding: "4px 10px", borderRadius: "var(--radius-sm)",
              border: `1px solid ${i === scaleIdx ? "#d946ef" : "var(--border)"}`,
              background: i === scaleIdx ? "#d946ef" : "var(--bg-surface)",
              color: i === scaleIdx ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-xs)",
            }}
          >
            {s}×
          </button>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`模型规模 ${scale} 倍时的生成质量示意`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          DiT 规模化:模型规模 vs 生成质量(示意曲线)
        </text>
        <line x1={40} y1={H - 40} x2={W - 40} y2={H - 40} stroke="var(--border)" />
        <path d={path} fill="none" stroke="#d946ef" strokeWidth={2} />
        <circle cx={toX(scale)} cy={toY(quality)} r={6} fill="#9d174d" />
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        与 DiT 论文一致的规模化规律:算力/参数量越大,质量持续提升但边际收益递减,没有观察到饱和天花板。
      </p>
    </div>
  );
}
