import { useState } from "react";
import { PRESET_CONFIGS, patchCounts } from "../lib/data";

export function SpacetimePatchWidget() {
  const [configIdx, setConfigIdx] = useState(0);
  const [patchSize, setPatchSize] = useState(2);
  const cfg = PRESET_CONFIGS[configIdx];
  const { pt, ph, pw, total } = patchCounts(cfg, patchSize);

  return (
    <div>
      <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-3)", flexWrap: "wrap" }}>
        {PRESET_CONFIGS.map((c, i) => (
          <button
            key={c.label} type="button" onClick={() => setConfigIdx(i)} aria-pressed={i === configIdx}
            style={{
              padding: "4px 10px", borderRadius: "var(--radius-sm)",
              border: `1px solid ${i === configIdx ? "#d946ef" : "var(--border)"}`,
              background: i === configIdx ? "#d946ef" : "var(--bg-surface)",
              color: i === configIdx ? "#fff" : "var(--ink-secondary)", cursor: "pointer", fontSize: "var(--fs-xs)",
            }}
          >
            {c.label}
          </button>
        ))}
      </div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        patch 大小 = {patchSize}
        <input type="range" min={1} max={4} value={patchSize} onChange={(e) => setPatchSize(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <div style={{ display: "flex", gap: 2, flexWrap: "wrap", maxWidth: 300 }}>
        {Array.from({ length: total }, (_, i) => (
          <div key={i} style={{ width: 14, height: 14, background: "#d946ef", opacity: 0.4 + (i % 5) * 0.1, borderRadius: 2 }} />
        ))}
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-3)" }}>
        当前配置({cfg.frames} 帧 × {cfg.height}×{cfg.width})切分成 {pt}×{ph}×{pw} = {total} 个时空 patch。不同长宽比/时长的视频都能统一表示成变长的 patch 序列喂给 DiT。
      </p>
    </div>
  );
}
