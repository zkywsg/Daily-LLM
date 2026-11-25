import { useState } from "react";
import { PRESET_CONFIGS, patchCounts } from "../lib/data";

// 同一个模型不改架构,直接切换不同长宽比/时长预设,展示 patch
// 数量如何自适应变化 —— 不需要为每种分辨率单独训练/裁剪。

export function NativeResolutionWidget() {
  const [configIdx, setConfigIdx] = useState(0);
  const patchSize = 2;

  return (
    <div>
      <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-4)", flexWrap: "wrap" }}>
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
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={{ textAlign: "left", fontSize: "var(--fs-xs)", color: "var(--ink-muted)" }}>配置</th>
            <th style={{ textAlign: "center", fontSize: "var(--fs-xs)", color: "var(--ink-muted)" }}>帧×高×宽</th>
            <th style={{ textAlign: "center", fontSize: "var(--fs-xs)", color: "var(--ink-muted)" }}>patch 数</th>
          </tr>
        </thead>
        <tbody>
          {PRESET_CONFIGS.map((c, i) => {
            const { total } = patchCounts(c, patchSize);
            const active = i === configIdx;
            return (
              <tr key={c.label} style={{ background: active ? "#fce7f3" : "transparent" }}>
                <td style={{ padding: "4px 8px", fontSize: "var(--fs-sm)", fontWeight: active ? 700 : 400, color: active ? "#9d174d" : undefined }}>{c.label}</td>
                <td style={{ padding: "4px 8px", textAlign: "center", fontSize: "var(--fs-sm)", color: active ? "#9d174d" : undefined }}>{c.frames}×{c.height}×{c.width}</td>
                <td style={{ padding: "4px 8px", textAlign: "center", fontSize: "var(--fs-sm)", color: active ? "#9d174d" : undefined }}>{total}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-3)" }}>
        同一套模型架构,不需要为每种长宽比/时长单独裁剪或重训 —— patch 序列长度自动适配输入尺寸。
      </p>
    </div>
  );
}
