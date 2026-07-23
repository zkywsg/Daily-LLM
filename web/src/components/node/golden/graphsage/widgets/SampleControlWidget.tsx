import { useState } from "react";
import { NODES, fullNeighbors, sampleNeighbors } from "../lib/data";
import { NeighborSetWidget } from "./NeighborSetWidget";

// 自带状态的采样控制面板:选中心节点 + k 值 + "重新采样"按钮(换 seed)。

export function SampleControlWidget() {
  const [center, setCenter] = useState(0);
  const [k, setK] = useState(3);
  const [seed, setSeed] = useState(1);

  const all = fullNeighbors(center);
  const sampled = sampleNeighbors(center, k, seed);

  return (
    <div>
      <div style={{ display: "flex", gap: 6, marginBottom: "var(--space-3)", flexWrap: "wrap" }}>
        {NODES.filter((n) => fullNeighbors(n).length > 0).map((n) => (
          <button
            key={n}
            type="button"
            onClick={() => setCenter(n)}
            aria-pressed={n === center}
            style={{
              width: 30, height: 30, borderRadius: "var(--radius-sm)",
              border: `1px solid ${n === center ? "#ec4899" : "var(--border)"}`,
              background: n === center ? "#ec4899" : "var(--bg-surface)",
              color: n === center ? "#fff" : "var(--ink-secondary)", cursor: "pointer",
            }}
          >
            {n}
          </button>
        ))}
      </div>
      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: "var(--space-4)" }}>
        <label style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)" }}>
          采样数 k = {k}
          <input type="range" min={1} max={Math.max(all.length, 1)} value={k} onChange={(e) => setK(Number(e.target.value))} style={{ marginLeft: 8 }} />
        </label>
        <button type="button" onClick={() => setSeed((s) => s + 1)} style={{ padding: "4px 12px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: "pointer", fontSize: "var(--fs-sm)" }}>
          重新采样
        </button>
      </div>
      <NeighborSetWidget center={center} sampled={sampled} />
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-2)" }}>
        节点 {center} 共有 {all.length} 个邻居,本次采样到 {sampled.length} 个:{sampled.join(", ") || "无"}
      </p>
    </div>
  );
}
