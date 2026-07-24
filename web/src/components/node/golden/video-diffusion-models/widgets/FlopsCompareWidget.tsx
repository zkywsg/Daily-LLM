import { useState } from "react";
import { flops3D, flopsFactorized } from "../lib/data";

const W = 680;
const H = 300;

export function FlopsCompareWidget() {
  const [resolution, setResolution] = useState(64);
  const frames = 16, kernel = 3, channels = 64;

  const f3d = flops3D(resolution, resolution, frames, kernel, channels);
  const ffact = flopsFactorized(resolution, resolution, frames, kernel, channels);
  const maxF = Math.max(f3d, ffact);

  const bar = (x: number, val: number, label: string, color: string) => {
    const h = Math.min((val / maxF) * (H - 100), H - 100);
    return (
      <g key={label}>
        <rect x={x} y={H - 50 - h} width={100} height={h} fill={color} rx={3} />
        <text x={x + 50} y={H - 50 - h - 8} textAnchor="middle" fontSize={10} fontWeight={700} fill="var(--ink-primary)">
          {(val / 1e9).toFixed(1)}G
        </text>
        <text x={x + 50} y={H - 30} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">
          {label}
        </text>
      </g>
    );
  };

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        分辨率 = {resolution}×{resolution}
        <input type="range" min={32} max={128} step={16} value={resolution} onChange={(e) => setResolution(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="3D 卷积与时空分解卷积的算力对比">
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          FLOPs 量级对比(16 帧,{resolution}×{resolution},估算值)
        </text>
        <line x1={30} y1={H - 50} x2={W - 30} y2={H - 50} stroke="var(--border)" />
        {bar(150, f3d, "完整 3D 卷积", "#9ca3af")}
        {bar(400, ffact, "2D+1D 分解", "#d946ef")}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        分辨率越高,3D 卷积的算力开销增长越快;2D+1D 分解把空间和时间维度拆开卷积,复用图像领域已经很成熟的 2D 卷积效率。
      </p>
    </div>
  );
}
