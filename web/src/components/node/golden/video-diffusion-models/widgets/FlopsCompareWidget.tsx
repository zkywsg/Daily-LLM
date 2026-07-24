import { useState } from "react";
import { flops3D, flopsFactorized } from "../lib/data";

const W = 680;
const H = 300;
const RESOLUTION = 64;
const FRAMES = 16;
const CHANNELS = 64;

export function FlopsCompareWidget() {
  const [kernel, setKernel] = useState(3);

  const f3d = flops3D(RESOLUTION, RESOLUTION, FRAMES, kernel, CHANNELS);
  const ffact = flopsFactorized(RESOLUTION, RESOLUTION, FRAMES, kernel, CHANNELS);
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
        卷积核大小 k = {kernel}
        <input type="range" min={1} max={7} step={2} value={kernel} onChange={(e) => setKernel(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="3D 卷积与时空分解卷积的算力对比">
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          FLOPs 量级对比(16 帧,64×64,卷积核 {kernel}×{kernel}×{kernel},估算值)
        </text>
        <line x1={30} y1={H - 50} x2={W - 30} y2={H - 50} stroke="var(--border)" />
        {bar(150, f3d, "完整 3D 卷积", "#9ca3af")}
        {bar(400, ffact, "2D+1D 分解", "#d946ef")}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        卷积核越大,3D 卷积相对 2D+1D 分解的算力劣势越明显(比值 ≈ k²/(k+1));分辨率/帧数对两者的影响是同倍数缩放的,不改变这个比例关系。
      </p>
    </div>
  );
}
