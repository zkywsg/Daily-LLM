import { useState } from "react";
import { TOY_WAVEFORM, downsample } from "../lib/data";

const W = 680;
const H = 260;

export function DownsampleWidget() {
  const [numLayers, setNumLayers] = useState(0);
  const frames = downsample(TOY_WAVEFORM, numLayers);
  const frameRateHz = Math.round(16000 / Math.pow(2, numLayers));

  const barW = Math.max(2, (W - 60) / frames.length - 1);
  const maxAbs = Math.max(...frames.map((v) => Math.abs(v)), 0.1);

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        CNN 下采样层数 = {numLayers}(帧率 ≈ {frameRateHz}Hz,共 {frames.length} 帧)
        <input type="range" min={0} max={5} value={numLayers} onChange={(e) => setNumLayers(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`下采样 ${numLayers} 层后共 ${frames.length} 帧,帧率约 ${frameRateHz}Hz`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          下采样后的帧序列(每帧一根柱子)
        </text>
        <line x1={30} y1={H / 2} x2={W - 30} y2={H / 2} stroke="var(--border)" />
        {frames.map((v, i) => {
          const x = 30 + i * ((W - 60) / frames.length);
          const h = Math.min((Math.abs(v) / maxAbs) * (H / 2 - 30), H / 2 - 30);
          const y = v >= 0 ? H / 2 - h : H / 2;
          return <rect key={i} x={x} y={y} width={barW} height={h} fill="#fb7185" />;
        })}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        层数越多,帧数越少、每帧覆盖的时间跨度越长——16kHz 原始波形经过约 7 层卷积后,帧率会压缩到真实 wav2vec 2.0 使用的约 50Hz。
      </p>
    </div>
  );
}
