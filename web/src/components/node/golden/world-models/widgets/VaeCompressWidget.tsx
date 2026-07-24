import { useState } from "react";
import { GRID_SIZE, TOY_FRAME, encodeVAE, decodeVAE } from "../lib/data";

// 8x8 toy 帧 → 可调维度的潜向量 → 重建。潜维度越小重建越模糊,
// 直观展示 VAE 压缩的信息损失权衡。

function GridSvg({ values, size = 160 }: { values: number[]; size?: number }) {
  const cell = size / GRID_SIZE;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {values.map((v, i) => {
        const x = (i % GRID_SIZE) * cell;
        const y = Math.floor(i / GRID_SIZE) * cell;
        const g = Math.round(v * 255);
        return <rect key={i} x={x} y={y} width={cell} height={cell} fill={`rgb(${g},${g},${g})`} stroke="var(--bg-canvas)" strokeWidth={0.5} />;
      })}
    </svg>
  );
}

export function VaeCompressWidget() {
  const [latentDim, setLatentDim] = useState(8);
  const z = encodeVAE(TOY_FRAME, latentDim);
  const recon = decodeVAE(z, latentDim);

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        潜向量维度 = {latentDim}
        <input type="range" min={1} max={16} value={latentDim} onChange={(e) => setLatentDim(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <div style={{ display: "flex", gap: "var(--space-6)", alignItems: "center", flexWrap: "wrap" }}>
        <div>
          <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 4 }}>原始帧(8×8)</div>
          <GridSvg values={TOY_FRAME} />
        </div>
        <div style={{ fontSize: "var(--fs-2xl)", color: "var(--ink-muted)" }}>→</div>
        <div>
          <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 4 }}>z(潜向量,{latentDim} 维)</div>
          <div style={{ display: "flex", gap: 2, flexWrap: "wrap", width: 160 }}>
            {z.map((v, i) => (
              <div key={i} title={v.toFixed(2)} style={{ width: 16, height: 16, background: `hsl(${v > 0 ? 200 : 0}, 70%, ${60 - Math.min(Math.abs(v) * 40, 30)}%)`, borderRadius: 2 }} />
            ))}
          </div>
        </div>
        <div style={{ fontSize: "var(--fs-2xl)", color: "var(--ink-muted)" }}>→</div>
        <div>
          <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", marginBottom: 4 }}>重建帧</div>
          <GridSvg values={recon} />
        </div>
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-3)" }}>
        维度越小,重建越模糊(信息损失越大);维度越大,重建越接近原图,但 C 后续要处理的状态空间也越大。
      </p>
    </div>
  );
}
