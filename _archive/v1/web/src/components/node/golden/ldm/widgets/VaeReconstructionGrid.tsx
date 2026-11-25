const W = 700;
const H = 320;

interface Props {
  vaeKind: "l2" | "perceptual";   // L2 baseline 模糊 / perceptual+GAN 清晰
}

// 模拟 VAE 重建:左原图,中 latent grid,右 decode 后图像
// 用纯色块 + 椭圆模拟图像主体
function Painting({ x, y, size, blurry }: { x: number; y: number; size: number; blurry: boolean }) {
  const cx = x + size / 2;
  const cy = y + size / 2;
  return (
    <g>
      {/* background */}
      <rect x={x} y={y} width={size} height={size} fill="rgb(120,180,140)" />
      {/* "subject" */}
      <ellipse cx={cx} cy={cy + 10} rx={size * 0.22} ry={size * 0.16} fill={blurry ? "rgb(180,130,110)" : "rgb(200,100,80)"} opacity={blurry ? 0.7 : 1} />
      {/* eyes */}
      {!blurry && (
        <>
          <circle cx={cx - 15} cy={cy - 5} r={6} fill="#1f2937" />
          <circle cx={cx + 15} cy={cy - 5} r={6} fill="#1f2937" />
          <path d={`M ${cx - 12} ${cy + 18} Q ${cx} ${cy + 28} ${cx + 12} ${cy + 18}`} fill="none" stroke="#1f2937" strokeWidth={2} />
        </>
      )}
      {blurry && (
        <>
          <circle cx={cx - 15} cy={cy - 5} r={9} fill="#374151" opacity={0.5} />
          <circle cx={cx + 15} cy={cy - 5} r={9} fill="#374151" opacity={0.5} />
        </>
      )}
    </g>
  );
}

export function VaeReconstructionGrid({ vaeKind }: Props) {
  const blurry = vaeKind === "l2";
  const orig_x = 30, orig_y = 60, size = 180;
  const lat_x = 280, lat_y = 60;
  const dec_x = 480, dec_y = 60;

  // 8x8 = 64 latent cells (代表 64×64 latent 简化)
  const cells = 8;
  const cellSize = size / cells;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="VAE reconstruction quality">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        VAE 重建 — {vaeKind === "l2" ? "纯 L2 训练 (模糊)" : "L1 + LPIPS + Adversarial (清晰)"}
      </text>

      <text x={orig_x + size / 2} y={orig_y - 8} textAnchor="middle" fontSize={11} fontWeight={600} fill="#6b7280">原图 512²</text>
      <Painting x={orig_x} y={orig_y} size={size} blurry={false} />

      {/* arrow → */}
      <text x={orig_x + size + 12} y={orig_y + size / 2 - 6} fontSize={11} fontWeight={700} fill="#ec4899">Encode</text>
      <line x1={orig_x + size + 6} y1={orig_y + size / 2} x2={lat_x - 6} y2={lat_y + size / 2} stroke="#ec4899" strokeWidth={1.8} markerEnd="url(#vae-arr)" />
      <defs>
        <marker id="vae-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#ec4899" />
        </marker>
      </defs>

      <text x={lat_x + size / 2} y={lat_y - 8} textAnchor="middle" fontSize={11} fontWeight={600} fill="#6b7280">latent 64²×4</text>
      {/* latent grid as colored cells */}
      {Array.from({ length: cells * cells }, (_, i) => {
        const r = Math.floor(i / cells);
        const c = i % cells;
        // pseudo-color: low-frequency content
        const hue = 200 + ((r + c) * 12) % 80;
        const sat = 40 + (i * 7) % 30;
        return (
          <rect key={i} x={lat_x + c * cellSize} y={lat_y + r * cellSize} width={cellSize} height={cellSize}
            fill={`hsl(${hue}, ${sat}%, 55%)`} stroke="#fff" strokeWidth={0.5} />
        );
      })}

      <text x={lat_x + size + 12} y={lat_y + size / 2 - 6} fontSize={11} fontWeight={700} fill="#10b981">Decode</text>
      <line x1={lat_x + size + 6} y1={lat_y + size / 2} x2={dec_x - 6} y2={dec_y + size / 2} stroke="#10b981" strokeWidth={1.8} markerEnd="url(#vae-arr2)" />
      <defs>
        <marker id="vae-arr2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#10b981" />
        </marker>
      </defs>

      <text x={dec_x + size / 2} y={dec_y - 8} textAnchor="middle" fontSize={11} fontWeight={600} fill="#6b7280">还原 512²</text>
      <Painting x={dec_x} y={dec_y} size={size} blurry={blurry} />

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        {blurry
          ? "L2 取平均 → 高频细节糊掉,在这上面 diffusion 训出来质量直接劣于 DDPM"
          : "perceptual + adversarial 保住细节 → diffusion 在 latent 上质量与 pixel diffusion 持平"}
      </text>
    </svg>
  );
}
