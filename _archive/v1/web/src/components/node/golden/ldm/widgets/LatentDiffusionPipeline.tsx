const W = 700;
const H = 360;

interface Props {
  step: number;        // 0..50 current sampling step
  totalSteps: number;
}

// 全 pipeline:
// pixel x → VAE.encode → z_0 → forward(noise) → z_T → reverse(U-Net) → z_0' → VAE.decode → pixel x'
// 中间高亮当前 step

function LatentBlock({ x, y, size, noisiness, label }: { x: number; y: number; size: number; noisiness: number; label: string }) {
  const cells = 6;
  const cellSize = size / cells;
  return (
    <g>
      {Array.from({ length: cells * cells }, (_, i) => {
        const r = Math.floor(i / cells);
        const c = i % cells;
        // Structure when noisiness=0, noise when noisiness=1
        const hue = 200 + ((r + c) * 18) % 80;
        const noise = (Math.sin(i * 12.99) * 43758.5453) % 1;
        const ns = (noise < 0 ? noise + 1 : noise);
        const finalHue = hue * (1 - noisiness) + ns * 360 * noisiness;
        const sat = 40 + (i * 9) % 30;
        const light = 55 - noisiness * 10 + ns * 20 * noisiness;
        return (
          <rect key={i} x={x + c * cellSize} y={y + r * cellSize} width={cellSize} height={cellSize}
            fill={`hsl(${finalHue}, ${sat}%, ${light}%)`} stroke="#fff" strokeWidth={0.4} />
        );
      })}
      <rect x={x} y={y} width={size} height={size} fill="none" stroke="#9ca3af" strokeWidth={1} />
      <text x={x + size / 2} y={y - 4} textAnchor="middle" fontSize={10} fontWeight={600} fill="#374151">{label}</text>
    </g>
  );
}

export function LatentDiffusionPipeline({ step, totalSteps }: Props) {
  const progress = step / totalSteps;  // 0 → pure noise, 1 → clean

  const latSize = 50;
  const PAD_Y = 100;

  // Show 5 latent snapshots: z_T (noise) → z_3T/4 → z_T/2 → z_T/4 → z_0
  const snapshots = [
    { t: totalSteps,       label: "z_T" },
    { t: totalSteps * 3/4, label: "z₃ₜ₋₄" },
    { t: totalSteps / 2,   label: "z_T/2" },
    { t: totalSteps / 4,   label: "z_T/4" },
    { t: 0,                label: "z_0" },
  ];

  const totalW = 5 * latSize + 4 * 60;
  const startX = (W - totalW) / 2;

  // Each snapshot's noisiness depends on whether step has reached it
  // step 0: all noisy; step totalSteps: all clean
  // current step = totalSteps - step (count down)
  const currT = totalSteps - step;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Latent diffusion sampling pipeline">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Latent Diffusion 采样 — step {step} / {totalSteps} (DDIM)
      </text>
      <text x={W / 2} y={42} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        全程在 64²×4 latent 空间内 · 最后一步才 VAE.decode 回像素
      </text>

      {snapshots.map((s, i) => {
        const x = startX + i * (latSize + 60);
        // noisiness based on whether we've already denoised past this t
        const ns = currT < s.t ? 1.0 : currT > s.t ? Math.max(0, (s.t / totalSteps) * 1.0) : (s.t / totalSteps);
        const isCurrent = s.t >= currT - 25 && s.t <= currT + 25;
        return (
          <g key={i}>
            <LatentBlock x={x} y={PAD_Y} size={latSize} noisiness={Math.max(0, ns)} label={s.label} />
            {isCurrent && (
              <rect x={x - 4} y={PAD_Y - 4} width={latSize + 8} height={latSize + 8} fill="none" stroke="#10b981" strokeWidth={2.5} rx={3} />
            )}
            <text x={x + latSize / 2} y={PAD_Y + latSize + 16} textAnchor="middle" fontSize={9} fill="#6b7280">
              t={Math.round(s.t)}
            </text>
            {i < snapshots.length - 1 && (
              <g>
                <line x1={x + latSize + 4} y1={PAD_Y + latSize / 2} x2={x + latSize + 56} y2={PAD_Y + latSize / 2} stroke="#9ca3af" strokeWidth={1.4} markerEnd="url(#ldp-arr)" />
                <text x={x + latSize + 30} y={PAD_Y + latSize / 2 - 6} textAnchor="middle" fontSize={9} fontWeight={600} fill="#10b981">ε_θ</text>
              </g>
            )}
          </g>
        );
      })}
      <defs>
        <marker id="ldp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
        </marker>
      </defs>

      {/* VAE decode arrow at the end */}
      <g transform={`translate(0, ${PAD_Y + latSize + 50})`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={11} fontWeight={700} fill="#374151">
          采样完成后 → VAE.decode(z₀) → 512×512 像素图
        </text>
        <rect x={W / 2 - 50} y={36} width={100} height={50} fill="hsl(150, 50%, 75%)" stroke="#10b981" strokeWidth={1.5} rx={4} />
        <text x={W / 2} y={66} textAnchor="middle" fontSize={11} fontWeight={600} fill="#065f46">image x'</text>
      </g>

      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        ε-prediction / linear schedule / DDIM 采样 — 与 DDPM 完全一致,只是 input shape 改成 4×64×64
      </text>
    </svg>
  );
}
