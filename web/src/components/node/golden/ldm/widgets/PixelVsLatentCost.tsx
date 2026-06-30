const W = 700;
const H = 340;

interface Props {
  resolution: 256 | 512 | 1024;
}

// 比较 pixel 空间 vs latent 空间 (f=8) 的 token / 显存 / 单步成本
function calcCost(res: number) {
  const pixelTokens = res * res * 3;
  const latentTokens = (res / 8) * (res / 8) * 4;
  return { pixelTokens, latentTokens, compressionRatio: pixelTokens / latentTokens };
}

export function PixelVsLatentCost({ resolution }: Props) {
  const c = calcCost(resolution);
  const PAD = 40;
  const ROW1_Y = 70;
  const ROW2_Y = 200;
  const ROW_H = 70;

  const maxBar = W - PAD * 2 - 200;
  const pxBar = maxBar; // pixel always = full
  const latBar = Math.max(maxBar * (c.latentTokens / c.pixelTokens), 12);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Pixel vs latent compute cost">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Pixel 空间 vs Latent 空间(f=8 压缩) — {resolution}×{resolution} 图像
      </text>

      {/* pixel row */}
      <text x={PAD} y={ROW1_Y + 18} fontSize={12} fontWeight={700} fill="#831843">Pixel</text>
      <text x={PAD} y={ROW1_Y + 36} fontSize={10} fill="#6b7280">{resolution}×{resolution}×3</text>
      <rect x={PAD + 80} y={ROW1_Y} width={pxBar} height={ROW_H} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.6} rx={3} />
      <text x={PAD + 80 + pxBar / 2} y={ROW1_Y + 30} textAnchor="middle" fontSize={13} fontWeight={700} fill="#831843">
        {(c.pixelTokens / 1000).toFixed(0)}K tokens
      </text>
      <text x={PAD + 80 + pxBar / 2} y={ROW1_Y + 50} textAnchor="middle" fontSize={11} fill="#6b7280">
        U-Net 每步要算 {(c.pixelTokens / 1000).toFixed(0)}K 个值 · 1000 step
      </text>

      {/* latent row */}
      <text x={PAD} y={ROW2_Y + 18} fontSize={12} fontWeight={700} fill="#065f46">Latent</text>
      <text x={PAD} y={ROW2_Y + 36} fontSize={10} fill="#6b7280">{resolution / 8}×{resolution / 8}×4</text>
      <rect x={PAD + 80} y={ROW2_Y} width={latBar} height={ROW_H} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.6} rx={3} />
      <text x={PAD + 80 + Math.max(latBar / 2, 80)} y={ROW2_Y + 30} fontSize={13} fontWeight={700} fill="#065f46" textAnchor={latBar < 120 ? "start" : "middle"}>
        {(c.latentTokens / 1000).toFixed(1)}K tokens
      </text>
      <text x={PAD + 80 + Math.max(latBar / 2, 80)} y={ROW2_Y + 50} fontSize={11} fill="#6b7280" textAnchor={latBar < 120 ? "start" : "middle"}>
        ({c.compressionRatio.toFixed(0)}× 压缩)
      </text>

      <text x={W / 2} y={H - 24} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--ink-primary)">
        显存降 ≈ {c.compressionRatio.toFixed(0)}× → V100 14G ➜ 2G 单卡可训
      </text>
      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        Diffusion 数学不变,只是搬到一个 49× 更小的空间
      </text>
    </svg>
  );
}
