const W = 700;
const H = 360;

interface Props {
  patchSize: 2 | 4 | 8;
}

// 32x32 latent grid → patch_size 切块 → token 序列 → 投影成 d 维 → token 表示
export function PatchifyPipeline({ patchSize }: Props) {
  const latentSize = 32;
  const grid = latentSize / patchSize; // 16 / 8 / 4
  const numTokens = grid * grid;

  // 左侧 latent 渲染
  const LAT_X = 30;
  const LAT_Y = 70;
  const LAT_SIZE = 180;
  const cellPx = LAT_SIZE / latentSize;

  // 中间 token 序列
  const SEQ_X = 260;
  const SEQ_Y = 130;
  const tokenH = 20;
  const tokenW = 14;
  const maxShown = Math.min(numTokens, 16);
  const seqW = maxShown * (tokenW + 2);

  // 右侧 transformer + 维度
  const TX = 530;
  const TY = 110;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="DiT patchify pipeline">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        DiT Patchify — VAE latent 32×32×4 → patch_size {patchSize} → {numTokens} 个 token
      </text>

      {/* 左:latent grid */}
      <text x={LAT_X + LAT_SIZE / 2} y={60} textAnchor="middle" fontSize={11} fontWeight={600} fill="#6b7280">VAE latent 32×32×4</text>

      {/* 整体边框 */}
      <rect x={LAT_X} y={LAT_Y} width={LAT_SIZE} height={LAT_SIZE} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.5} />

      {/* patch 切分线 */}
      {Array.from({ length: grid - 1 }).map((_, i) => (
        <g key={i}>
          <line x1={LAT_X + (i + 1) * patchSize * cellPx} y1={LAT_Y}
                x2={LAT_X + (i + 1) * patchSize * cellPx} y2={LAT_Y + LAT_SIZE}
                stroke="#ec4899" strokeWidth={1.5} />
          <line x1={LAT_X} y1={LAT_Y + (i + 1) * patchSize * cellPx}
                x2={LAT_X + LAT_SIZE} y2={LAT_Y + (i + 1) * patchSize * cellPx}
                stroke="#ec4899" strokeWidth={1.5} />
        </g>
      ))}

      {/* 标几个 patch 编号 */}
      {[0, 1, grid - 1, grid].slice(0, Math.min(4, numTokens)).map((idx) => {
        const r = Math.floor(idx / grid);
        const c = idx % grid;
        return (
          <text key={idx}
                x={LAT_X + (c + 0.5) * patchSize * cellPx}
                y={LAT_Y + (r + 0.5) * patchSize * cellPx + 3}
                textAnchor="middle" fontSize={9} fontWeight={700} fill="#831843">
            {idx + 1}
          </text>
        );
      })}

      <text x={LAT_X + LAT_SIZE / 2} y={LAT_Y + LAT_SIZE + 18}
            textAnchor="middle" fontSize={10} fill="#6b7280">
        切成 {grid}×{grid} = {numTokens} 个 patch · 每 patch {patchSize}×{patchSize}×4 = {patchSize * patchSize * 4} 维
      </text>

      {/* 箭头 → */}
      <defs>
        <marker id="patch-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
        </marker>
      </defs>
      <line x1={LAT_X + LAT_SIZE + 8} y1={LAT_Y + LAT_SIZE / 2} x2={SEQ_X - 8} y2={LAT_Y + LAT_SIZE / 2}
            stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#patch-arr)" />

      {/* 中:token 序列 */}
      <text x={SEQ_X + seqW / 2} y={SEQ_Y - 20} textAnchor="middle" fontSize={11} fontWeight={600} fill="#6b7280">
        token 序列 {numTokens > maxShown ? `(显示前 ${maxShown})` : `(${numTokens} 个)`}
      </text>
      {Array.from({ length: maxShown }).map((_, i) => (
        <g key={i}>
          <rect x={SEQ_X + i * (tokenW + 2)} y={SEQ_Y} width={tokenW} height={tokenH}
                fill="#fce7f3" stroke="#ec4899" strokeWidth={1} rx={1.5} />
          <text x={SEQ_X + i * (tokenW + 2) + tokenW / 2} y={SEQ_Y + tokenH / 2 + 3}
                textAnchor="middle" fontSize={7} fill="#831843">{i + 1}</text>
        </g>
      ))}
      {numTokens > maxShown && (
        <text x={SEQ_X + seqW + 4} y={SEQ_Y + tokenH / 2 + 3} fontSize={9} fill="#9ca3af">…</text>
      )}

      <text x={SEQ_X + seqW / 2} y={SEQ_Y + tokenH + 14} textAnchor="middle" fontSize={9} fill="#6b7280">
        + 2D sin/cos 位置编码
      </text>

      {/* projection 箭头 */}
      <line x1={SEQ_X + seqW / 2} y1={SEQ_Y + tokenH + 24}
            x2={SEQ_X + seqW / 2} y2={SEQ_Y + tokenH + 50}
            stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#patch-arr)" />
      <text x={SEQ_X + seqW / 2} y={SEQ_Y + tokenH + 40} fontSize={9} fill="#6b7280">Linear → d_model</text>

      {/* d-dim token 行 */}
      <rect x={SEQ_X} y={SEQ_Y + tokenH + 52} width={seqW} height={28}
            fill="#ecfdf5" stroke="#10b981" strokeWidth={1.4} rx={3} />
      <text x={SEQ_X + seqW / 2} y={SEQ_Y + tokenH + 70} textAnchor="middle" fontSize={10} fontWeight={600} fill="#065f46">
        T × d 序列 (T = {numTokens}, d = 1152)
      </text>

      {/* 右:Transformer */}
      <line x1={SEQ_X + seqW + 8} y1={SEQ_Y + tokenH / 2 + 26} x2={TX - 8} y2={SEQ_Y + tokenH / 2 + 26}
            stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#patch-arr)" />

      <rect x={TX} y={TY} width={140} height={120} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.5} rx={4} />
      <text x={TX + 70} y={TY + 28} textAnchor="middle" fontSize={11} fontWeight={700} fill="#92400e">DiT-XL</text>
      <text x={TX + 70} y={TY + 46} textAnchor="middle" fontSize={10} fill="#92400e">28 层 Transformer</text>
      <text x={TX + 70} y={TY + 62} textAnchor="middle" fontSize={10} fill="#92400e">+ adaLN-Zero</text>
      <text x={TX + 70} y={TY + 80} textAnchor="middle" fontSize={10} fill="#6b7280">d=1152 h=16</text>
      <text x={TX + 70} y={TY + 98} textAnchor="middle" fontSize={9} fontStyle="italic" fill="#9ca3af">FLOPs ∝ T²</text>

      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        从输入到输出都是纯 Transformer + 2 个 linear · 没有任何卷积
      </text>
    </svg>
  );
}
