const W = 700;
const H = 300;

interface Props {
  numPatches: number; // 模拟不同大小的输入特征数
}

export function PerceiverResamplerDiagram({ numPatches }: Props) {
  const startX = 40;
  const y1 = 80;

  const patchW = Math.min(40, (300 - numPatches * 2) / numPatches);
  const patches = Array.from({ length: numPatches });

  const queryY = 200;
  const numQuery = 8; // 展示用缩略,实际 64

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Perceiver Resampler diagram">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Perceiver Resampler — 任意大小视觉特征 → 固定 64 tokens
      </text>

      <text x={startX} y={y1 - 10} fontSize={10} fontWeight={600} fill="#6b7280">
        视觉编码器输出({numPatches} patches,可变大小)
      </text>
      {patches.map((_, i) => (
        <rect key={i} x={startX + i * (patchW + 2)} y={y1} width={patchW} height={30}
              fill="#dbeafe" stroke="#3b82f6" strokeWidth={1} rx={2} />
      ))}

      {/* cross attention arrows down to query */}
      <text x={W / 2} y={155} textAnchor="middle" fontSize={10} fontWeight={600} fill="#831843">
        多层 cross-attention(query 提炼 key/value)
      </text>
      {Array.from({ length: 6 }).map((_, i) => (
        <line key={i} x1={startX + 40 + i * 100} y1={y1 + 30} x2={100 + i * 80} y2={queryY - 4}
              stroke="#ec4899" strokeWidth={1} opacity={0.4} />
      ))}

      <text x={startX} y={queryY - 12} fontSize={10} fontWeight={600} fill="#6b7280">
        输出:64 个可学习 query tokens(固定,仅展示 {numQuery})
      </text>
      {Array.from({ length: numQuery }).map((_, i) => (
        <rect key={i} x={startX + i * 46} y={queryY} width={38} height={30}
              fill="#fce7f3" stroke="#ec4899" strokeWidth={1.4} rx={2} />
      ))}
      <text x={startX + numQuery * 46 + 10} y={queryY + 20} fontSize={11} fill="#9ca3af">... (共 64)</text>

      <rect x={40} y={250} width={W - 80} height={36} rx={4} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.2} opacity={0.6} />
      <text x={W / 2} y={273} textAnchor="middle" fontSize={11} fontWeight={700} fill="#065f46">
        无论输入 {numPatches} 个 patch,输出永远是 [64, hidden_dim]
      </text>
    </svg>
  );
}
