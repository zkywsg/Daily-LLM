const W = 640;
const H = 280;

export function FrozenVsTrainableDiagram() {
  const layerCount = 4; // 示意:12 层压缩展示为 4 层 + 省略号
  const layerH = 48;
  const layerGap = 10;
  const startY = 40;
  const blockX = 120;
  const blockW = 260;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="冻结 BERT 权重 vs 可训练 Adapter 模块">
      <text x={W / 2} y={16} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--ink-primary)">
        冻结 base(灰,🔒)+ 只训 Adapter(粉,可训)
      </text>

      {Array.from({ length: layerCount }).map((_, i) => {
        const y = startY + i * (layerH + layerGap);
        return (
          <g key={i}>
            <rect x={blockX} y={y} width={blockW * 0.62} height={layerH} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1.2} rx={6} />
            <text x={blockX + (blockW * 0.62) / 2} y={y + 20} textAnchor="middle" fontSize={9} fontWeight={600} fill="#6b7280">
              BERT Layer {i + 1}
            </text>
            <text x={blockX + (blockW * 0.62) / 2} y={y + 34} textAnchor="middle" fontSize={8} fill="#9ca3af">
              🔒 冻结(Attention + FFN)
            </text>

            <rect x={blockX + blockW * 0.68} y={y} width={blockW * 0.32} height={layerH} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.8} rx={6} />
            <text x={blockX + blockW * 0.68 + (blockW * 0.32) / 2} y={y + 20} textAnchor="middle" fontSize={9} fontWeight={700} fill="#be185d">
              Adapter₁, Adapter₂
            </text>
            <text x={blockX + blockW * 0.68 + (blockW * 0.32) / 2} y={y + 34} textAnchor="middle" fontSize={8} fill="#be185d">
              可训 + LayerNorm
            </text>
          </g>
        );
      })}

      <text x={blockX + blockW / 2} y={startY + layerCount * (layerH + layerGap) + 4} textAnchor="middle" fontSize={14} fill="var(--ink-muted)">
        ⋮ (共 12 层)
      </text>

      <g transform={`translate(${blockX}, ${startY + layerCount * (layerH + layerGap) + 24})`}>
        <rect x={0} y={0} width={blockW * 0.62} height={26} fill="#f3f4f6" stroke="#9ca3af" rx={5} />
        <text x={(blockW * 0.62) / 2} y={17} textAnchor="middle" fontSize={9} fontWeight={700} fill="#6b7280">
          99% 参数 · 保留通用知识
        </text>
        <rect x={blockW * 0.68} y={0} width={blockW * 0.32} height={26} fill="#fce7f3" stroke="#ec4899" rx={5} />
        <text x={blockW * 0.68 + (blockW * 0.32) / 2} y={17} textAnchor="middle" fontSize={9} fontWeight={700} fill="#be185d">
          &lt;1% 参数
        </text>
      </g>
    </svg>
  );
}
