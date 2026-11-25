interface Props {
  width?: number;
  height?: number;
}

/**
 * DenseNet-BC 的两处压缩:
 * (1) Bottleneck —— 每个 H_ℓ 内部先用 1×1 把输入压到 4k,再 3×3 出 k 通道
 * (2) Compression —— transition layer 用 1×1 把输出通道数减半(θ=0.5)
 */
export function BottleneckCompressionDiagram({ width = 620, height = 300 }: Props) {
  const rowY1 = 90;
  const rowY2 = 210;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ maxWidth: "100%", height: "auto", display: "block" }}
      role="img"
      aria-label="DenseNet-BC Bottleneck 与 Compression 结构图"
    >
      <text x={width / 2} y={22} textAnchor="middle" fontSize={15} fontWeight={600} fill="var(--ink-primary)">
        DenseNet-BC:Bottleneck(层内)+ Compression(block 之间)
      </text>

      {/* Row 1: Bottleneck 内部 H_ℓ */}
      <text x={40} y={rowY1 - 20} fontSize={12} fontWeight={600} fill="var(--ink-secondary)">Bottleneck(单个 H_ℓ 内部)</text>

      <rect x={40} y={rowY1} width={90} height={40} rx={6} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.5} />
      <text x={85} y={rowY1 + 24} textAnchor="middle" fontSize={11} fill="#1e3a8a">输入 in_c</text>

      <line x1={130} y1={rowY1 + 20} x2={172} y2={rowY1 + 20} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#dn-bc-arrow)" />

      <rect x={172} y={rowY1} width={100} height={40} rx={6} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.5} />
      <text x={222} y={rowY1 + 18} textAnchor="middle" fontSize={11} fill="#9d174d">1×1 Conv</text>
      <text x={222} y={rowY1 + 32} textAnchor="middle" fontSize={10} fill="#9d174d">压到 4k</text>

      <line x1={272} y1={rowY1 + 20} x2={314} y2={rowY1 + 20} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#dn-bc-arrow)" />

      <rect x={314} y={rowY1} width={100} height={40} rx={6} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.5} />
      <text x={364} y={rowY1 + 18} textAnchor="middle" fontSize={11} fill="#9d174d">3×3 Conv</text>
      <text x={364} y={rowY1 + 32} textAnchor="middle" fontSize={10} fill="#9d174d">出 k 通道</text>

      <line x1={414} y1={rowY1 + 20} x2={456} y2={rowY1 + 20} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#dn-bc-arrow)" />

      <rect x={456} y={rowY1} width={120} height={40} rx={6} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.5} />
      <text x={516} y={rowY1 + 18} textAnchor="middle" fontSize={11} fill="#065f46">concat([x,out])</text>
      <text x={516} y={rowY1 + 32} textAnchor="middle" fontSize={10} fill="#065f46">in_c + k</text>

      {/* Row 2: Transition layer Compression */}
      <text x={40} y={rowY2 - 20} fontSize={12} fontWeight={600} fill="var(--ink-secondary)">Compression(transition layer,block 之间)</text>

      <rect x={40} y={rowY2} width={100} height={40} rx={6} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.5} />
      <text x={90} y={rowY2 + 18} textAnchor="middle" fontSize={11} fill="#1e3a8a">Dense Block</text>
      <text x={90} y={rowY2 + 32} textAnchor="middle" fontSize={10} fill="#1e3a8a">输出 C 通道</text>

      <line x1={140} y1={rowY2 + 20} x2={182} y2={rowY2 + 20} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#dn-bc-arrow)" />

      <rect x={182} y={rowY2} width={110} height={40} rx={6} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.5} />
      <text x={237} y={rowY2 + 18} textAnchor="middle" fontSize={11} fill="#92400e">1×1 Conv</text>
      <text x={237} y={rowY2 + 32} textAnchor="middle" fontSize={10} fill="#92400e">θ=0.5 减半</text>

      <line x1={292} y1={rowY2 + 20} x2={334} y2={rowY2 + 20} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#dn-bc-arrow)" />

      <rect x={334} y={rowY2} width={100} height={40} rx={6} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.5} />
      <text x={384} y={rowY2 + 18} textAnchor="middle" fontSize={11} fill="#92400e">2×2 AvgPool</text>
      <text x={384} y={rowY2 + 32} textAnchor="middle" fontSize={10} fill="#92400e">下采样</text>

      <line x1={434} y1={rowY2 + 20} x2={476} y2={rowY2 + 20} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#dn-bc-arrow)" />

      <rect x={476} y={rowY2} width={100} height={40} rx={6} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.5} />
      <text x={526} y={rowY2 + 18} textAnchor="middle" fontSize={11} fill="#065f46">下一 Block</text>
      <text x={526} y={rowY2 + 32} textAnchor="middle" fontSize={10} fill="#065f46">输入 0.5C 通道</text>

      <defs>
        <marker id="dn-bc-arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#9ca3af" />
        </marker>
      </defs>

      <text x={width / 2} y={height - 16} textAnchor="middle" fontSize={12} fontStyle="italic" fill="var(--ink-muted)">
        两处压缩合起来把 DenseNet-121 参数量从 20M+ 砍到 7.0M
      </text>
    </svg>
  );
}
