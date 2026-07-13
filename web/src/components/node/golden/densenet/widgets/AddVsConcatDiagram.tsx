interface Props {
  mode: "add" | "concat";
  width?: number;
  height?: number;
}

/**
 * 并排展示 ResNet 的 add(把两路叠加到同一张量,通道数不变,信息混叠)
 * 与 DenseNet 的 concat(两路拼接到通道维,通道数翻倍,信息无损保留)。
 */
export function AddVsConcatDiagram({ mode, width = 560, height = 320 }: Props) {
  const isAdd = mode === "add";
  const centerX = width / 2;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ maxWidth: "100%", height: "auto", display: "block" }}
      role="img"
      aria-label={isAdd ? "ResNet 加法残差连接" : "DenseNet 拼接残差连接"}
    >
      <text
        x={centerX}
        y={28}
        textAnchor="middle"
        fontSize={16}
        fontWeight={600}
        fill="var(--ink-primary)"
      >
        {isAdd ? "ResNet: y = F(x) + x" : "DenseNet: y = [x, F(x)]"}
      </text>

      {/* x 输入(浅层特征,黄) */}
      <g>
        <rect x={70} y={90} width={90} height={40} rx={6} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.5} />
        <text x={115} y={115} textAnchor="middle" fontSize={13} fill="#92400e">x (浅层特征)</text>
      </g>

      {/* F(x)(深层特征,粉) */}
      <g>
        <rect x={70} y={170} width={90} height={40} rx={6} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.5} />
        <text x={115} y={195} textAnchor="middle" fontSize={13} fill="#9d174d">F(x) (深层特征)</text>
      </g>

      {/* 箭头汇入合并点 */}
      <line x1={160} y1={110} x2={isAdd ? 260 : 250} y2={isAdd ? 145 : 130} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#dn-arrow)" />
      <line x1={160} y1={190} x2={isAdd ? 260 : 250} y2={isAdd ? 155 : 170} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#dn-arrow)" />

      <defs>
        <marker id="dn-arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#9ca3af" />
        </marker>
      </defs>

      {isAdd ? (
        <>
          {/* Add: 圆圈 ⊕,输出通道数不变(仍是黄绿混色,表示混叠) */}
          <circle cx={290} cy={150} r={24} fill="#fef3c7" stroke="#f59e0b" strokeWidth={2} />
          <text x={290} y={158} textAnchor="middle" fontSize={22} fontWeight={700} fill="#92400e">⊕</text>

          <line x1={314} y1={150} x2={370} y2={150} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#dn-arrow)" />

          <rect x={370} y={130} width={110} height={40} rx={6} fill="#f3f4f6" stroke="#9ca3af" strokeWidth={1.5} />
          <text x={425} y={155} textAnchor="middle" fontSize={12} fill="#4b5563">y, 通道数不变</text>
          <text x={425} y={190} textAnchor="middle" fontSize={11} fill="var(--ink-muted)">同一张量内混叠</text>
          <text x={425} y={206} textAnchor="middle" fontSize={11} fill="var(--ink-muted)">无法区分来源</text>
        </>
      ) : (
        <>
          {/* Concat: 拼接框,输出通道数翻倍,x 与 F(x) 分区保留 */}
          <rect x={250} y={90} width={16} height={40} rx={2} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1} />
          <rect x={250} y={170} width={16} height={40} rx={2} fill="#fce7f3" stroke="#ec4899" strokeWidth={1} />
          <text x={258} y={145} textAnchor="middle" fontSize={16} fill="var(--ink-secondary)">‖</text>

          <line x1={266} y1={150} x2={330} y2={150} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#dn-arrow)" />

          <rect x={330} y={90} width={20} height={40} rx={2} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.5} />
          <rect x={330} y={170} width={20} height={40} rx={2} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.5} />
          <rect x={330} y={90} width={20} height={120} fill="none" stroke="#10b981" strokeWidth={2} rx={2} />

          <rect x={370} y={90} width={110} height={120} rx={6} fill="#ecfdf5" stroke="#10b981" strokeWidth={1.5} />
          <text x={425} y={140} textAnchor="middle" fontSize={12} fill="#065f46">y, 通道数翻倍</text>
          <text x={425} y={158} textAnchor="middle" fontSize={11} fill="#065f46">[x, F(x)]</text>
          <text x={425} y={195} textAnchor="middle" fontSize={11} fill="var(--ink-muted)">两路各自保留</text>
          <text x={425} y={211} textAnchor="middle" fontSize={11} fill="var(--ink-muted)">下游可精确取用</text>
        </>
      )}

      <text x={centerX} y={height - 20} textAnchor="middle" fontSize={12} fontStyle="italic" fill="var(--ink-muted)">
        {isAdd ? "加法 = 信息混叠,不知道数值由哪一层贡献" : "concat = 信息保留,谁想用哪层特征就直接拿"}
      </text>
    </svg>
  );
}
