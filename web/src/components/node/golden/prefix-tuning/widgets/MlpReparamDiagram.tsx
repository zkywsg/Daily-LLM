import { MLP_REPARAM } from "../lib/data";

const W = 640;
const H = 380;

export function MlpReparamDiagram() {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="MLP 重参数化:P_small 通过 MLP 扩展到所有层的 K/V prefix">
      <text x={W / 2} y={18} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--ink-primary)">
        MLP 重参数化:小规模 P_small → 展开到所有层 K/V
      </text>

      {/* P_small block */}
      <rect x={40} y={60} width={140} height={70} rx={8} fill="#fef3c7" stroke="#f59e0b" strokeWidth={2} />
      <text x={110} y={88} textAnchor="middle" fontSize={11} fontWeight={700} fill="#92400e">P_small</text>
      <text x={110} y={104} textAnchor="middle" fontSize={9} fill="#92400e">
        m={MLP_REPARAM.m} × d_small={MLP_REPARAM.dSmall}
      </text>
      <text x={110} y={148} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">少量可训练参数</text>

      {/* arrow to MLP */}
      <line x1={182} y1={95} x2={238} y2={95} stroke="#9ca3af" strokeWidth={1.6} markerEnd="url(#arrowhead-mlp)" />

      {/* MLP block */}
      <rect x={240} y={50} width={140} height={90} rx={8} fill="#dbeafe" stroke="#3b82f6" strokeWidth={2} />
      <text x={310} y={80} textAnchor="middle" fontSize={11} fontWeight={700} fill="#1d4ed8">MLP</text>
      <text x={310} y={96} textAnchor="middle" fontSize={9} fill="#1d4ed8">Linear→Tanh→Linear</text>
      <text x={310} y={112} textAnchor="middle" fontSize={8} fill="#1d4ed8">
        out = L × 2 × d
      </text>
      <text x={310} y={125} textAnchor="middle" fontSize={8} fill="var(--ink-muted)">
        (L={MLP_REPARAM.numLayers}, d={MLP_REPARAM.hiddenSize})
      </text>

      {/* arrow to expanded */}
      <line x1={382} y1={95} x2={438} y2={95} stroke="#9ca3af" strokeWidth={1.6} markerEnd="url(#arrowhead-mlp)" />

      {/* Expanded per-layer K/V prefixes */}
      <text x={520} y={45} textAnchor="middle" fontSize={10} fontWeight={700} fill="var(--ink-primary)">
        每层的 K/V prefix
      </text>
      {Array.from({ length: MLP_REPARAM.numLayers }).map((_, i) => {
        const y = 56 + i * 12;
        if (i >= 10) return null; // 12 层示意,压缩显示前 10 + 省略
        return (
          <g key={`layer-${i}`}>
            <rect x={440} y={y} width={70} height={9} rx={2} fill="#fce7f3" stroke="#ec4899" strokeWidth={1} />
            <rect x={514} y={y} width={70} height={9} rx={2} fill="#fce7f3" stroke="#ec4899" strokeWidth={1} />
          </g>
        );
      })}
      <text x={475} y={56 + 10 * 12 + 10} textAnchor="middle" fontSize={8} fill="var(--ink-muted)">P_K per layer</text>
      <text x={549} y={56 + 10 * 12 + 10} textAnchor="middle" fontSize={8} fill="var(--ink-muted)">P_V per layer</text>
      <text x={512} y={56 + 10 * 12 + 24} textAnchor="middle" fontSize={9} fill="#be185d" fontWeight={700}>
        共 {MLP_REPARAM.numLayers} 层 × 2(K+V)
      </text>

      {/* summary box */}
      <rect x={40} y={200} width={560} height={150} rx={8} fill="var(--bg-surface)" stroke="var(--border)" strokeWidth={1} />
      <text x={60} y={226} fontSize={11} fontWeight={700} fill="var(--ink-primary)">压缩 vs 展开</text>
      <text x={60} y={250} fontSize={10} fill="var(--ink-secondary)">
        训练:只优化 P_small(m × d_small)+ MLP 权重 —— 规模小、训练稳定
      </text>
      <text x={60} y={270} fontSize={10} fill="var(--ink-secondary)">
        推理:提前用 MLP 算出完整 P,缓存后不再需要 MLP 前向
      </text>
      <text x={60} y={294} fontSize={10} fill="var(--ink-secondary)">
        参数量 ~ m × L × d × 2(K+V) —— 直接学全量 prefix(不重参数化)在大模型上不稳定
      </text>
      <text x={60} y={318} fontSize={10} fill="#be185d" fontWeight={700}>
        GPT-2 large(354M),m=10 → 240K 参数 ≈ 0.07%
      </text>
      <text x={60} y={338} fontSize={9} fill="var(--ink-muted)">
        HuggingFace peft 实测:184,320 / 774,030,080 ≈ 0.024%
      </text>

      <defs>
        <marker id="arrowhead-mlp" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#9ca3af" />
        </marker>
      </defs>
    </svg>
  );
}
