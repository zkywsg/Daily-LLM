const W = 700;
const H = 260;

interface Props {
  mode: "store" | "recompute";
}

export function RecomputeTradeoffDiagram({ mode }: Props) {
  const isStore = mode === "store";

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="反向传播 存储 vs 重计算 对比">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        反向传播:{isStore ? "存储 S/P(标准做法)" : "Recomputation(FlashAttention)"}
      </text>

      <g transform="translate(60, 50)">
        <rect x={0} y={0} width={90} height={40} fill="#dbeafe" stroke="#3b82f6" rx={4} />
        <text x={45} y={24} textAnchor="middle" fontSize={11} fontWeight={700} fill="#1e40af">Forward</text>

        <line x1={90} y1={20} x2={150} y2={20} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#arrow)" />

        {isStore ? (
          <>
            <rect x={150} y={0} width={140} height={40} fill="#fce7f3" stroke="#ec4899" strokeWidth={2} rx={4} />
            <text x={220} y={18} textAnchor="middle" fontSize={10} fontWeight={700} fill="#be185d">存 S, P 到 HBM</text>
            <text x={220} y={32} textAnchor="middle" fontSize={9} fill="#be185d">O(N²) 显存</text>

            <line x1={290} y1={20} x2={350} y2={20} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#arrow)" />
            <rect x={350} y={0} width={110} height={40} fill="#dbeafe" stroke="#3b82f6" rx={4} />
            <text x={405} y={18} textAnchor="middle" fontSize={10} fontWeight={700} fill="#1e40af">Backward</text>
            <text x={405} y={32} textAnchor="middle" fontSize={9} fill="#1e40af">读 S/P 算梯度</text>
          </>
        ) : (
          <>
            <rect x={150} y={0} width={140} height={40} fill="#ecfdf5" stroke="#10b981" strokeWidth={2} rx={4} />
            <text x={220} y={18} textAnchor="middle" fontSize={10} fontWeight={700} fill="#065f46">只存 O, (m,ℓ), Q/K/V</text>
            <text x={220} y={32} textAnchor="middle" fontSize={9} fill="#065f46">O(N) 显存</text>

            <line x1={290} y1={20} x2={350} y2={20} stroke="#9ca3af" strokeWidth={1.5} markerEnd="url(#arrow)" />
            <rect x={350} y={0} width={110} height={40} fill="#fef3c7" stroke="#f59e0b" strokeWidth={2} rx={4} />
            <text x={405} y={18} textAnchor="middle" fontSize={10} fontWeight={700} fill="#b45309">Backward</text>
            <text x={405} y={32} textAnchor="middle" fontSize={9} fill="#b45309">重算 S/P + 梯度</text>
          </>
        )}

        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="#9ca3af" />
          </marker>
        </defs>
      </g>

      <g transform="translate(60, 140)">
        <text x={0} y={0} fontSize={10} fontWeight={700} fill="var(--ink-muted)">代价对比</text>
        <text x={0} y={22} fontSize={11} fill={isStore ? "#be185d" : "var(--ink-secondary)"}>
          显存:{isStore ? "O(N²) — 机制一二白做,反向把显存吃回去" : "O(N) — 训练全程保持"}
        </text>
        <text x={0} y={44} fontSize={11} fill={!isStore ? "#b45309" : "var(--ink-secondary)"}>
          反向 FLOPs:{isStore ? "1×(直接读)" : "~2.5×(多算一遍,但省下的内存带宽更值)"}
        </text>
        <text x={0} y={66} fontSize={11} fontWeight={700} fill={!isStore ? "#065f46" : "var(--ink-secondary)"}>
          实际 wall-clock:{isStore ? "受限于 O(N²) 内存搬运" : "更短 — 内存带宽是真瓶颈,多算的 FLOPs 换来更少等待"}
        </text>
      </g>

      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        经典"以计算换内存"权衡 — 多 ~20% 总算力,端到端训练时间反而更短
      </text>
    </svg>
  );
}
