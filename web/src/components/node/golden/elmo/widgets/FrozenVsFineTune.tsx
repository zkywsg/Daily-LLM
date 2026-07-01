const W = 700;
const H = 340;

interface Props {
  mode: "elmo" | "bert";
}

function Box({ x, y, w, h, fill, stroke, label, sub, dashed }: {
  x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string; sub?: string; dashed?: boolean;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={4} fill={fill} stroke={stroke} strokeWidth={1.4} strokeDasharray={dashed ? "5 3" : undefined} />
      <text x={x + w / 2} y={y + h / 2 - 2} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1f2937">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 13} textAnchor="middle" fontSize={9} fill="#6b7280">{sub}</text>}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2, color, id }: { x1: number; y1: number; x2: number; y2: number; color: string; id: string }) {
  return (
    <g>
      <defs>
        <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={1.4} markerEnd={`url(#${id})`} />
    </g>
  );
}

export function FrozenVsFineTune({ mode }: Props) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="ELMo frozen feature vs BERT fine-tune">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {mode === "elmo"
          ? "ELMo 用法:frozen feature 拼接到下游 input"
          : "BERT 用法:端到端 fine-tune 所有参数"}
      </text>

      {mode === "elmo" ? (
        <>
          {/* Input tokens */}
          <Box x={40} y={60} w={100} h={36} fill="#f3f4f6" stroke="#d1d5db" label="input tokens" />

          {/* GloVe 分支 */}
          <Arrow x1={90} y1={96} x2={90} y2={130} color="#9ca3af" id="e-a" />
          <Box x={40} y={130} w={100} h={36} fill="#dbeafe" stroke="#3b82f6" label="GloVe" sub="300 维" />
          <Arrow x1={90} y1={166} x2={90} y2={210} color="#3b82f6" id="e-b" />

          {/* ELMo 分支 */}
          <Box x={200} y={60} w={140} h={36} fill="#fef3c7" stroke="#f59e0b" label="ELMo (frozen 🔒)" sub="94M 参数不更新" dashed />
          <Arrow x1={270} y1={96} x2={270} y2={210} color="#f59e0b" id="e-c" />

          {/* Concat */}
          <Box x={80} y={210} w={220} h={36} fill="#ecfdf5" stroke="#10b981" label="concat [glove ; elmo]" sub="1324 维" />
          <Arrow x1={190} y1={246} x2={190} y2={270} color="#10b981" id="e-d" />

          {/* 下游模型 */}
          <Box x={80} y={270} w={220} h={40} fill="#fce7f3" stroke="#ec4899" label="下游 BiLSTM-CRF" sub="只训这部分 + 加权 s_j / γ" />

          {/* 右侧说明 */}
          <g>
            <text x={420} y={80} fontSize={11} fontWeight={700} fill="#831843">优势</text>
            <text x={420} y={100} fontSize={10} fill="#374151">✓ 下游训练快(不反传 94M)</text>
            <text x={420} y={118} fontSize={10} fill="#374151">✓ 多任务共享同一份 ELMo</text>
            <text x={420} y={136} fontSize={10} fill="#374151">✓ 每任务只学 s_j + γ 权重</text>
            <text x={420} y={168} fontSize={11} fontWeight={700} fill="#831843">局限</text>
            <text x={420} y={188} fontSize={10} fill="#374151">✗ ELMo 内部不能调优</text>
            <text x={420} y={206} fontSize={10} fill="#374151">✗ 下游任务表达力受限</text>
            <text x={420} y={224} fontSize={10} fill="#374151">✗ 后被 BERT fine-tune 超越</text>
          </g>
        </>
      ) : (
        <>
          <Box x={40} y={60} w={100} h={36} fill="#f3f4f6" stroke="#d1d5db" label="input tokens" />
          <Arrow x1={90} y1={96} x2={90} y2={130} color="#9ca3af" id="b-a" />
          <Box x={40} y={130} w={280} h={80} fill="#fce7f3" stroke="#ec4899" label="BERT (all trainable 🔥)" sub="340M 参数全部端到端更新" />
          <Arrow x1={180} y1={210} x2={180} y2={250} color="#ec4899" id="b-b" />
          <Box x={80} y={250} w={200} h={40} fill="#ecfdf5" stroke="#10b981" label="task head" sub="小分类层" />

          <g>
            <text x={420} y={80} fontSize={11} fontWeight={700} fill="#065f46">优势</text>
            <text x={420} y={100} fontSize={10} fill="#374151">✓ BERT 内部针对任务调优</text>
            <text x={420} y={118} fontSize={10} fill="#374151">✓ 表达力更强</text>
            <text x={420} y={136} fontSize={10} fill="#374151">✓ 端到端简洁</text>
            <text x={420} y={168} fontSize={11} fontWeight={700} fill="#065f46">代价</text>
            <text x={420} y={188} fontSize={10} fill="#374151">✗ 每任务都要复制 BERT 权重</text>
            <text x={420} y={206} fontSize={10} fill="#374151">✗ 训练慢 · 显存大</text>
            <text x={420} y={224} fontSize={10} fill="#374151">✗ 多任务部署成本高</text>
          </g>
        </>
      )}

      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        ELMo frozen feature 是承前启后的工程范式 · BERT fine-tune 后来成主流
      </text>
    </svg>
  );
}
