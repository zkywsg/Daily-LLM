const W = 700;
const H = 280;

interface Props {
  highlight: "scan" | "train" | "reuse" | null;
}

function Box({ x, y, w, h, fill, stroke, label, sub, opacity = 1 }: {
  x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string; sub?: string; opacity?: number;
}) {
  return (
    <g opacity={opacity}>
      <rect x={x} y={y} width={w} height={h} rx={4} fill={fill} stroke={stroke} strokeWidth={1.4} />
      <text x={x + w / 2} y={y + h / 2 - 2} textAnchor="middle" fontSize={11} fontWeight={600} fill="#1f2937">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 12} textAnchor="middle" fontSize={9} fill="#6b7280">{sub}</text>}
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

export function TrainingPipeline({ highlight }: Props) {
  const dimScan = highlight && highlight !== "scan" ? 0.35 : 1;
  const dimTrain = highlight && highlight !== "train" ? 0.35 : 1;
  const dimReuse = highlight && highlight !== "reuse" ? 0.35 : 1;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="GloVe training pipeline">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        GloVe 训练 Pipeline — 全局共现矩阵替代局部窗口
      </text>

      <g opacity={dimScan}>
        <Box x={30} y={70} w={160} h={60} fill="#dbeafe" stroke="#3b82f6" label="① 扫一遍全语料" sub="构建 V×V 共现矩阵" />
      </g>

      <Arrow x1={190} y1={100} x2={250} y2={100} color="#3b82f6" id="tp-a1" />

      <g opacity={dimTrain}>
        <Box x={250} y={70} w={160} h={60} fill="#fce7f3" stroke="#ec4899" label="② AdaGrad 优化" sub="weighted squared loss" />
      </g>

      <Arrow x1={410} y1={100} x2={470} y2={100} color="#ec4899" id="tp-a2" />

      <g opacity={dimReuse}>
        <Box x={470} y={70} w={200} h={60} fill="#ecfdf5" stroke="#10b981" label="③ 输出词向量" sub="(embed + context_embed)/2" />
      </g>

      <line x1={190} y1={130} x2={190} y2={200} stroke="#3b82f6" strokeWidth={1.2} strokeDasharray="3 3" />
      <text x={190} y={215} textAnchor="middle" fontSize={9} fill="#3b82f6">一次建好,可复用</text>
      <line x1={190} y1={200} x2={330} y2={200} stroke="#3b82f6" strokeWidth={1.2} strokeDasharray="3 3" markerEnd="url(#tp-a3)" />
      <defs>
        <marker id="tp-a3" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#3b82f6" />
        </marker>
      </defs>
      <text x={260} y={190} textAnchor="middle" fontSize={9} fill="#3b82f6">训不同维度/超参不用重扫语料</text>

      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        Word2Vec 每步扫新 mini-batch;GloVe 一次统计,后续都从矩阵采样
      </text>
    </svg>
  );
}
