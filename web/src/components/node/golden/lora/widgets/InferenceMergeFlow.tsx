interface Props {
  merged: boolean;
}

const W = 700;
const H = 220;

// 两种推理时态:
//   未合并:x → W₀x + (α/r)·BAx → y(两条分支,2 次 matmul)
//   合并  :x → W'x → y      其中 W' = W₀ + (α/r)·BA(1 次 matmul,跟原模型同延迟)
// 切 toggle 看两条数据流的高亮变化,viewer 立刻看到"零延迟"的含义。

function Box({
  x,
  y,
  w,
  h,
  fill,
  stroke,
  label,
  sub,
  faded,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  fill: string;
  stroke: string;
  label: string;
  sub?: string;
  faded?: boolean;
}) {
  return (
    <g opacity={faded ? 0.25 : 1}>
      <rect x={x} y={y} width={w} height={h} rx={5} fill={fill} stroke={stroke} strokeWidth={1.5} />
      <text x={x + w / 2} y={y + h / 2 - 2} textAnchor="middle" fontSize={12} fontWeight={600} fill="#1f2937">
        {label}
      </text>
      {sub && (
        <text x={x + w / 2} y={y + h / 2 + 12} textAnchor="middle" fontSize={10} fill="#6b7280">
          {sub}
        </text>
      )}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2, label, highlight, faded }: { x1: number; y1: number; x2: number; y2: number; label?: string; highlight?: boolean; faded?: boolean }) {
  const color = highlight ? "#10b981" : faded ? "#d1d5db" : "#9ca3af";
  const id = `arr-${x1}-${y1}-${x2}-${y2}-${highlight ? "h" : faded ? "f" : "n"}`;
  return (
    <g opacity={faded ? 0.4 : 1}>
      <defs>
        <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={highlight ? 2.5 : 1.5} markerEnd={`url(#${id})`} />
      {label && (
        <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 6} textAnchor="middle" fontSize={11} fontStyle="italic" fill={color}>
          {label}
        </text>
      )}
    </g>
  );
}

export function InferenceMergeFlow({ merged }: Props) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`Inference flow, merged=${merged}`}>
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--ink-primary)">
        {merged ? "推理时:合并后(W' = W₀ + α/r · BA)" : "推理时:未合并(两条分支并联)"}
      </text>

      {/* 输入 x */}
      <Box x={20} y={90} w={70} h={40} fill="#fef3c7" stroke="#f59e0b" label="x" sub="输入" />

      {/* 主路 W₀ */}
      <Arrow x1={90} y1={110} x2={200} y2={110} highlight={merged} faded={merged} />
      <Box x={200} y={90} w={90} h={40} fill="#e5e7eb" stroke="#6b7280" label="W₀ x" sub="frozen matmul" faded={merged} />

      {/* 旁路 B·A */}
      <Arrow x1={90} y1={110} x2={200} y2={170} highlight={!merged && false} faded={merged} />
      <Box x={200} y={150} w={90} h={40} fill="#dbeafe" stroke="#3b82f6" label="(α/r)·B·A·x" sub="LoRA matmul" faded={merged} />

      {/* + */}
      <Arrow x1={290} y1={110} x2={370} y2={130} highlight={!merged} faded={merged} />
      <Arrow x1={290} y1={170} x2={370} y2={140} highlight={!merged} faded={merged} />
      <Box x={370} y={110} w={50} h={40} fill="#fce7f3" stroke="#ec4899" label="+" faded={merged} />
      <Arrow x1={420} y1={130} x2={500} y2={130} highlight={!merged} faded={merged} />

      {/* 合并后单路 */}
      {merged && (
        <>
          <Arrow x1={90} y1={110} x2={300} y2={110} highlight />
          <Box x={300} y={90} w={150} h={40} fill="#ecfdf5" stroke="#10b981" label="W' x" sub="single matmul · 零延迟" />
          <Arrow x1={450} y1={110} x2={520} y2={110} highlight />
        </>
      )}

      {/* 输出 y */}
      <Box x={merged ? 520 : 500} y={90} w={60} h={40} fill="#fef3c7" stroke="#f59e0b" label="y" sub="输出" />

      {/* 注脚 */}
      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={11} fontStyle="italic" fill={merged ? "#10b981" : "#6b7280"}>
        {merged
          ? "✓ 合并后只剩一次矩阵乘 → 跟原模型完全一样的推理延迟"
          : "训练 / 切换 task 时保留两路:可以 swap 不同 task 的 BA"}
      </text>
    </svg>
  );
}
