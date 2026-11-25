interface Props {
  mode: "e2e" | "pipeline";
}

const W = 720;
const H = 320;

// Lewis 2020 原版:retriever 和 generator 端到端联合训练(BART + DPR)。
// 现代工业 RAG:retriever 和 generator 解耦,大多用现成 embed + GPT-4。
// 用两种 mode 切换显示两条不同的 data flow。

function Box({
  x, y, w, h, fill, stroke, label, sub,
}: {
  x: number; y: number; w: number; h: number; fill: string; stroke: string; label: string; sub?: string;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={6} fill={fill} stroke={stroke} strokeWidth={1.5} />
      <text x={x + w / 2} y={y + h / 2 - 2} textAnchor="middle" fontSize={12} fontWeight={600} fill="#1f2937">{label}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fontSize={10} fill="#6b7280">{sub}</text>}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2, label, dashed }: { x1: number; y1: number; x2: number; y2: number; label?: string; dashed?: boolean }) {
  const id = `pipe2-${x1}-${y1}-${x2}-${y2}-${dashed ? "d" : ""}`;
  return (
    <g>
      <defs>
        <marker id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af" />
        </marker>
      </defs>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#9ca3af" strokeWidth={1.5} strokeDasharray={dashed ? "4 3" : "none"} markerEnd={`url(#${id})`} />
      {label && <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 6} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#6b7280">{label}</text>}
    </g>
  );
}

export function E2EvsPipeline({ mode }: Props) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`${mode === "e2e" ? "Lewis 2020 end-to-end" : "现代工业 pipeline"} RAG`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {mode === "e2e"
          ? "① Lewis 2020 原版 — 端到端联合训练(retriever + generator)"
          : "② 现代工业 RAG — retriever / generator 解耦"}
      </text>

      {/* query */}
      <Box x={30} y={130} w={100} h={50} fill="#fef3c7" stroke="#f59e0b" label="query" sub="user 输入" />
      <Arrow x1={130} y1={155} x2={200} y2={155} />

      {/* retriever (e2e 用 DPR;pipeline 用 ada/cohere embed) */}
      <Box
        x={200}
        y={130}
        w={140}
        h={50}
        fill="#fce7f3"
        stroke="#ec4899"
        label={mode === "e2e" ? "DPR retriever" : "Embed API"}
        sub={mode === "e2e" ? "BERT dual encoder" : "OpenAI / Cohere / BGE"}
      />
      <Arrow x1={340} y1={155} x2={400} y2={155} label="top-k" />

      {/* generator */}
      <Box
        x={400}
        y={130}
        w={140}
        h={50}
        fill="#dbeafe"
        stroke="#3b82f6"
        label={mode === "e2e" ? "BART generator" : "GPT-4 / Claude"}
        sub={mode === "e2e" ? "seq2seq" : "现成 LLM API"}
      />
      <Arrow x1={540} y1={155} x2={620} y2={155} />

      {/* answer */}
      <Box x={620} y={130} w={80} h={50} fill="#ecfdf5" stroke="#10b981" label="answer" />

      {/* e2e: 梯度联合回传(粉色实线) */}
      {mode === "e2e" && (
        <>
          <text x={W / 2} y={H - 70} textAnchor="middle" fontSize={11} fontWeight={700} fill="#ec4899">
            ↓ 联合梯度回传 ↓
          </text>
          <path
            d={`M 470 180 Q 470 250 270 250 Q 270 180 270 180`}
            fill="none"
            stroke="#ec4899"
            strokeWidth={2}
            strokeDasharray="6 3"
          />
          <text x={W / 2} y={H - 18} textAnchor="middle" fontSize={11} fontStyle="italic" fill="#831843">
            ✓ retriever 在 \"对生成有用的 doc\" 上变得更敏锐 · ✗ 工程复杂、必须自训
          </text>
        </>
      )}
      {/* pipeline: retriever / generator 解耦,各自独立 */}
      {mode === "pipeline" && (
        <>
          <Box x={200} y={230} w={140} h={36} fill="#f3f4f6" stroke="#9ca3af" label="冻结" sub="无梯度回传" />
          <Box x={400} y={230} w={140} h={36} fill="#f3f4f6" stroke="#9ca3af" label="冻结" sub="无梯度回传" />
          <text x={W / 2} y={H - 18} textAnchor="middle" fontSize={11} fontStyle="italic" fill="#10b981">
            ✓ 全是现成模块,只需把它们\"拼起来\" · ✗ retriever 和 generator 不能互相适配
          </text>
        </>
      )}
    </svg>
  );
}
