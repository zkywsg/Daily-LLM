const W = 700;
const H = 320;

// 两栏对比表:Lewis 2020 vs 现代工业 RAG。
// 哪些参数固定、哪些更新、典型组件、调优方向。

function Card({
  x, y, w, h, fill, stroke, title, rows,
}: {
  x: number; y: number; w: number; h: number; fill: string; stroke: string;
  title: string; rows: Array<{ k: string; v: string }>;
}) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={6} fill={fill} stroke={stroke} strokeWidth={1.5} />
      <text x={x + w / 2} y={y + 22} textAnchor="middle" fontSize={13} fontWeight={700} fill="#1f2937">
        {title}
      </text>
      {rows.map((r, i) => (
        <g key={i}>
          <text x={x + 14} y={y + 48 + i * 26} fontSize={10} fontWeight={600} fill="#4b5563">
            {r.k}
          </text>
          <text x={x + 100} y={y + 48 + i * 26} fontSize={10} fill="#1f2937">
            {r.v}
          </text>
        </g>
      ))}
    </g>
  );
}

export function ArchitectureCompare() {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Lewis 2020 vs modern industrial RAG comparison">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Lewis 2020 vs 现代工业 RAG
      </text>

      <Card
        x={20}
        y={36}
        w={330}
        h={260}
        fill="#dbeafe"
        stroke="#3b82f6"
        title="Lewis 2020 端到端"
        rows={[
          { k: "retriever:", v: "DPR (BERT dual encoder)" },
          { k: "generator:", v: "BART seq2seq" },
          { k: "训练:", v: "联合训练,梯度回传" },
          { k: "知识库:", v: "Wikipedia (~21M chunks)" },
          { k: "数据:", v: "NQ / TriviaQA QA 对" },
          { k: "代价:", v: "工程复杂,需自训" },
          { k: "效果:", v: "强,retriever 学到适配 generator" },
        ]}
      />
      <Card
        x={360}
        y={36}
        w={330}
        h={260}
        fill="#ecfdf5"
        stroke="#10b981"
        title="现代工业 pipeline"
        rows={[
          { k: "retriever:", v: "OpenAI/BGE/Cohere embed API" },
          { k: "generator:", v: "GPT-4 / Claude / Llama" },
          { k: "训练:", v: "都冻结,只调 prompt 和 k" },
          { k: "知识库:", v: "用户私有数据 (Pinecone/Weaviate)" },
          { k: "数据:", v: "不需要训练数据" },
          { k: "代价:", v: "API 调用费,无 GPU" },
          { k: "效果:", v: "够用 + 上线快,LangChain / LlamaIndex 主流" },
        ]}
      />
    </svg>
  );
}
