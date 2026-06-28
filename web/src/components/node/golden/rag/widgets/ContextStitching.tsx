import { QUERIES, topK } from "../lib/data";

interface Props {
  queryIdx: number;
  k: number;
}

const W = 700;
const H = 380;

// 把召回的 chunks 按 RAG prompt 模板拼装,显示最终送给 LLM 的 prompt。

export function ContextStitching({ queryIdx, k }: Props) {
  const q = QUERIES[queryIdx];
  const results = topK(q, k);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`Context stitched prompt for query: ${q.text}`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Context Augmentation — 把召回 chunks 拼进 prompt
      </text>

      {/* 最终 prompt 框 */}
      <rect x={20} y={36} width={W - 40} height={H - 60} rx={6} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.5} />

      <text x={32} y={58} fontSize={10} fontWeight={700} fontFamily="ui-monospace, monospace" fill="#92400e">
        [system]
      </text>
      <text x={32} y={74} fontSize={10} fontFamily="ui-monospace, monospace" fill="var(--ink-primary)">
        请根据下面的参考资料回答问题。如果资料里没有答案,说\"不知道\"。
      </text>

      <text x={32} y={102} fontSize={10} fontWeight={700} fontFamily="ui-monospace, monospace" fill="#92400e">
        [retrieved context]
      </text>
      {results.map((r, i) => (
        <g key={r.doc.id}>
          <rect x={32} y={108 + i * 56} width={W - 80} height={50} rx={3} fill="#fce7f3" stroke="#ec4899" strokeWidth={1} />
          <text x={42} y={124 + i * 56} fontSize={10} fontWeight={700} fontFamily="ui-monospace, monospace" fill="#831843">
            [chunk-{i + 1}] {r.doc.id}: {r.doc.title}
          </text>
          <text x={42} y={140 + i * 56} fontSize={9} fontFamily="ui-monospace, monospace" fill="var(--ink-primary)">
            {r.doc.text.length > 70 ? r.doc.text.slice(0, 70) + "…" : r.doc.text}
          </text>
        </g>
      ))}

      <text x={32} y={H - 50} fontSize={10} fontWeight={700} fontFamily="ui-monospace, monospace" fill="#92400e">
        [user question]
      </text>
      <text x={32} y={H - 34} fontSize={10} fontFamily="ui-monospace, monospace" fill="var(--ink-primary)">
        {q.text}
      </text>
    </svg>
  );
}
