import { CORPUS, GROUP_COLOR, QUERIES, topK } from "../lib/data";

interface Props {
  queryIdx: number;
  k: number;
}

const W = 700;
const H = 420;

// 2D 投影散点图:CORPUS 10 个 doc 用 group 色 + ID 标号显示,
// query 用粉色五角星表示,top-k 召回的 doc 高亮 + 圆环。

export function CorpusEmbeddingMap({ queryIdx, k }: Props) {
  const query = QUERIES[queryIdx];
  const retrieved = topK(query, k);
  const retrievedIds = new Set(retrieved.map((r) => r.doc.id));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`Corpus embedding map, query: ${query.text}`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Dense Retrieval — corpus 2D 投影 + query 找 top-{k}
      </text>
      <text x={W / 2} y={38} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        语义相似的 docs 聚成 cluster · query 投影后找最近 k 个
      </text>

      {/* 检索连接虚线(query → retrieved docs) */}
      {retrieved.map((r) => (
        <line
          key={`link-${r.doc.id}`}
          x1={query.pos.x}
          y1={query.pos.y}
          x2={r.doc.pos.x}
          y2={r.doc.pos.y}
          stroke="#ec4899"
          strokeWidth={1}
          strokeDasharray="3 3"
          opacity={0.55}
        />
      ))}

      {/* CORPUS docs */}
      {CORPUS.map((d) => {
        const inTopK = retrievedIds.has(d.id);
        return (
          <g key={d.id}>
            {inTopK && <circle cx={d.pos.x} cy={d.pos.y} r={20} fill="none" stroke="#ec4899" strokeWidth={1.6} strokeDasharray="2 2" opacity={0.6} />}
            <circle cx={d.pos.x} cy={d.pos.y} r={10} fill={GROUP_COLOR[d.group]} opacity={0.85} />
            <text x={d.pos.x} y={d.pos.y + 4} textAnchor="middle" fontSize={10} fontWeight={700} fill="#fff">
              {d.id}
            </text>
            <text x={d.pos.x} y={d.pos.y + 24} textAnchor="middle" fontSize={9} fill="var(--ink-secondary)">
              {d.title.length > 8 ? d.title.slice(0, 8) + "…" : d.title}
            </text>
          </g>
        );
      })}

      {/* Query 五角星 */}
      <polygon
        points={`${query.pos.x},${query.pos.y - 14} ${query.pos.x + 4.5},${query.pos.y - 4} ${query.pos.x + 14},${query.pos.y - 4} ${query.pos.x + 6.5},${query.pos.y + 3} ${query.pos.x + 9},${query.pos.y + 13} ${query.pos.x},${query.pos.y + 7} ${query.pos.x - 9},${query.pos.y + 13} ${query.pos.x - 6.5},${query.pos.y + 3} ${query.pos.x - 14},${query.pos.y - 4} ${query.pos.x - 4.5},${query.pos.y - 4}`}
        fill="#fce7f3"
        stroke="#ec4899"
        strokeWidth={2}
      />
      <text x={query.pos.x} y={query.pos.y - 22} textAnchor="middle" fontSize={10} fontWeight={700} fill="#831843">
        Query
      </text>

      {/* 图例 */}
      <g transform={`translate(20, ${H - 60})`}>
        {(Object.keys(GROUP_COLOR) as Array<keyof typeof GROUP_COLOR>).map((g, i) => (
          <g key={g} transform={`translate(${i * 110}, 0)`}>
            <circle cx={5} cy={5} r={5} fill={GROUP_COLOR[g]} />
            <text x={14} y={9} fontSize={10} fill="var(--ink-secondary)">{g}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}
