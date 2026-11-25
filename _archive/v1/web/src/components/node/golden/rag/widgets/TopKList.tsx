import { QUERIES, topK, GROUP_COLOR } from "../lib/data";

interface Props {
  queryIdx: number;
  k: number;
}

const W = 700;
const H = 360;

// query → top-k 召回结果列表,带 cosine sim 分数条。
// 让 viewer 看到\"语义检索\"输出长什么样。

export function TopKList({ queryIdx, k }: Props) {
  const q = QUERIES[queryIdx];
  const results = topK(q, k);
  const maxSim = Math.max(...results.map((r) => r.sim));
  const rowH = (H - 80) / Math.max(k, 1);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`Top-${k} retrieved docs`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Top-{k} 召回文档 — 按 cosine similarity 排序
      </text>

      {/* Query header */}
      <rect x={20} y={32} width={W - 40} height={28} rx={4} fill="#fef3c7" stroke="#f59e0b" />
      <text x={32} y={50} fontSize={11} fontWeight={700} fill="#92400e">Query:</text>
      <text x={86} y={50} fontSize={11} fill="var(--ink-primary)">{q.text}</text>

      {/* Top-K rows */}
      {results.map((r, i) => {
        const y = 76 + i * rowH;
        const isRelevant = q.relevant.includes(r.doc.id);
        const barW = (W - 380) * (r.sim / Math.max(0.01, maxSim));
        return (
          <g key={r.doc.id}>
            {/* rank */}
            <text x={28} y={y + 18} fontSize={11} fontWeight={700} fill="var(--ink-muted)">
              #{i + 1}
            </text>
            {/* group dot */}
            <circle cx={56} cy={y + 14} r={5} fill={GROUP_COLOR[r.doc.group]} />
            {/* doc id + title */}
            <text x={70} y={y + 14} fontSize={11} fontWeight={600} fill="var(--ink-primary)">
              {r.doc.id} · {r.doc.title}
            </text>
            <text x={70} y={y + 28} fontSize={9} fill="var(--ink-muted)">
              {r.doc.text.length > 60 ? r.doc.text.slice(0, 60) + "…" : r.doc.text}
            </text>
            {/* sim bar */}
            <rect x={370} y={y + 4} width={Math.max(2, barW)} height={16} rx={2} fill={isRelevant ? "#10b981" : "#9ca3af"} opacity={0.7} />
            <text x={375 + barW} y={y + 16} fontSize={10} fontWeight={600} fill="var(--ink-primary)">
              {r.sim.toFixed(2)}
            </text>
            {/* 相关性标记 */}
            {isRelevant && (
              <text x={W - 60} y={y + 18} fontSize={10} fontWeight={700} fill="#10b981">
                ✓ relevant
              </text>
            )}
          </g>
        );
      })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        绿色条 = labeler 标注为相关 · 灰色 = 不相关(可能召回了同 cluster 的近邻)
      </text>
    </svg>
  );
}
