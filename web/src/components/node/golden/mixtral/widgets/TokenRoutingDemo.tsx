import { DEMO_SENTENCES, NUM_EXPERTS, topKExperts } from "../lib/data";

interface Props {
  sentenceIdx: number;
  k: number;
}

const W = 700;
const H = 380;

// 一句话每个 token 都有自己的 top-k expert 路由。
// 用一个 token × expert 的网格,选中的 cell 用粉色+权重显示。
// 让 viewer 看到 \"per-token routing\" —— 同一句不同词去不同 expert。

export function TokenRoutingDemo({ sentenceIdx, k }: Props) {
  const sentence = DEMO_SENTENCES[sentenceIdx];
  const tokens = sentence.tokens;

  const leftPad = 90;
  const topPad = 60;
  const cellW = (W - leftPad - 30) / NUM_EXPERTS;
  const cellH = 36;

  // 每个 token 的 top-k 路由
  const routings = tokens.map((t) => topKExperts(t, k));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`Per-token routing: ${sentence.label}`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Per-token routing — 同一句话每个 token 各自选 top-{k} expert
      </text>

      {/* 列标题:expert 0-7 */}
      {Array.from({ length: NUM_EXPERTS }, (_, e) => (
        <text key={`col-${e}`} x={leftPad + cellW * (e + 0.5)} y={topPad - 8} textAnchor="middle" fontSize={10} fontWeight={600} fill="var(--ink-secondary)">
          E{e}
        </text>
      ))}

      {/* 行:每个 token */}
      {tokens.map((t, i) => {
        const y = topPad + i * cellH;
        const routed = new Map(routings[i].map((r) => [r.expert, r.weight]));
        return (
          <g key={`tok-${i}`}>
            <text x={leftPad - 8} y={y + cellH / 2 + 4} textAnchor="end" fontSize={11} fontWeight={500} fill="var(--ink-primary)">
              {t}
            </text>
            {/* 每个 expert cell */}
            {Array.from({ length: NUM_EXPERTS }, (_, e) => {
              const weight = routed.get(e);
              const filled = weight != null;
              return (
                <g key={`c-${i}-${e}`}>
                  <rect
                    x={leftPad + e * cellW}
                    y={y + 2}
                    width={cellW - 2}
                    height={cellH - 4}
                    fill={filled ? `hsl(330, 70%, ${95 - weight * 50}%)` : "var(--bg-surface)"}
                    stroke={filled ? "#ec4899" : "var(--border)"}
                    strokeWidth={filled ? 1.4 : 0.6}
                  />
                  {filled && (
                    <text x={leftPad + (e + 0.5) * cellW} y={y + cellH / 2 + 4} textAnchor="middle" fontSize={10} fontWeight={700} fill="var(--ink-primary)">
                      {weight.toFixed(2)}
                    </text>
                  )}
                </g>
              );
            })}
          </g>
        );
      })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        每行只有 {k} 个粉色 cell · 不同 token 路由到不同 expert · 路由决策是 per-token 的
      </text>
    </svg>
  );
}
