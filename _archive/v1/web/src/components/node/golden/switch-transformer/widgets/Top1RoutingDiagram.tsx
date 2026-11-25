import { DEMO_SENTENCES, NUM_EXPERTS, topKExperts } from "../lib/data";

interface Props {
  sentenceIdx: number;
  k: number;
}

const W = 700;
const H = 380;

// 一句话每个 token 的路由:k=1 时(Switch)每行只有 1 个粉色 cell,
// 切到 k=2/4 对比 Shazeer/Mixtral 风格的 top-k,直观看到 "只走一个 expert" 有多稀疏。

export function Top1RoutingDiagram({ sentenceIdx, k }: Props) {
  const sentence = DEMO_SENTENCES[sentenceIdx];
  const tokens = sentence.tokens;

  const leftPad = 90;
  const topPad = 60;
  const cellW = (W - leftPad - 30) / NUM_EXPERTS;
  const cellH = 36;

  const routings = tokens.map((t) => topKExperts(t, k));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`Top-${k} routing: ${sentence.label}`}>
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        {k === 1 ? "Top-1 routing (Switch Transformer)" : `Top-${k} routing (对比)`} — 每 token 各自选 {k} 个 expert
      </text>

      {Array.from({ length: NUM_EXPERTS }, (_, e) => (
        <text key={`col-${e}`} x={leftPad + cellW * (e + 0.5)} y={topPad - 8} textAnchor="middle" fontSize={10} fontWeight={600} fill="var(--ink-secondary)">
          E{e}
        </text>
      ))}

      {tokens.map((t, i) => {
        const y = topPad + i * cellH;
        const routed = new Map(routings[i].map((r) => [r.expert, r.weight]));
        return (
          <g key={`tok-${i}`}>
            <text x={leftPad - 8} y={y + cellH / 2 + 4} textAnchor="end" fontSize={11} fontWeight={500} fill="var(--ink-primary)">
              {t}
            </text>
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
                    fill={filled ? `hsl(38, 92%, ${95 - weight * 45}%)` : "var(--bg-surface)"}
                    stroke={filled ? "#f59e0b" : "var(--border)"}
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
        {k === 1
          ? "每行只有 1 个黄色 cell · 路由和通信量只有 top-4 (Shazeer) 的 1/4"
          : `每行 ${k} 个黄色 cell · 算力与通信量随 k 线性增长`}
      </text>
    </svg>
  );
}
