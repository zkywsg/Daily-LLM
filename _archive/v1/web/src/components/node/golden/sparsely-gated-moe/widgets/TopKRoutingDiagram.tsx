import { DEMO_TOKENS, SAMPLE_SIZE, topKExperts } from "../lib/data";

interface Props {
  sentenceIdx: number;
  tokenIdx: number;
}

const W = 700;
const H = 340;
const COLS = 16;
const ROWS = SAMPLE_SIZE / COLS; // 4

// N=2048 个 expert 画不下,展示一个 64 个 expert 的采样网格(8x8=64 排成 16x4)。
// 选中的 top-K=4 用粉色高亮,其余全部灰暗——直观体现"精确 0,完全不计算"。

export function TopKRoutingDiagram({ sentenceIdx, tokenIdx }: Props) {
  const sentence = DEMO_TOKENS[sentenceIdx];
  const token = sentence.tokens[tokenIdx] ?? sentence.tokens[0];
  const routed = new Map(topKExperts(token, 4, sentenceIdx * 97 + tokenIdx).map((r) => [r.expert, r.weight]));

  const leftPad = 30;
  const topPad = 70;
  const cellW = (W - leftPad * 2) / COLS;
  const cellH = 40;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label={`Top-K gating routing diagram for token ${token}`}
    >
      <text x={W / 2} y={22} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Top-K Gating — token "{token}" 从 N=2048 个 expert 里选出 top-K=4
      </text>
      <text x={W / 2} y={40} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        (下图展示 {SAMPLE_SIZE} 个采样 expert,真实模型是 2048 个;灰色 = 精确 0,完全不参与计算)
      </text>

      {Array.from({ length: ROWS }, (_, row) =>
        Array.from({ length: COLS }, (_, col) => {
          const e = row * COLS + col;
          const weight = routed.get(e);
          const filled = weight != null;
          const x = leftPad + col * cellW;
          const y = topPad + row * cellH;
          return (
            <g key={`c-${e}`}>
              <rect
                x={x + 2}
                y={y + 2}
                width={cellW - 4}
                height={cellH - 6}
                rx={3}
                fill={filled ? `hsl(330, 70%, ${92 - weight * 45}%)` : "#f3f4f6"}
                stroke={filled ? "#ec4899" : "#d1d5db"}
                strokeWidth={filled ? 1.6 : 0.6}
              />
              <text
                x={x + cellW / 2}
                y={y + cellH / 2 - 2}
                textAnchor="middle"
                fontSize={8.5}
                fontWeight={filled ? 700 : 400}
                fill={filled ? "var(--ink-primary)" : "#9ca3af"}
              >
                E{e}
              </text>
              {filled && (
                <text x={x + cellW / 2} y={y + cellH / 2 + 10} textAnchor="middle" fontSize={8} fontWeight={700} fill="#831843">
                  {weight.toFixed(2)}
                </text>
              )}
            </g>
          );
        }),
      )}

      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        只有 4 个粉色 cell 参与该 token 的计算 · 其余 2044 个 expert 这一步完全跳过,算力真的省下来
      </text>
    </svg>
  );
}
