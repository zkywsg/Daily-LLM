interface Props {
  tokens: string[];
  /** 当前 query token 的 index */
  queryIdx: number;
  onQueryIdxChange: (i: number) => void;
}

const W = 700;
const H = 240;

// 一行 token,选中某个 query 后画弧线指向所有它可以 attend 的 key。
// BERT 是双向 → 弧线左右都有;对比 GPT 只指向左侧。
// 点击 token 切换 query 看不同位置的 attention 形状。
export function BidirectionalDemo({ tokens, queryIdx, onQueryIdxChange }: Props) {
  const n = tokens.length;
  const totalW = W - 60;
  const tokenSpacing = totalW / Math.max(1, n);
  const cy = H / 2 + 20;
  const tokenY = cy + 0;

  const tokenX = (i: number) => 30 + tokenSpacing * (i + 0.5);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="BERT bidirectional attention demo">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        点 token 切换 query —— BERT 同时看左右上下文
      </text>

      {/* 弧线:从 query 指向所有其他 token */}
      {tokens.map((_, i) => {
        if (i === queryIdx) return null;
        const x1 = tokenX(queryIdx);
        const x2 = tokenX(i);
        const dx = x2 - x1;
        const arcHeight = 40 + Math.abs(dx) * 0.4;
        // 弧的方向:左侧的 key 用蓝色弧(过去),右侧用粉色弧(未来) —— 直观体现"BERT 看双向"
        const isLeft = i < queryIdx;
        const color = isLeft ? "#3b82f6" : "#ec4899";
        return (
          <path
            key={`arc-${i}`}
            d={`M ${x1} ${tokenY - 8} Q ${(x1 + x2) / 2} ${tokenY - arcHeight} ${x2} ${tokenY - 8}`}
            fill="none"
            stroke={color}
            strokeWidth={1.5}
            opacity={0.6}
          />
        );
      })}

      {/* token 圆圈 */}
      {tokens.map((t, i) => {
        const isQuery = i === queryIdx;
        return (
          <g
            key={`tok-${i}`}
            transform={`translate(${tokenX(i)}, ${tokenY})`}
            style={{ cursor: "pointer" }}
            onClick={() => onQueryIdxChange(i)}
          >
            <circle
              r={18}
              fill={isQuery ? "#fce7f3" : "var(--bg-surface)"}
              stroke={isQuery ? "#ec4899" : "var(--border)"}
              strokeWidth={isQuery ? 2 : 1}
            />
            <text
              y={4}
              textAnchor="middle"
              fontSize={11}
              fontWeight={isQuery ? 700 : 500}
              fill="var(--ink-primary)"
            >
              {t.length > 5 ? t.slice(0, 5) + "…" : t}
            </text>
          </g>
        );
      })}

      {/* 图例 */}
      <g transform={`translate(${W / 2 - 130}, ${H - 30})`}>
        <line x1={0} x2={20} y1={0} y2={0} stroke="#3b82f6" strokeWidth={1.5} />
        <text x={26} y={4} fontSize={10} fill="var(--ink-secondary)">← 看过去 (左侧上下文)</text>
        <line x1={130} x2={150} y1={0} y2={0} stroke="#ec4899" strokeWidth={1.5} />
        <text x={156} y={4} fontSize={10} fill="var(--ink-secondary)">→ 看未来 (右侧上下文)</text>
      </g>
    </svg>
  );
}
