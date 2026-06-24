import { MASKING_STRATEGY } from "../lib/data";

const W = 700;
const H = 180;

// 三段堆叠条:80% [MASK] + 10% 随机 + 10% 保持。
// 这是 BERT 论文里"为啥不全 [MASK]"的关键招式 —— 解释训推不匹配怎么解决。
export function MaskingStrategyBar() {
  const innerW = W - 60;
  let xCursor = 30;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="BERT MLM masking strategy">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        被选中的 15% token 内部再分三种处理(BERT 论文 §3.1)
      </text>

      {/* 主条 */}
      {MASKING_STRATEGY.map((s) => {
        const w = (innerW * s.pct) / 100;
        const x = xCursor;
        xCursor += w;
        return (
          <g key={s.label}>
            <rect x={x} y={50} width={w} height={36} fill={s.color} opacity={0.85} />
            <text x={x + w / 2} y={72} textAnchor="middle" fontSize={12} fontWeight={600} fill="#fff">
              {s.pct}%
            </text>
            <text x={x + w / 2} y={104} textAnchor="middle" fontSize={11} fill="var(--ink-primary)">
              {s.label}
            </text>
            <text x={x + w / 2} y={122} textAnchor="middle" fontSize={10} fill="var(--ink-muted)" fontStyle="italic">
              {s.note}
            </text>
          </g>
        );
      })}

      {/* 注脚 */}
      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        全 100% 用 [MASK] 会让模型"只在 [MASK] 出现时努力" — 混入随机词和原词避免过拟合 mask 信号
      </text>
    </svg>
  );
}
