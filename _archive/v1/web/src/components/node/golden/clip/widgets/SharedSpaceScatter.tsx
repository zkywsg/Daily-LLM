import { PAIRS, GROUP_COLOR } from "../lib/data";

const W = 700;
const H = 280;

// 2D 投影散点图:每对图文画两个点(emoji + caption tag),并用虚线连起来。
// 同一对的两个点应该非常接近 → 让 viewer 直观看到"对齐"。
// 同 group(动物/交通/食物)的点用同色 → 看到"语义聚类"。

export function SharedSpaceScatter() {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="CLIP shared embedding space 2D projection">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        共享 embedding 空间(2D 投影)— 每对图文应聚得很近
      </text>

      {/* 坐标轴(只是装饰) */}
      <line x1={30} y1={H - 20} x2={W - 20} y2={H - 20} stroke="var(--border)" />
      <line x1={30} y1={30} x2={30} y2={H - 20} stroke="var(--border)" />

      {/* 对子连接虚线 */}
      {PAIRS.map((p, i) => (
        <line
          key={`link-${i}`}
          x1={p.imagePos.x}
          y1={p.imagePos.y}
          x2={p.textPos.x}
          y2={p.textPos.y}
          stroke={GROUP_COLOR[p.group]}
          strokeWidth={1}
          strokeDasharray="2 3"
          opacity={0.5}
        />
      ))}

      {/* image 点 = emoji */}
      {PAIRS.map((p, i) => (
        <g key={`img-${i}`}>
          <circle cx={p.imagePos.x} cy={p.imagePos.y} r={14} fill="var(--bg-surface)" stroke={GROUP_COLOR[p.group]} strokeWidth={2} />
          <text x={p.imagePos.x} y={p.imagePos.y + 6} textAnchor="middle" fontSize={16}>
            {p.emoji}
          </text>
        </g>
      ))}

      {/* text 点 = caption 缩写 */}
      {PAIRS.map((p, i) => (
        <g key={`txt-${i}`}>
          <rect
            x={p.textPos.x - 28}
            y={p.textPos.y + 12}
            width={56}
            height={16}
            rx={3}
            fill="var(--bg-surface)"
            stroke={GROUP_COLOR[p.group]}
            strokeWidth={1.2}
          />
          <text x={p.textPos.x} y={p.textPos.y + 23} textAnchor="middle" fontSize={9} fill="var(--ink-primary)">
            "{p.caption.split(" ").pop()}"
          </text>
        </g>
      ))}

      {/* 图例 */}
      <g transform={`translate(${W - 160}, 50)`}>
        {(["animal", "vehicle", "food"] as const).map((g, i) => (
          <g key={g} transform={`translate(0, ${i * 18})`}>
            <circle cx={6} cy={4} r={5} fill={GROUP_COLOR[g]} />
            <text x={16} y={8} fontSize={10} fill="var(--ink-secondary)">{g}</text>
          </g>
        ))}
      </g>

      <text x={W / 2} y={H - 4} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        虚线把每对图文连起来 · 训练目标就是把这对距离拉到最小
      </text>
    </svg>
  );
}
