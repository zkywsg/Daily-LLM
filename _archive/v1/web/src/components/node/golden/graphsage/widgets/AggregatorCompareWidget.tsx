import { NODE_FEATURES, meanAggregate, maxPoolAggregate, orderSensitiveAggregate } from "../lib/data";

interface Props {
  neighborIds: number[];
}

const W = 680;
const H = 320;

// 同一组邻居特征向量,分别用 mean / max-pool / order-sensitive 三种
// 聚合器计算,画在 2D 平面上(x/y 是特征的两个维度)。
// order-sensitive 的结果会随 neighborIds 顺序变化,mean/max 不会。

export function AggregatorCompareWidget({ neighborIds }: Props) {
  const vectors = neighborIds.map((id) => NODE_FEATURES[id]);
  const mean = meanAggregate(vectors);
  const maxp = maxPoolAggregate(vectors);
  const order = orderSensitiveAggregate(vectors);

  const scale = 260;
  const originX = 60;
  const originY = H - 50;
  const toXY = (v: [number, number]) => [originX + v[0] * scale, originY - v[1] * scale];

  const points: Array<{ v: [number, number]; label: string; color: string }> = [
    ...vectors.map((v, i) => ({ v, label: `n${neighborIds[i]}`, color: "var(--ink-muted)" })),
    { v: mean, label: "mean", color: "#3b82f6" },
    { v: maxp, label: "max", color: "#10b981" },
    { v: order, label: "order-sensitive", color: "#ec4899" },
  ];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="三种聚合器输出对比">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        同一组邻居,三种聚合器的输出位置
      </text>

      <line x1={originX} y1={originY} x2={originX + scale + 20} y2={originY} stroke="var(--border)" />
      <line x1={originX} y1={originY} x2={originX} y2={originY - scale - 20} stroke="var(--border)" />

      {points.map((p, idx) => {
        const [x, y] = toXY(p.v);
        const isAgg = idx >= vectors.length;
        return (
          <g key={idx}>
            <circle cx={x} cy={y} r={isAgg ? 7 : 5} fill={p.color} opacity={isAgg ? 1 : 0.6} />
            <text x={x + 8} y={y + 4} fontSize={10} fontWeight={isAgg ? 700 : 400} fill={p.color}>
              {p.label}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        灰点 = 各邻居原始特征 · 蓝/绿/粉 = mean / max-pool / order-sensitive 聚合结果
      </text>
    </svg>
  );
}
