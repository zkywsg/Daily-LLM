import { TOY_POINTS, initialCenters, assignClusters } from "../lib/data";

const W = 320;
const H = 320;
const PALETTE = ["#fb7185", "#f59e0b", "#8b5cf6"];

export function ClusteringWidget() {
  const centers = initialCenters();
  const assignments = assignClusters(TOY_POINTS, centers);

  const toXY = (p: [number, number]) => [W / 2 + p[0] * 100, H / 2 - p[1] * 100];

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="k-means 聚类分配可视化">
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          离线 k-means 聚类:每个特征点被分配到最近的聚类中心
        </text>
        {centers.map((c, k) => {
          const [x, y] = toXY(c);
          return <circle key={k} cx={x} cy={y} r={10} fill="none" stroke={PALETTE[k]} strokeWidth={3} />;
        })}
        {TOY_POINTS.map((p, i) => {
          const [x, y] = toXY(p);
          const k = assignments[i];
          return <circle key={i} cx={x} cy={y} r={7} fill={PALETTE[k]} />;
        })}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        实心点是特征点,颜色代表分配到的聚类;空心圆环是聚类中心。这一步产生的聚类标签就是 HuBERT 第一轮训练用的离散伪标签。
      </p>
    </div>
  );
}
