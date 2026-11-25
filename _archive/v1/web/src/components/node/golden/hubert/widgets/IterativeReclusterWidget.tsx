import { useState } from "react";
import { TOY_POINTS, NUM_CLUSTERS, initialCenters, assignClusters, updateCenters } from "../lib/data";

const W = 320;
const H = 320;
const PALETTE = ["#fb7185", "#f59e0b", "#8b5cf6"];

// 迭代式重新聚类:每点一次"跑下一轮",用当前分配重新计算聚类中心(Lloyd's
// 算法一步更新),再重新分配——聚类边界逐轮收敛,模拟 HuBERT 用模型自身
// 隐藏层特征重新聚类、提纯伪标签的过程。

export function IterativeReclusterWidget() {
  const [round, setRound] = useState(0);
  const [centers, setCenters] = useState(initialCenters());
  const [converged, setConverged] = useState(false);

  const assignments = assignClusters(TOY_POINTS, centers);
  const toXY = (p: [number, number]) => [W / 2 + p[0] * 100, H / 2 - p[1] * 100];

  const nextRound = () => {
    const newCenters = updateCenters(TOY_POINTS, assignments);
    const isSame = newCenters.every(
      (c, k) => Math.abs(c[0] - centers[k][0]) < 1e-9 && Math.abs(c[1] - centers[k][1]) < 1e-9
    );
    if (isSame) {
      setConverged(true);
      return;
    }
    setCenters(newCenters);
    setRound((r) => r + 1);
  };

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`迭代重聚类第 ${round} 轮`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          迭代式重新聚类 — 第 {round} 轮
        </text>
        {centers.map((c, k) => {
          const [x, y] = toXY(c);
          return <circle key={k} cx={x} cy={y} r={10} fill="none" stroke={PALETTE[k % NUM_CLUSTERS]} strokeWidth={3} />;
        })}
        {TOY_POINTS.map((p, i) => {
          const [x, y] = toXY(p);
          const k = assignments[i];
          return <circle key={i} cx={x} cy={y} r={7} fill={PALETTE[k % NUM_CLUSTERS]} />;
        })}
      </svg>
      <div style={{ display: "flex", gap: 8, marginTop: "var(--space-2)" }}>
        <button
          type="button" onClick={nextRound} disabled={round >= 4 || converged}
          style={{ padding: "4px 14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: round >= 4 || converged ? "not-allowed" : "pointer", fontSize: "var(--fs-sm)", opacity: round >= 4 || converged ? 0.5 : 1 }}
        >
          {converged ? "已收敛,无需继续" : "跑下一轮重聚类"}
        </button>
        <button
          type="button" onClick={() => { setCenters(initialCenters()); setRound(0); setConverged(false); }}
          style={{ padding: "4px 14px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border)", background: "var(--bg-surface)", cursor: "pointer", fontSize: "var(--fs-sm)" }}
        >
          重置
        </button>
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-2)" }}>
        {converged
          ? "聚类中心已不再变化,Lloyd's 算法已收敛——这正是 k-means 的重要性质:迭代重聚类会稳定到一个固定点。"
          : "每一轮都用当前分配重新计算聚类中心(移动到各自簇内点的均值),边界逐轮收敛更稳定——这正是 HuBERT 用模型隐藏层特征重新聚类提纯伪标签的过程。"}
      </p>
    </div>
  );
}
