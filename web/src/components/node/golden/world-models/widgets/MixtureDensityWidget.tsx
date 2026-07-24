import { useState } from "react";
import { predictMixture, sampleMixtureMean } from "../lib/data";

const W = 680;
const H = 280;

// 给定当前 z0,用 K 个高斯分量可视化 MDN-RNN 预测的下一状态分布——
// 混合分量数越多,能表达的"下一步可能走向"越多样(多峰)。

export function MixtureDensityWidget() {
  const [k, setK] = useState(3);
  const z0 = 0.2;
  const comps = predictMixture(z0, k);
  const mean = sampleMixtureMean(comps);

  const xMin = -1.5, xMax = 1.5;
  const toX = (v: number) => ((v - xMin) / (xMax - xMin)) * (W - 60) + 30;
  const points = Array.from({ length: 200 }, (_, i) => xMin + (i / 199) * (xMax - xMin));
  const density = points.map((x) =>
    comps.reduce((s, c) => s + c.weight * Math.exp(-((x - c.mean) ** 2) / (2 * c.std ** 2)) / (c.std * Math.sqrt(2 * Math.PI)), 0)
  );
  const maxD = Math.max(...density, 0.1);
  const toY = (d: number) => H - 40 - Math.min((d / maxD) * (H - 80), H - 80);

  const path = points.map((x, i) => `${i === 0 ? "M" : "L"} ${toX(x)} ${toY(density[i])}`).join(" ");

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        混合分量数 K = {k}
        <input type="range" min={1} max={6} value={k} onChange={(e) => setK(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`K=${k} 个高斯分量混合的下一状态预测分布`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          下一状态 z' 的预测分布(K={k} 个高斯分量混合)
        </text>
        <line x1={30} y1={H - 40} x2={W - 30} y2={H - 40} stroke="var(--border)" />
        <path d={path} fill="none" stroke="#d946ef" strokeWidth={2} />
        {comps.map((c, i) => (
          <circle key={i} cx={toX(c.mean)} cy={H - 40} r={3 + c.weight * 10} fill="#d946ef" opacity={0.5} />
        ))}
        <line x1={toX(mean)} y1={30} x2={toX(mean)} y2={H - 40} stroke="var(--ink-muted)" strokeDasharray="3 3" />
        <text x={toX(mean)} y={26} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">期望值</text>
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        K 越大,分布能表达的"下一步走向"越多样(多峰);K=1 退化成单一高斯,只能预测一种确定性走向。
      </p>
    </div>
  );
}
