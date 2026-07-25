import { useState } from "react";
import { NUM_CLUSTERS, classDistribution } from "../lib/data";

const W = 500;
const H = 260;

export function ClassDistributionWidget() {
  const [sharpness, setSharpness] = useState(1);
  const trueClass = 1;
  const dist = classDistribution(trueClass, sharpness);

  const barW = (W - 80) / NUM_CLUSTERS - 10;

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        分类置信度(sharpness)= {sharpness.toFixed(1)}
        <input type="range" min={0} max={5} step={0.5} value={sharpness} onChange={(e) => setSharpness(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`真实类别 ${trueClass} 的掩码分类置信度分布`}>
        <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
          掩码分类头输出的类别概率分布(真实类别 = {trueClass})
        </text>
        <line x1={40} y1={H - 40} x2={W - 40} y2={H - 40} stroke="var(--border)" />
        {dist.map((p, k) => {
          const x = 60 + k * (barW + 30);
          const h = Math.min(p * (H - 100), H - 100);
          return (
            <g key={k}>
              <rect x={x} y={H - 40 - h} width={barW} height={h} fill={k === trueClass ? "#fb7185" : "#9ca3af"} />
              <text x={x + barW / 2} y={H - 40 - h - 6} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">{p.toFixed(2)}</text>
              <text x={x + barW / 2} y={H - 18} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">类 {k}</text>
            </g>
          );
        })}
      </svg>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)" }}>
        这是标准的分类任务(交叉熵),不是对比学习——sharpness 越大,模型对正确类别的置信度越高,分布越尖锐。
      </p>
    </div>
  );
}
