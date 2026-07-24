import { useState } from "react";
import { categoricalLatent } from "../lib/data";

export function CategoricalLatentWidget() {
  const [numCategoricals, setNumCategoricals] = useState(4);
  const numClasses = 8;
  const dists = categoricalLatent(numCategoricals, numClasses);

  return (
    <div>
      <label style={{ display: "block", fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginBottom: "var(--space-3)" }}>
        类别变量组数 = {numCategoricals}(每组 {numClasses} 类)
        <input type="range" min={1} max={8} value={numCategoricals} onChange={(e) => setNumCategoricals(Number(e.target.value))} style={{ display: "block", width: "100%", marginTop: 6 }} />
      </label>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {dists.map((dist, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ width: 50, fontSize: "var(--fs-xs)", color: "var(--ink-muted)" }}>组 {i}</span>
            <div style={{ display: "flex", gap: 1, flex: 1 }}>
              {dist.map((p, k) => (
                <div key={k} title={p.toFixed(2)} style={{ height: 20, flex: 1, background: "#d946ef", opacity: 0.2 + p * 3 }} />
              ))}
            </div>
          </div>
        ))}
      </div>
      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginTop: "var(--space-3)" }}>
        每组是一个 {numClasses} 类的离散分布(颜色深浅表示概率)。组数越多,隐状态能表达的组合数越多(numClasses^numCategoricals),表达力呈指数增长。
      </p>
    </div>
  );
}
