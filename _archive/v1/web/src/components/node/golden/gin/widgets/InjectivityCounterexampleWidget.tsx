import { MULTISET_X, MULTISET_Y, mean, max, sum } from "../lib/data";
import { MultisetDisplayWidget } from "./MultisetDisplayWidget";

// 两个不同的邻居多重集,mean/max 塌缩成相同结果,sum 保留了区别。

export function InjectivityCounterexampleWidget() {
  const rows: Array<{ name: string; fn: (xs: number[]) => number }> = [
    { name: "mean", fn: mean },
    { name: "max", fn: max },
    { name: "sum", fn: sum },
  ];

  return (
    <div>
      <MultisetDisplayWidget label="多重集 X" values={MULTISET_X} color="#3b82f6" />
      <MultisetDisplayWidget label="多重集 Y" values={MULTISET_Y} color="#f59e0b" />

      <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "var(--space-4)" }}>
        <thead>
          <tr>
            <th style={{ textAlign: "left", fontSize: "var(--fs-sm)", color: "var(--ink-muted)", padding: "4px 8px" }}>聚合函数</th>
            <th style={{ textAlign: "center", fontSize: "var(--fs-sm)", color: "var(--ink-muted)", padding: "4px 8px" }}>结果(X)</th>
            <th style={{ textAlign: "center", fontSize: "var(--fs-sm)", color: "var(--ink-muted)", padding: "4px 8px" }}>结果(Y)</th>
            <th style={{ textAlign: "center", fontSize: "var(--fs-sm)", color: "var(--ink-muted)", padding: "4px 8px" }}>能否区分</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ name, fn }) => {
            const rx = fn(MULTISET_X);
            const ry = fn(MULTISET_Y);
            const distinguishable = rx !== ry;
            return (
              <tr key={name} style={{ borderTop: "1px solid var(--border)" }}>
                <td style={{ padding: "6px 8px", fontWeight: 600 }}>{name}</td>
                <td style={{ padding: "6px 8px", textAlign: "center" }}>{rx}</td>
                <td style={{ padding: "6px 8px", textAlign: "center" }}>{ry}</td>
                <td style={{ padding: "6px 8px", textAlign: "center", color: distinguishable ? "#059669" : "#dc2626", fontWeight: 700 }}>
                  {distinguishable ? "✓ 能区分" : "✗ 塌缩成相同值"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
