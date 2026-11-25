import { FULL_FT_VS_PREFIX } from "../lib/data";

export function FullFtVsPrefixCompareTable() {
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "var(--fs-sm)" }}>
        <thead>
          <tr style={{ borderBottom: "2px solid var(--border)" }}>
            <th style={{ textAlign: "left", padding: "8px 10px", color: "var(--ink-muted)", fontWeight: 700 }}>维度</th>
            <th style={{ textAlign: "left", padding: "8px 10px", color: "#6b7280", fontWeight: 700 }}>Full FT</th>
            <th style={{ textAlign: "left", padding: "8px 10px", color: "#be185d", fontWeight: 700 }}>Prefix Tuning</th>
          </tr>
        </thead>
        <tbody>
          {FULL_FT_VS_PREFIX.map((row, i) => (
            <tr key={`${row.dimension}-${i}`} style={{ borderBottom: "1px solid var(--border)" }}>
              <td style={{ padding: "8px 10px", fontWeight: 600, color: "var(--ink-primary)" }}>{row.dimension}</td>
              <td style={{ padding: "8px 10px", color: "var(--ink-secondary)" }}>{row.fullFt}</td>
              <td style={{ padding: "8px 10px", color: "var(--ink-secondary)" }}>{row.prefixTuning}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
