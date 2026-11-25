import { TRAINING_RECIPE_COMPARE } from "../lib/data";

export function TrainingRecipeCompareTable() {
  return (
    <div style={{ overflowX: "auto" }}>
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: "var(--fs-sm)",
          fontFamily: "system-ui",
        }}
        role="table"
        aria-label="ResNet-50 原版训练 recipe 与 ConvNeXt 训练 recipe 对比"
      >
        <thead>
          <tr>
            <th style={thStyle}>维度</th>
            <th style={{ ...thStyle, color: "#9ca3af" }}>ResNet-50 原版(2015)</th>
            <th style={{ ...thStyle, color: "#ec4899" }}>ConvNeXt(2022)</th>
          </tr>
        </thead>
        <tbody>
          {TRAINING_RECIPE_COMPARE.map((row, i) => (
            <tr key={row.dim} style={{ background: i % 2 === 0 ? "transparent" : "var(--bg-surface)" }}>
              <td style={{ ...tdStyle, fontWeight: 700, color: "var(--ink-primary)" }}>{row.dim}</td>
              <td style={{ ...tdStyle, color: "var(--ink-muted)", background: "#f3f4f6" }}>{row.oldVal}</td>
              <td style={{ ...tdStyle, color: "var(--ink-primary)", background: "#fce7f3", fontWeight: 600 }}>
                {row.newVal}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const thStyle: React.CSSProperties = {
  textAlign: "left",
  padding: "8px 10px",
  borderBottom: "2px solid var(--border)",
  fontSize: "var(--fs-xs)",
  textTransform: "uppercase",
  letterSpacing: "0.02em",
};

const tdStyle: React.CSSProperties = {
  padding: "7px 10px",
  borderBottom: "1px solid var(--border)",
  lineHeight: 1.5,
};
