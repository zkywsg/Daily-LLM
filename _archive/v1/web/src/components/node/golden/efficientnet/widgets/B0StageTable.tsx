import { B0_STAGES } from "../lib/data";

export function B0StageTable() {
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
        aria-label="EfficientNet-B0 的 7 个 NAS 搜索 stage 配置"
      >
        <thead>
          <tr>
            <th style={thStyle}>Stage</th>
            <th style={thStyle}>Block</th>
            <th style={thStyle}>Kernel</th>
            <th style={thStyle}>通道</th>
            <th style={thStyle}>Block 数</th>
            <th style={thStyle}>Stride</th>
          </tr>
        </thead>
        <tbody>
          {B0_STAGES.map((row, i) => (
            <tr key={row.stage} style={{ background: i % 2 === 0 ? "transparent" : "var(--bg-surface)" }}>
              <td style={{ ...tdStyle, fontWeight: 700, color: "var(--ink-primary)" }}>{row.stage}</td>
              <td style={tdStyle}>{row.block}</td>
              <td style={tdStyle}>{row.kernel}</td>
              <td style={tdStyle}>{row.channels}</td>
              <td style={tdStyle}>{row.numBlocks}</td>
              <td style={tdStyle}>{row.stride}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const thStyle: React.CSSProperties = {
  textAlign: "left",
  padding: "6px 8px",
  borderBottom: "2px solid var(--border)",
  fontSize: "var(--fs-xs)",
  textTransform: "uppercase",
  letterSpacing: "0.02em",
};

const tdStyle: React.CSSProperties = {
  padding: "6px 8px",
  borderBottom: "1px solid var(--border)",
  lineHeight: 1.5,
};
