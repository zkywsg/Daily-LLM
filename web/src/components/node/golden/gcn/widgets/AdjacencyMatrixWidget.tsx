import { NODES, adjacencyCell } from "../lib/data";

interface Props {
  withSelfLoop: boolean;
}

// 6x6 邻接矩阵网格,加自环时对角线格子会从 0 变成 1(高亮)。

export function AdjacencyMatrixWidget({ withSelfLoop }: Props) {
  return (
    <div style={{ display: "inline-block" }}>
      <table style={{ borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={{ width: 28 }} />
            {NODES.map((j) => (
              <th key={j} style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", width: 32 }}>
                {j}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {NODES.map((i) => (
            <tr key={i}>
              <td style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", textAlign: "right", paddingRight: 6 }}>
                {i}
              </td>
              {NODES.map((j) => {
                const v = adjacencyCell(i, j, withSelfLoop);
                const isDiag = i === j;
                return (
                  <td
                    key={j}
                    style={{
                      width: 32,
                      height: 32,
                      textAlign: "center",
                      border: "1px solid var(--border)",
                      background: v ? (isDiag ? "#ec4899" : "var(--bg-subtle)") : "var(--bg-surface)",
                      color: v && isDiag ? "#fff" : "var(--ink-primary)",
                      fontWeight: isDiag ? 700 : 400,
                      fontSize: "var(--fs-sm)",
                    }}
                  >
                    {v}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
