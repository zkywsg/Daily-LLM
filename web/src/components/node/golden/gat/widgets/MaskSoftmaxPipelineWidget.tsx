import { NODES, attentionLogit, softmax, rawNeighbors } from "../lib/data";

interface Props {
  center: number;
}

// 三阶段对比:①对全部节点算出的原始 logits(含非邻居)→ ②只保留邻居的 mask
// 后 logits(非邻居直接置为 "-∞"/隐藏)→ ③softmax 归一化后的最终权重。

export function MaskSoftmaxPipelineWidget({ center }: Props) {
  const neighborSet = new Set(rawNeighbors(center));
  const rawLogits = NODES.filter((n) => n !== center).map((n) => ({ n, logit: attentionLogit(center, n, 1) }));
  const maskedNeighbors = rawLogits.filter((x) => neighborSet.has(x.n));
  const weights = softmax(maskedNeighbors.map((x) => x.logit));

  const rowStyle: React.CSSProperties = { display: "flex", gap: 8, alignItems: "center", padding: "4px 0" };
  const cellStyle = (active: boolean): React.CSSProperties => ({
    width: 60, padding: "3px 6px", borderRadius: "var(--radius-sm)", textAlign: "center", fontSize: "var(--fs-sm)",
    background: active ? "#fce7f3" : "var(--bg-subtle)", color: active ? "#9d174d" : "var(--ink-muted)",
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
      <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
        ① 全部节点的原始 logits
      </div>
      {rawLogits.map(({ n, logit }) => (
        <div key={n} style={rowStyle}>
          <span style={{ width: 50, fontSize: "var(--fs-sm)" }}>节点 {n}</span>
          <span style={cellStyle(neighborSet.has(n))}>{logit.toFixed(2)}</span>
          <span style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)" }}>{neighborSet.has(n) ? "是邻居" : "非邻居(将被屏蔽)"}</span>
        </div>
      ))}

      <div style={{ fontSize: "var(--fs-xs)", color: "var(--ink-muted)", textTransform: "uppercase", letterSpacing: "0.05em", marginTop: "var(--space-3)" }}>
        ② mask 后只剩邻居 → ③ softmax 归一化权重
      </div>
      {maskedNeighbors.map(({ n }, idx) => (
        <div key={n} style={rowStyle}>
          <span style={{ width: 50, fontSize: "var(--fs-sm)" }}>节点 {n}</span>
          <span style={cellStyle(true)}>{weights[idx].toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}
