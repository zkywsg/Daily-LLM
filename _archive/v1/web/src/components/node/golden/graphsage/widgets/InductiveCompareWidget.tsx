import {
  GRAPH_B_NEW_NODE,
  GRAPH_B_NEW_NODE_FEATURE,
  GRAPH_B_NEW_NODE_NEIGHBORS,
  GRAPH_B_NEIGHBOR_FEATURES,
  meanAggregate,
} from "../lib/data";

// 图 B 里的新节点(id=100)在"训练时"根本不存在。
// GraphSAGE 的聚合函数(这里用 mean 举例)不依赖任何"记住哪个节点是哪个"
// 的查表操作,纯粹是邻居特征的函数 —— 所以可以直接对这个新节点算出 embedding。
// GCN 做不到:它的传播矩阵是针对固定邻接矩阵 A 求逆/归一化的,换一张图(哪怕
// 只加一个节点)整个矩阵都要重新定义。

export function InductiveCompareWidget() {
  const neighborVectors = GRAPH_B_NEW_NODE_NEIGHBORS.map((id) => GRAPH_B_NEIGHBOR_FEATURES[id]);
  const newEmbedding = meanAggregate(neighborVectors);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-4)" }}>
      <div style={{ padding: "var(--space-4)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-surface)" }}>
        <div style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginBottom: 6 }}>
          图 B 里训练时从未出现过的新节点
        </div>
        <div style={{ fontSize: "var(--fs-md)", fontWeight: 600 }}>
          节点 #{GRAPH_B_NEW_NODE},原始特征 = [{GRAPH_B_NEW_NODE_FEATURE.join(", ")}]
        </div>
        <div style={{ fontSize: "var(--fs-sm)", color: "var(--ink-secondary)", marginTop: 4 }}>
          邻居:{GRAPH_B_NEW_NODE_NEIGHBORS.map((id) => `#${id}[${GRAPH_B_NEIGHBOR_FEATURES[id].join(",")}]`).join(" · ")}
        </div>
      </div>

      <div style={{ padding: "var(--space-4)", border: "1px solid #ec4899", borderRadius: "var(--radius-md)", background: "#fce7f3" }}>
        <div style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", marginBottom: 6 }}>
          直接套用同一个(训练好的)mean 聚合函数
        </div>
        <div style={{ fontSize: "var(--fs-lg)", fontWeight: 700, color: "#9d174d" }}>
          embedding(#{GRAPH_B_NEW_NODE}) = mean(邻居特征) = [{newEmbedding[0].toFixed(2)}, {newEmbedding[1].toFixed(2)}]
        </div>
      </div>

      <p style={{ fontSize: "var(--fs-sm)", color: "var(--ink-muted)", lineHeight: 1.6 }}>
        这一步没有查任何"节点 #{GRAPH_B_NEW_NODE} 的专属参数"——聚合函数只认邻居的特征向量,不认节点 id。
        这正是"归纳式(inductive)"的含义:同一套函数可以直接应用到任意新图、新节点。
        GCN 的 D̃^(-1/2)ÃD̃^(-1/2) 是针对固定图算出来的一个具体矩阵,图变了矩阵就要重新算,没法直接套用到没见过的节点上——这是"直推式(transductive)"的局限。
      </p>
    </div>
  );
}
