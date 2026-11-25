import { EDGES, NODES, POSITIONS, degree } from "../lib/data";

interface Props {
  withSelfLoop: boolean;
}

const W = 680;
const H = 360;

// 6 节点 toy 图,切换 "原始邻接 A" vs "加自环 Ã = A + I"。
// 打开自环时每个节点旁边画一个小圆环,并在节点内显示更新后的度数。

export function GraphSelfLoopWidget({ withSelfLoop }: Props) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label={`GCN toy 图,自环${withSelfLoop ? "已开启" : "未开启"}`}>
      <text x={W / 2} y={24} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        {withSelfLoop ? "Ã = A + I(每个节点加一条指向自己的边)" : "原始邻接矩阵 A(只有真实的边)"}
      </text>

      {EDGES.map((e, idx) => {
        const [x1, y1] = POSITIONS[e.a];
        const [x2, y2] = POSITIONS[e.b];
        return <line key={idx} x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--border)" strokeWidth={2} />;
      })}

      {NODES.map((n) => {
        const [x, y] = POSITIONS[n];
        return (
          <g key={n}>
            {withSelfLoop && (
              <circle cx={x + 26} cy={y - 26} r={14} fill="none" stroke="#ec4899" strokeWidth={2} strokeDasharray="3 2" />
            )}
            <circle cx={x} cy={y} r={22} fill="var(--bg-surface)" stroke="#ec4899" strokeWidth={2} />
            <text x={x} y={y + 5} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
              {n}
            </text>
            <text x={x} y={y + 40} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">
              deg={degree(n, withSelfLoop)}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        {withSelfLoop
          ? "每个节点度数都 +1 —— 后面聚合时,节点会把自己的旧特征也算进新特征里"
          : "此时聚合只用邻居信息,节点自身的旧特征在下一层会被完全覆盖"}
      </text>
    </svg>
  );
}
