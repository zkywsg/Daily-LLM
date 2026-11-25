const W = 700;
const H = 340;

// Expert parallelism: 8 个 GPU 各放 1 个 expert,token 跨 GPU all-to-all 通信。
// 画 8 个 GPU 节点 + 几个 token 的路由路径(跨 GPU)。

const NUM_GPU = 8;

export function ExpertParallelism() {
  const cx = W / 2;
  const cy = H / 2 + 10;
  const r = 110;

  const gpuPositions = Array.from({ length: NUM_GPU }, (_, i) => {
    const angle = (i / NUM_GPU) * 2 * Math.PI - Math.PI / 2;
    return {
      x: cx + r * Math.cos(angle),
      y: cy + r * Math.sin(angle),
      label: `GPU ${i}`,
      expert: i,
    };
  });

  // 几个 token 示例,每个 token 路由到 2 个 GPU
  const TOKEN_ROUTES = [
    { token: "T1", from: 0, to: [3, 5] },
    { token: "T2", from: 1, to: [2, 7] },
    { token: "T3", from: 6, to: [0, 4] },
  ];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Expert parallelism: 8 GPUs each hosts one expert with all-to-all communication">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Expert Parallelism — 8 GPU 各放 1 expert · token 跨 GPU all-to-all
      </text>

      {/* 路由弧线 */}
      {TOKEN_ROUTES.map((r) =>
        r.to.map((dst) => (
          <line
            key={`${r.token}-${dst}`}
            x1={gpuPositions[r.from].x}
            y1={gpuPositions[r.from].y}
            x2={gpuPositions[dst].x}
            y2={gpuPositions[dst].y}
            stroke="#ec4899"
            strokeWidth={1.5}
            strokeDasharray="3 3"
            opacity={0.55}
          />
        )),
      )}

      {/* GPU 节点 */}
      {gpuPositions.map((p, i) => (
        <g key={i}>
          <circle cx={p.x} cy={p.y} r={26} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.5} />
          <text x={p.x} y={p.y - 2} textAnchor="middle" fontSize={11} fontWeight={700} fill="#1e3a8a">
            GPU{i}
          </text>
          <text x={p.x} y={p.y + 12} textAnchor="middle" fontSize={9} fill="#3b82f6">
            E{p.expert}
          </text>
        </g>
      ))}

      {/* 中心:router */}
      <circle cx={cx} cy={cy} r={32} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.5} />
      <text x={cx} y={cy - 2} textAnchor="middle" fontSize={11} fontWeight={700} fill="#831843">
        Router
      </text>
      <text x={cx} y={cy + 12} textAnchor="middle" fontSize={9} fill="#831843">
        per-layer
      </text>

      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        每 token 选 top-2 GPU 跑 expert → all-to-all 把结果送回原 GPU 加权求和
      </text>
    </svg>
  );
}
