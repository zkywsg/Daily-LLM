import { EXPERT_PARALLELISM } from "../lib/data";

const W = 700;
const H = 340;

// Expert parallelism 简化示意:2048 个 expert 分布在 128 块 GPU 上(每 GPU 16 个),
// 这里只画 8 个 GPU 的缩略图,每个 GPU 画几个 expert 方块,加上 all-to-all 路由弧线。

const NUM_GPU = 8;
const EXPERTS_SHOWN_PER_GPU = 4; // 缩略展示,真实是 16 个

export function ExpertParallelismDiagram() {
  const cx = W / 2;
  const cy = H / 2 + 6;
  const r = 118;

  const gpuPositions = Array.from({ length: NUM_GPU }, (_, i) => {
    const angle = (i / NUM_GPU) * 2 * Math.PI - Math.PI / 2;
    return {
      x: cx + r * Math.cos(angle),
      y: cy + r * Math.sin(angle),
      label: `GPU ${i}`,
    };
  });

  const TOKEN_ROUTES = [
    { token: "T1", from: 0, to: [3, 5] },
    { token: "T2", from: 1, to: [2, 6] },
    { token: "T3", from: 7, to: [0, 4] },
  ];

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="Expert parallelism across GPUs with all-to-all token routing"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        Expert Parallelism —— {EXPERT_PARALLELISM.numGpus} 块 GPU,每块装 {EXPERT_PARALLELISM.expertsPerGpu} 个 expert
      </text>
      <text x={W / 2} y={36} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        (下图缩略展示 {NUM_GPU} 块 GPU,每块 {EXPERTS_SHOWN_PER_GPU} 个 expert;真实规模是 2048 个 expert / 128 GPU)
      </text>

      {TOKEN_ROUTES.map((rt) =>
        rt.to.map((dst) => (
          <line
            key={`${rt.token}-${dst}`}
            x1={gpuPositions[rt.from].x}
            y1={gpuPositions[rt.from].y}
            x2={gpuPositions[dst].x}
            y2={gpuPositions[dst].y}
            stroke="#ec4899"
            strokeWidth={1.5}
            strokeDasharray="3 3"
            opacity={0.55}
          />
        )),
      )}

      {gpuPositions.map((p, i) => (
        <g key={i}>
          <rect x={p.x - 34} y={p.y - 26} width={68} height={52} rx={6} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.4} />
          <text x={p.x} y={p.y - 10} textAnchor="middle" fontSize={10} fontWeight={700} fill="#1e3a8a">
            GPU {i}
          </text>
          {Array.from({ length: EXPERTS_SHOWN_PER_GPU }, (_, k) => (
            <rect
              key={k}
              x={p.x - 28 + (k % 4) * 14}
              y={p.y + 0}
              width={10}
              height={10}
              rx={2}
              fill="#3b82f6"
              opacity={0.7}
            />
          ))}
          <text x={p.x} y={p.y + 24} textAnchor="middle" fontSize={8} fill="#3b82f6">
            16 experts
          </text>
        </g>
      ))}

      <circle cx={cx} cy={cy} r={30} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.5} />
      <text x={cx} y={cy - 2} textAnchor="middle" fontSize={10} fontWeight={700} fill="#831843">
        all-to-all
      </text>
      <text x={cx} y={cy + 11} textAnchor="middle" fontSize={8} fill="#831843">
        通信
      </text>

      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        每 GPU 先算 gating → all-to-all 把 token 发到目标 expert 所在 GPU → 本地计算 → 反向 all-to-all 收回原 GPU
      </text>
    </svg>
  );
}
