const W = 700;
const H = 320;

// 8 层 AlexNet 的双 GPU 切分:大部分层各 GPU 独立,只在 conv3/fc 处跨卡通信
// 上下两条 lane:GPU0 (顶) + GPU1 (底)

interface Layer {
  name: string;
  shared: boolean;  // 是否跨 GPU 通信
  shape: string;
}

const LAYERS: Layer[] = [
  { name: "Input",  shared: false, shape: "224·3"     },
  { name: "Conv1",  shared: false, shape: "55·48"    },
  { name: "Conv2",  shared: false, shape: "27·128"   },
  { name: "Conv3",  shared: true,  shape: "13·192"   },
  { name: "Conv4",  shared: false, shape: "13·192"   },
  { name: "Conv5",  shared: false, shape: "13·128"   },
  { name: "FC6",    shared: true,  shape: "2048"     },
  { name: "FC7",    shared: true,  shape: "2048"     },
  { name: "FC8",    shared: true,  shape: "500"      },
];

export function DualGpuSplit() {
  const PAD = 30;
  const COL_W = (W - PAD * 2) / LAYERS.length;
  const LANE_GPU0_Y = 70;
  const LANE_GPU1_Y = 190;
  const BOX_H = 38;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="AlexNet dual-GPU split architecture">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        双 GPU 切分 — 通道维劈成两半,只在 conv3 / fc 跨卡通信
      </text>

      {/* GPU lane labels */}
      <text x={PAD - 6} y={LANE_GPU0_Y + BOX_H / 2 + 4} textAnchor="end" fontSize={11} fontWeight={700} fill="#ec4899">GPU 0</text>
      <text x={PAD - 6} y={LANE_GPU1_Y + BOX_H / 2 + 4} textAnchor="end" fontSize={11} fontWeight={700} fill="#3b82f6">GPU 1</text>

      {/* lane background */}
      <rect x={PAD} y={LANE_GPU0_Y - 8} width={W - PAD * 2} height={BOX_H + 16} fill="#fce7f3" opacity={0.15} rx={6} />
      <rect x={PAD} y={LANE_GPU1_Y - 8} width={W - PAD * 2} height={BOX_H + 16} fill="#dbeafe" opacity={0.25} rx={6} />

      {LAYERS.map((L, i) => {
        const x = PAD + i * COL_W + 4;
        const w = COL_W - 8;
        // shared layers: connect both lanes via crossing lines
        return (
          <g key={i}>
            <rect x={x} y={LANE_GPU0_Y} width={w} height={BOX_H} rx={4}
              fill={L.shared ? "#fef3c7" : "#fce7f3"} stroke={L.shared ? "#f59e0b" : "#ec4899"} strokeWidth={1.4} />
            <text x={x + w / 2} y={LANE_GPU0_Y + 16} textAnchor="middle" fontSize={10} fontWeight={600} fill="#1f2937">{L.name}</text>
            <text x={x + w / 2} y={LANE_GPU0_Y + 30} textAnchor="middle" fontSize={9} fill="#6b7280">{L.shape}</text>

            <rect x={x} y={LANE_GPU1_Y} width={w} height={BOX_H} rx={4}
              fill={L.shared ? "#fef3c7" : "#dbeafe"} stroke={L.shared ? "#f59e0b" : "#3b82f6"} strokeWidth={1.4} />
            <text x={x + w / 2} y={LANE_GPU1_Y + 16} textAnchor="middle" fontSize={10} fontWeight={600} fill="#1f2937">{L.name}</text>
            <text x={x + w / 2} y={LANE_GPU1_Y + 30} textAnchor="middle" fontSize={9} fill="#6b7280">{L.shape}</text>

            {/* cross link if shared */}
            {L.shared && (
              <>
                <line x1={x + w / 2} y1={LANE_GPU0_Y + BOX_H} x2={x + w / 2} y2={LANE_GPU1_Y} stroke="#f59e0b" strokeWidth={1.8} strokeDasharray="4 3" />
                <text x={x + w / 2} y={(LANE_GPU0_Y + BOX_H + LANE_GPU1_Y) / 2 + 4} textAnchor="middle" fontSize={9} fontWeight={700} fill="#92400e">⇅</text>
              </>
            )}

            {/* sequential arrow within lane */}
            {i < LAYERS.length - 1 && (
              <>
                <line x1={x + w + 1} y1={LANE_GPU0_Y + BOX_H / 2} x2={x + w + COL_W - 8 + 4 - 1} y2={LANE_GPU0_Y + BOX_H / 2} stroke="#9ca3af" strokeWidth={1} />
                <line x1={x + w + 1} y1={LANE_GPU1_Y + BOX_H / 2} x2={x + w + COL_W - 8 + 4 - 1} y2={LANE_GPU1_Y + BOX_H / 2} stroke="#9ca3af" strokeWidth={1} />
              </>
            )}
          </g>
        );
      })}

      {/* legend */}
      <g transform={`translate(${PAD}, ${H - 50})`}>
        <rect x={0} y={0} width={14} height={14} fill="#fce7f3" stroke="#ec4899" />
        <text x={20} y={11} fontSize={10} fill="#374151">GPU 私有(通道维只一半)</text>
        <rect x={170} y={0} width={14} height={14} fill="#fef3c7" stroke="#f59e0b" />
        <text x={190} y={11} fontSize={10} fill="#374151">跨 GPU 共享(全通道)</text>
      </g>

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        2× GTX 580 (3 GB / 块) — 显存约束驱动的工程妥协,后来演化成 "group convolution"
      </text>
    </svg>
  );
}
