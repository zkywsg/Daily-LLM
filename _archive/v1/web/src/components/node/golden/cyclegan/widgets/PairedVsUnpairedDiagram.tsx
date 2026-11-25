interface Props {
  mode: "paired" | "unpaired";
}

const W = 700;
const H = 280;

export function PairedVsUnpairedDiagram({ mode }: Props) {
  const isPaired = mode === "paired";

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="pix2pix 配对数据 vs CycleGAN 无配对数据对比"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {isPaired ? "pix2pix — 需要严格配对数据 {(x_i, y_i)}" : "CycleGAN — 无需配对,两个独立域集合"}
      </text>

      {isPaired ? (
        <g>
          {[0, 1, 2].map((i) => {
            const y = 60 + i * 65;
            return (
              <g key={i}>
                <rect x={120} y={y} width={80} height={45} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.4} rx={5} />
                <text x={160} y={y + 27} textAnchor="middle" fontSize={11} fill="var(--ink-primary)">x_{i + 1}</text>
                <line x1={200} y1={y + 22} x2={280} y2={y + 22} stroke="#10b981" strokeWidth={2} />
                <text x={240} y={y + 14} textAnchor="middle" fontSize={9} fill="#10b981">配对</text>
                <rect x={280} y={y} width={80} height={45} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.4} rx={5} />
                <text x={320} y={y + 27} textAnchor="middle" fontSize={11} fill="var(--ink-primary)">y_{i + 1}</text>
              </g>
            );
          })}
          <text x={W / 2} y={H - 30} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">
            每个 x_i 都必须有对应的 y_i(如同一姿态的边缘图↔真实照片)
          </text>
          <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
            现实中马↔斑马、莫奈画↔照片、夏↔冬 这类任务根本拿不到这种配对
          </text>
        </g>
      ) : (
        <g>
          {/* domain X set */}
          <rect x={80} y={50} width={220} height={180} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.4} rx={8} opacity={0.3} />
          <text x={190} y={40} textAnchor="middle" fontSize={11} fontWeight={700} fill="#3b82f6">域 X 集合(马,~1000 张)</text>
          {[0, 1, 2, 3].map((i) => (
            <rect
              key={i}
              x={100 + (i % 2) * 100}
              y={70 + Math.floor(i / 2) * 70}
              width={80}
              height={45}
              fill="#dbeafe"
              stroke="#3b82f6"
              strokeWidth={1.2}
              rx={4}
            />
          ))}

          {/* domain Y set */}
          <rect x={400} y={50} width={220} height={180} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.4} rx={8} opacity={0.3} />
          <text x={510} y={40} textAnchor="middle" fontSize={11} fontWeight={700} fill="#ec4899">域 Y 集合(斑马,~1000 张)</text>
          {[0, 1, 2, 3].map((i) => (
            <rect
              key={i}
              x={420 + (i % 2) * 100}
              y={70 + Math.floor(i / 2) * 70}
              width={80}
              height={45}
              fill="#fce7f3"
              stroke="#ec4899"
              strokeWidth={1.2}
              rx={4}
            />
          ))}

          <text x={350} y={140} textAnchor="middle" fontSize={20} fill="var(--ink-muted)">≠</text>
          <text x={W / 2} y={H - 30} textAnchor="middle" fontSize={11} fill="var(--ink-secondary)">
            X 和 Y 是两个独立集合,没有一一对应关系,靠 cycle loss 学习映射
          </text>
          <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
            这是 CycleGAN 能处理马↔斑马、莫奈↔照片、夏↔冬 的关键前提
          </text>
        </g>
      )}
    </svg>
  );
}
