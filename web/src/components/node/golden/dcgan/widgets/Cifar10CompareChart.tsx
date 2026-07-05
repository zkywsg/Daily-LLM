import { CIFAR10_COMPARE } from "../lib/data";

const W = 700;
const H = 220;

export function Cifar10CompareChart() {
  const PAD_L = 180;
  const PAD_T = 40;
  const barMaxW = 420;
  const rowH = 40;
  const maxAcc = 90;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="CIFAR-10 上 DCGAN 特征 + L2-SVM 与其他方法的分类准确率对比"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        CIFAR-10 分类准确率 — DCGAN 的 D 当特征提取器 + L2-SVM
      </text>

      {CIFAR10_COMPARE.map((row, i) => {
        const y = PAD_T + i * rowH;
        const w = (row.acc / maxAcc) * barMaxW;
        const fill = row.highlight ? "#f59e0b" : "#9ca3af";
        const bg = row.highlight ? "#fef3c7" : "#f3f4f6";
        return (
          <g key={row.model}>
            <text x={PAD_L - 10} y={y + 16} textAnchor="end" fontSize={10} fill="var(--ink-secondary)">
              {row.model}
            </text>
            <rect x={PAD_L} y={y} width={barMaxW} height={22} fill={bg} rx={3} />
            <rect x={PAD_L} y={y} width={w} height={22} fill={fill} rx={3} />
            <text x={PAD_L + w + 8} y={y + 16} fontSize={11} fontWeight={700} fill="var(--ink-primary)">
              {row.acc}%
            </text>
          </g>
        );
      })}
    </svg>
  );
}
