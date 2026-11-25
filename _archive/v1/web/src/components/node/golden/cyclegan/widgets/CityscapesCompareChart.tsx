import { CITYSCAPES_COMPARE } from "../lib/data";

const W = 700;
const H = 260;

export function CityscapesCompareChart() {
  const PAD_L = 190;
  const PAD_T = 50;
  const barMaxW = 380;
  const rowH = 55;
  const maxVal = 0.75;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="Cityscapes labels↔photos 定量对比:per-pixel acc / per-class acc / class IoU"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Cityscapes labels↔photos 定量对比
      </text>
      <text x={W / 2} y={36} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        per-pixel acc(柱状) — CycleGAN 无配对下接近 pix2pix 配对监督的一半
      </text>

      {CITYSCAPES_COMPARE.map((row, i) => {
        const y = PAD_T + i * rowH;
        const w = (row.perPixelAcc / maxVal) * barMaxW;
        const fill = row.highlight ? "#f59e0b" : row.paired ? "#3b82f6" : "#9ca3af";
        const bg = row.highlight ? "#fef3c7" : row.paired ? "#dbeafe" : "#f3f4f6";
        return (
          <g key={row.method}>
            <text x={PAD_L - 10} y={y + 16} textAnchor="end" fontSize={10} fill="var(--ink-secondary)">
              {row.method}
            </text>
            <rect x={PAD_L} y={y} width={barMaxW} height={22} fill={bg} rx={3} />
            <rect x={PAD_L} y={y} width={w} height={22} fill={fill} rx={3} />
            <text x={PAD_L + w + 8} y={y + 16} fontSize={11} fontWeight={700} fill="var(--ink-primary)">
              {row.perPixelAcc.toFixed(2)}
            </text>
            <text x={PAD_L} y={y + 36} fontSize={9} fill="var(--ink-muted)">
              per-class acc {row.perClassAcc.toFixed(2)} · class IoU {row.classIoU.toFixed(2)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
