import { MBCONV_STEPS } from "../lib/data";

const W = 700;
const H = 400;

const FILL_BY_INDEX = [
  { bg: "#fce7f3", stroke: "#ec4899" }, // 1x1 expand
  { bg: "#dbeafe", stroke: "#3b82f6" }, // depthwise
  { bg: "#fef3c7", stroke: "#f59e0b" }, // SE
  { bg: "#ecfdf5", stroke: "#10b981" }, // 1x1 project
  { bg: "#f3f4f6", stroke: "#9ca3af" }, // shortcut
];

export function MBConvBlockDiagram() {
  const boxW = 300;
  const boxH = 46;
  const gapY = 20;
  const startY = 44;
  const cx = W / 2;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="MBConv Block 内部结构:1x1 升维、depthwise 卷积、SE 门控、1x1 降维、shortcut"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        MBConv(Mobile Inverted Bottleneck + SE)
      </text>

      {MBCONV_STEPS.map((step, i) => {
        const y = startY + i * (boxH + gapY);
        const { bg, stroke } = FILL_BY_INDEX[i];
        return (
          <g key={`${step.label}-${i}`}>
            <rect x={cx - boxW / 2} y={y} width={boxW} height={boxH} fill={bg} stroke={stroke} strokeWidth={1.6} rx={6} />
            <text x={cx} y={y + 19} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--ink-primary)">
              {step.label}
            </text>
            <text x={cx} y={y + 34} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
              {step.detail}
            </text>

            {i < MBCONV_STEPS.length - 1 && (
              <line
                x1={cx}
                y1={y + boxH}
                x2={cx}
                y2={y + boxH + gapY}
                stroke="var(--border)"
                strokeWidth={1.5}
                markerEnd="url(#arrow-mbconv-block)"
              />
            )}
          </g>
        );
      })}

      <text x={W / 2} y={H - 12} textAnchor="middle" fontSize={9.5} fill="var(--ink-muted)">
        expand=6 时通道先 ×6 再 depthwise,project 后无激活(线性瓶颈)· 激活函数全网用 Swish/SiLU
      </text>

      <defs>
        <marker id="arrow-mbconv-block" markerWidth={8} markerHeight={8} refX={6} refY={4} orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="var(--border)" />
        </marker>
      </defs>
    </svg>
  );
}
