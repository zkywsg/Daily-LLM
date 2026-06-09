interface Milestone {
  year: number;
  method: string;
  error: number; // Top-5 错误率 %
  highlight?: boolean;
}

const DATA: Milestone[] = [
  { year: 2010, method: "NEC-UIUC", error: 28.2 },
  { year: 2011, method: "XRCE", error: 25.8 },
  { year: 2012, method: "AlexNet", error: 16.4, highlight: true },
  { year: 2014, method: "VGG", error: 7.3, highlight: true },
  { year: 2015, method: "ResNet", error: 3.57, highlight: true },
];

const WIDTH = 560;
const HEIGHT = 220;
const PAD_L = 130;
const PAD_R = 50;
const PAD_T = 40;
const PAD_B = 30;
const BAR_H = 22;
const ROW_H = 30;
const MAX_ERROR = 30; // 用于 x scale

export function ImageNetMilestones() {
  const chartW = WIDTH - PAD_L - PAD_R;

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      style={{ maxWidth: "100%", height: "auto", display: "block" }}
      role="img"
      aria-label="ImageNet Top-5 错误率年表 2010-2015"
    >
      <text
        x={WIDTH / 2}
        y={22}
        textAnchor="middle"
        fontSize={14}
        fontWeight={600}
        fill="var(--ink-primary)"
      >
        ImageNet Top-5 错误率年表
      </text>

      {DATA.map((d, i) => {
        const y = PAD_T + i * ROW_H;
        const barW = (d.error / MAX_ERROR) * chartW;
        const color = d.highlight ? "#2563eb" : "#9ca3af";
        const textColor = d.highlight ? "#1e40af" : "var(--ink-secondary)";
        return (
          <g key={d.year}>
            {/* 年份 + 方法 */}
            <text
              x={PAD_L - 8}
              y={y + BAR_H / 2 + 4}
              textAnchor="end"
              fontSize={12}
              fill={textColor}
              fontWeight={d.highlight ? 600 : 400}
            >
              {d.year} {d.method}
            </text>
            {/* 条 */}
            <rect
              x={PAD_L}
              y={y}
              width={barW}
              height={BAR_H}
              rx={3}
              fill={color}
              opacity={d.highlight ? 0.9 : 0.5}
            />
            {/* 错误率值 */}
            <text
              x={PAD_L + barW + 6}
              y={y + BAR_H / 2 + 4}
              fontSize={12}
              fill={textColor}
              fontWeight={d.highlight ? 600 : 400}
            >
              {d.error}%
            </text>
          </g>
        );
      })}

      {/* 底注 */}
      <text
        x={WIDTH / 2}
        y={HEIGHT - 8}
        textAnchor="middle"
        fontSize={10}
        fill="var(--ink-muted)"
        fontStyle="italic"
      >
        蓝色 = CNN 时代 · 灰色 = 手工特征时代
      </text>
    </svg>
  );
}
