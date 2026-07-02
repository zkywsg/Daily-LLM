import { buildFreqDims } from "../lib/data";

const W = 700;
const H = 300;

interface Props {
  highlightIdx: number;
}

export function MultiPlaneDiagram({ highlightIdx }: Props) {
  const dims = buildFreqDims();
  const panelSize = 100;
  const gap = 30;
  const startX = (W - dims.length * panelSize - (dims.length - 1) * gap) / 2;
  const startY = 70;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Multiple 2D rotation planes at different frequencies">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        d/2 个 2D 平面,每个用不同频率旋转
      </text>

      {dims.map((f, i) => {
        const cx = startX + i * (panelSize + gap) + panelSize / 2;
        const cy = startY + panelSize / 2;
        const r = panelSize / 2 - 10;
        const isHigh = i === highlightIdx || highlightIdx === -1;
        // 用固定位置 m=5 展示旋转角度
        const angle = (5 * f.theta) % (2 * Math.PI);
        const x2 = cx + r * Math.cos(angle);
        const y2 = cy - r * Math.sin(angle);
        return (
          <g key={i} opacity={isHigh ? 1 : 0.3}>
            <circle cx={cx} cy={cy} r={r} fill="none" stroke="#e5e7eb" strokeWidth={1} />
            <line x1={cx} y1={cy} x2={x2} y2={y2} stroke="#ec4899" strokeWidth={2} markerEnd="url(#mp-arr)" />
            <text x={cx} y={startY + panelSize + 18} textAnchor="middle" fontSize={10} fontWeight={600} fill="#374151">{f.name}</text>
            <text x={cx} y={startY + panelSize + 32} textAnchor="middle" fontSize={9} fill="#9ca3af">{f.role}</text>
          </g>
        );
      })}

      <defs>
        <marker id="mp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#ec4899" />
        </marker>
      </defs>

      <text x={W / 2} y={H - 20} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        同一位置 m=5 在不同频率平面上旋转角度不同 — 高频转得快,低频转得慢
      </text>
    </svg>
  );
}
