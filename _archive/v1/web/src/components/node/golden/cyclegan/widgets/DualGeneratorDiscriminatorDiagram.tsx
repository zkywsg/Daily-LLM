interface Props {
  highlight: "xy" | "yx";
}

const W = 700;
const H = 300;

export function DualGeneratorDiscriminatorDiagram({ highlight }: Props) {
  const isXY = highlight === "xy";

  const boxX = { x: 60, y: 120, w: 140, h: 70 };
  const boxY = { x: 500, y: 120, w: 140, h: 70 };
  const dY = { x: 500, y: 30, w: 140, h: 50 };
  const dX = { x: 60, y: 220, w: 140, h: 50 };

  const activeColor = "#f59e0b";
  const inactiveColor = "#9ca3af";

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="CycleGAN 双向 G/F + D_X/D_Y 结构"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        双向 Generator + 双向 Discriminator
      </text>

      {/* domain X box */}
      <rect x={boxX.x} y={boxX.y} width={boxX.w} height={boxX.h} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.6} rx={6} />
      <text x={boxX.x + boxX.w / 2} y={boxX.y + boxX.h / 2 + 5} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        域 X(马)
      </text>

      {/* domain Y box */}
      <rect x={boxY.x} y={boxY.y} width={boxY.w} height={boxY.h} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.6} rx={6} />
      <text x={boxY.x + boxY.w / 2} y={boxY.y + boxY.h / 2 + 5} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        域 Y(斑马)
      </text>

      {/* G: X -> Y (top arrow) */}
      <path
        d={`M ${boxX.x + boxX.w} ${boxX.y + 20} C 300 ${boxX.y - 30}, 400 ${boxX.y - 30}, ${boxY.x} ${boxY.y + 20}`}
        fill="none"
        stroke={isXY ? activeColor : inactiveColor}
        strokeWidth={isXY ? 3 : 1.6}
        markerEnd="url(#arrow-cyclegan)"
      />
      <text x={W / 2} y={boxX.y - 35} textAnchor="middle" fontSize={12} fontWeight={700} fill={isXY ? activeColor : "var(--ink-muted)"}>
        G: X → Y
      </text>

      {/* F: Y -> X (bottom arrow) */}
      <path
        d={`M ${boxY.x} ${boxY.y + boxY.h - 20} C 400 ${boxY.y + boxY.h + 40}, 300 ${boxY.y + boxY.h + 40}, ${boxX.x + boxX.w} ${boxX.y + boxX.h - 20}`}
        fill="none"
        stroke={!isXY ? activeColor : inactiveColor}
        strokeWidth={!isXY ? 3 : 1.6}
        markerEnd="url(#arrow-cyclegan)"
      />
      <text x={W / 2} y={boxY.y + boxY.h + 60} textAnchor="middle" fontSize={12} fontWeight={700} fill={!isXY ? activeColor : "var(--ink-muted)"}>
        F: Y → X
      </text>

      {/* D_Y judging Y */}
      <rect x={dY.x} y={dY.y} width={dY.w} height={dY.h} fill={isXY ? "#fef3c7" : "#f3f4f6"} stroke={isXY ? "#f59e0b" : "#9ca3af"} strokeWidth={1.4} rx={5} />
      <text x={dY.x + dY.w / 2} y={dY.y + dY.h / 2 + 5} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
        D_Y 判别真假 Y
      </text>
      <line x1={boxY.x + boxY.w / 2} y1={boxY.y} x2={dY.x + dY.w / 2} y2={dY.y + dY.h} stroke={isXY ? activeColor : "var(--border)"} strokeWidth={1.4} markerEnd="url(#arrow-cyclegan)" />

      {/* D_X judging X */}
      <rect x={dX.x} y={dX.y} width={dX.w} height={dX.h} fill={!isXY ? "#fef3c7" : "#f3f4f6"} stroke={!isXY ? "#f59e0b" : "#9ca3af"} strokeWidth={1.4} rx={5} />
      <text x={dX.x + dX.w / 2} y={dX.y + dX.h / 2 + 5} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink-primary)">
        D_X 判别真假 X
      </text>
      <line x1={boxX.x + boxX.w / 2} y1={boxX.y + boxX.h} x2={dX.x + dX.w / 2} y2={dX.y} stroke={!isXY ? activeColor : "var(--border)"} strokeWidth={1.4} markerEnd="url(#arrow-cyclegan)" />

      <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        {isXY
          ? "G 把马变斑马,D_Y 判断输出是否像真斑马"
          : "F 把斑马变回马,D_X 判断输出是否像真马"}
      </text>

      <defs>
        <marker id="arrow-cyclegan" markerWidth={8} markerHeight={8} refX={6} refY={4} orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="var(--border)" />
        </marker>
      </defs>
    </svg>
  );
}
