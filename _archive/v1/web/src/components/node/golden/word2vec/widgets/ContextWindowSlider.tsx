const W = 700;
const H = 140;

interface Props {
  window: number;        // c, half window size
  centerIndex: number;   // which token is center
  tokens: string[];
}

export function ContextWindowSlider({ window, centerIndex, tokens }: Props) {
  const TILE_W = 78;
  const TILE_H = 36;
  const GAP = 6;
  const totalW = tokens.length * TILE_W + (tokens.length - 1) * GAP;
  const startX = (W - totalW) / 2;
  const baseY = 60;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Context window slider visualization">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--ink-primary)">
        滑动上下文窗口 — 半径 c = {window}
      </text>

      {tokens.map((tok, i) => {
        const x = startX + i * (TILE_W + GAP);
        const isCenter = i === centerIndex;
        const dist = Math.abs(i - centerIndex);
        const inWindow = dist > 0 && dist <= window;
        const fill = isCenter ? "#ecfdf5" : inWindow ? "#fef3c7" : "#f3f4f6";
        const stroke = isCenter ? "#10b981" : inWindow ? "#f59e0b" : "#d1d5db";
        const txt = isCenter ? "#065f46" : inWindow ? "#92400e" : "#9ca3af";
        return (
          <g key={i}>
            <rect x={x} y={baseY} width={TILE_W} height={TILE_H} rx={4} fill={fill} stroke={stroke} strokeWidth={isCenter ? 2 : 1.2} />
            <text x={x + TILE_W / 2} y={baseY + TILE_H / 2 + 4} textAnchor="middle" fontSize={12} fontWeight={isCenter ? 700 : 500} fill={txt}>
              {tok}
            </text>
            {isCenter && (
              <text x={x + TILE_W / 2} y={baseY - 6} textAnchor="middle" fontSize={9} fontWeight={700} fill="#10b981">center</text>
            )}
            {inWindow && (
              <text x={x + TILE_W / 2} y={baseY + TILE_H + 14} textAnchor="middle" fontSize={9} fill="#92400e">ctx</text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
