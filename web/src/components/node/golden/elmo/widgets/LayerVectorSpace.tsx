import { LAYER_VECTORS } from "../lib/data";

const W = 700;
const H = 320;

// 3 个 mini scatter: char / L1 / L2 上 river-bank vs money-bank 的 2D 投影距离
export function LayerVectorSpace() {
  const panelW = 200;
  const panelH = 200;
  const gap = 30;
  const startX = (W - 3 * panelW - 2 * gap) / 2;
  const startY = 60;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Per-layer bank vector distance">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        3 层 hidden 上 "river bank" vs "money bank" 的 2D 投影
      </text>
      <text x={W / 2} y={42} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        底层几乎重叠(字面相同) · 顶层完全分开(语义不同)
      </text>

      {LAYER_VECTORS.map((lv, i) => {
        const px = startX + i * (panelW + gap);
        const py = startY;
        // 从 [-1, 1] 投到面板
        const toX = (v: number) => px + (v + 1) / 2 * panelW;
        const toY = (v: number) => py + (1 - (v + 1) / 2) * panelH;

        return (
          <g key={i}>
            {/* panel border */}
            <rect x={px} y={py} width={panelW} height={panelH}
                  fill="none" stroke="#d1d5db" strokeWidth={1} rx={4} />

            {/* axes */}
            <line x1={px} y1={py + panelH / 2} x2={px + panelW} y2={py + panelH / 2} stroke="#e5e7eb" />
            <line x1={px + panelW / 2} y1={py} x2={px + panelW / 2} y2={py + panelH} stroke="#e5e7eb" />

            {/* title */}
            <text x={px + panelW / 2} y={py - 8} textAnchor="middle" fontSize={11} fontWeight={700}
                  fill={lv.layer === "char" ? "#f59e0b" : lv.layer === "L1" ? "#ec4899" : "#3b82f6"}>
              {lv.layer === "char" ? "Layer 0 · char-CNN" : lv.layer === "L1" ? "Layer 1 · LSTM 底" : "Layer 2 · LSTM 顶"}
            </text>

            {/* river bank dot (green) */}
            <circle cx={toX(lv.riverBank[0])} cy={toY(lv.riverBank[1])} r={8}
                    fill="#10b981" stroke="#fff" strokeWidth={2} />
            <text x={toX(lv.riverBank[0]) + 10} y={toY(lv.riverBank[1]) + 4}
                  fontSize={10} fontWeight={700} fill="#065f46">river</text>

            {/* money bank dot (blue) */}
            <circle cx={toX(lv.moneyBank[0])} cy={toY(lv.moneyBank[1])} r={8}
                    fill="#3b82f6" stroke="#fff" strokeWidth={2} />
            <text x={toX(lv.moneyBank[0]) + 10} y={toY(lv.moneyBank[1]) + 4}
                  fontSize={10} fontWeight={700} fill="#1e40af">money</text>

            {/* 距离标注 */}
            <line x1={toX(lv.riverBank[0])} y1={toY(lv.riverBank[1])}
                  x2={toX(lv.moneyBank[0])} y2={toY(lv.moneyBank[1])}
                  stroke="#9ca3af" strokeWidth={1} strokeDasharray="3 3" />
            {(() => {
              const dx = lv.riverBank[0] - lv.moneyBank[0];
              const dy = lv.riverBank[1] - lv.moneyBank[1];
              const d = Math.sqrt(dx * dx + dy * dy);
              return (
                <text x={px + panelW / 2} y={py + panelH + 16}
                      textAnchor="middle" fontSize={10} fontWeight={600} fill="#374151">
                  距离 ≈ {d.toFixed(2)}
                </text>
              );
            })()}
          </g>
        );
      })}
    </svg>
  );
}
