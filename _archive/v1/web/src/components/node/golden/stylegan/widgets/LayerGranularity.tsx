import { LAYER_SPECS } from "../lib/data";

const W = 700;
const H = 340;

interface Props {
  splitLayer: number;   // 0..9,style mixing 切换点(< splitLayer 用 A,>= 用 B)
}

// 9 层 stylegan 分辨率 4→1024,横向渲染,颜色区分粒度
export function LayerGranularity({ splitLayer }: Props) {
  const PAD_L = 50;
  const PAD_R = 30;
  const startY = 90;
  const cellW = (W - PAD_L - PAD_R) / LAYER_SPECS.length - 4;
  const cellH = 100;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="StyleGAN layer granularity and style mixing">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        9 层 style 注入 — 分辨率对应语义粒度
      </text>
      <text x={W / 2} y={42} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        Style Mixing: 前 {splitLayer} 层用 w_A(粉),后续用 w_B(绿)
      </text>

      {/* 顶部 A / B 标注 */}
      <text x={PAD_L} y={70} fontSize={10} fontWeight={700} fill="#ec4899">
        ← w_A (人 A 的 style)
      </text>
      <text x={W - PAD_R} y={70} textAnchor="end" fontSize={10} fontWeight={700} fill="#10b981">
        w_B (人 B 的 style) →
      </text>

      {LAYER_SPECS.map((L, i) => {
        const x = PAD_L + i * (cellW + 4);
        const isA = i < splitLayer;
        const borderColor = isA ? "#ec4899" : "#10b981";
        const fillColor = isA ? "#fce7f3" : "#ecfdf5";
        return (
          <g key={i}>
            {/* 大主格 */}
            <rect x={x} y={startY} width={cellW} height={cellH} rx={4}
                  fill={fillColor} stroke={borderColor} strokeWidth={1.5} />

            {/* 分辨率标注 */}
            <text x={x + cellW / 2} y={startY + 20} textAnchor="middle" fontSize={11} fontWeight={700} fill="#1f2937">
              {L.res}²
            </text>

            {/* 粒度色标 */}
            <rect x={x + 8} y={startY + 30} width={cellW - 16} height={16} rx={2} fill={L.color} fillOpacity={0.5} />
            <text x={x + cellW / 2} y={startY + 42} textAnchor="middle" fontSize={9} fontWeight={700} fill={L.color === "#f59e0b" ? "#92400e" : L.color === "#ec4899" ? "#831843" : "#065f46"}>
              {L.granularity}
            </text>

            {/* 控制内容 */}
            <text x={x + cellW / 2} y={startY + 66} textAnchor="middle" fontSize={9} fill="#374151">
              {L.controls.slice(0, 2)}
            </text>
            <text x={x + cellW / 2} y={startY + 80} textAnchor="middle" fontSize={9} fill="#374151">
              {L.controls.slice(2)}
            </text>

            {/* 底部 A/B 标签 */}
            <text x={x + cellW / 2} y={startY + cellH + 14} textAnchor="middle" fontSize={9} fontWeight={700} fill={borderColor}>
              {isA ? "w_A" : "w_B"}
            </text>
          </g>
        );
      })}

      {/* Split 分割线 */}
      {splitLayer > 0 && splitLayer < LAYER_SPECS.length && (
        <>
          <line x1={PAD_L + splitLayer * (cellW + 4) - 2} y1={startY - 8}
                x2={PAD_L + splitLayer * (cellW + 4) - 2} y2={startY + cellH + 20}
                stroke="#1f2937" strokeWidth={2} strokeDasharray="4 3" />
          <text x={PAD_L + splitLayer * (cellW + 4) - 2} y={startY - 12} textAnchor="middle"
                fontSize={9} fontWeight={700} fill="#1f2937">切换点</text>
        </>
      )}
    </svg>
  );
}
