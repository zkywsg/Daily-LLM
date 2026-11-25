import { SCALING_DATA } from "../lib/data";

const W = 700;
const H = 320;
const PAD = { left: 64, right: 80, top: 36, bottom: 50 };

// ViT vs ResNet 精度随训练数据集规模变化:
//   小数据(1.3M ImageNet)→ ResNet 强(80.4 vs 77.9)
//   大数据(JFT-300M)→ ViT 反超(88.0 vs 85.3)
// 解释为什么:CNN 的 locality / translation invariance 是内置归纳偏置,小数据下省事;
//          ViT 没这种偏置,但数据够多就能自己学,而且学得更深。
export function ScalingCurveCompare() {
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  const logMin = Math.log10(SCALING_DATA[0].numSamples);
  const logMax = Math.log10(SCALING_DATA[SCALING_DATA.length - 1].numSamples);
  const xScale = (n: number) => PAD.left + ((Math.log10(n) - logMin) / (logMax - logMin)) * innerW;

  const accMin = 75;
  const accMax = 90;
  const yScale = (a: number) => PAD.top + (1 - (a - accMin) / (accMax - accMin)) * innerH;

  const vitPts = SCALING_DATA.map((d) => `${xScale(d.numSamples)},${yScale(d.vit)}`).join(" ");
  const resnetPts = SCALING_DATA.map((d) => `${xScale(d.numSamples)},${yScale(d.resnet)}`).join(" ");

  const fmtSize = (n: number) => {
    if (n >= 1e9) return `${(n / 1e9).toFixed(0)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
    return `${n}`;
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="ViT vs ResNet accuracy scaling with dataset size">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--ink-primary)">
        训练数据规模 → 模型选择:小数据 CNN 赢 / 大数据 ViT 反超
      </text>

      {/* 轴 */}
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />

      {/* x 刻度 */}
      {SCALING_DATA.map((d) => (
        <g key={d.dataset}>
          <line x1={xScale(d.numSamples)} y1={H - PAD.bottom} x2={xScale(d.numSamples)} y2={H - PAD.bottom + 4} stroke="var(--ink-muted)" />
          <text x={xScale(d.numSamples)} y={H - PAD.bottom + 18} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
            {fmtSize(d.numSamples)}
          </text>
          <text x={xScale(d.numSamples)} y={H - PAD.bottom + 32} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
            {d.dataset}
          </text>
        </g>
      ))}
      <text x={W / 2 - 30} y={H - 8} textAnchor="middle" fontSize={11} fill="var(--ink-muted)">
        预训练数据量 (log) →
      </text>

      {/* y 刻度 */}
      {[76, 80, 84, 88].map((a) => (
        <g key={a}>
          <line x1={PAD.left - 4} x2={W - PAD.right} y1={yScale(a)} y2={yScale(a)} stroke="var(--border)" strokeDasharray="1 4" />
          <text x={PAD.left - 8} y={yScale(a) + 4} textAnchor="end" fontSize={10} fill="var(--ink-muted)">
            {a}
          </text>
        </g>
      ))}

      {/* ResNet 曲线 */}
      <polyline fill="none" stroke="#f59e0b" strokeWidth={2.4} points={resnetPts} />
      {SCALING_DATA.map((d) => (
        <circle key={`r-${d.dataset}`} cx={xScale(d.numSamples)} cy={yScale(d.resnet)} r={4} fill="#f59e0b" />
      ))}

      {/* ViT 曲线 */}
      <polyline fill="none" stroke="#ec4899" strokeWidth={2.4} points={vitPts} />
      {SCALING_DATA.map((d) => (
        <circle key={`v-${d.dataset}`} cx={xScale(d.numSamples)} cy={yScale(d.vit)} r={4} fill="#ec4899" />
      ))}

      {/* 交叉点高亮 */}
      <circle cx={xScale(SCALING_DATA[1].numSamples)} cy={yScale((SCALING_DATA[1].vit + SCALING_DATA[1].resnet) / 2)} r={14} fill="none" stroke="#10b981" strokeWidth={1.5} strokeDasharray="3 3" />
      <text x={xScale(SCALING_DATA[1].numSamples)} y={yScale((SCALING_DATA[1].vit + SCALING_DATA[1].resnet) / 2) + 32} textAnchor="middle" fontSize={10} fontStyle="italic" fill="#10b981">
        ⊕ ViT 反超
      </text>

      {/* 图例 */}
      <g transform={`translate(${W - PAD.right + 6}, ${PAD.top + 12})`}>
        <g>
          <line x1={0} x2={14} y1={0} y2={0} stroke="#ec4899" strokeWidth={2.4} />
          <text x={18} y={4} fontSize={10} fontWeight={600} fill="#ec4899">ViT-L/16</text>
        </g>
        <g transform="translate(0, 16)">
          <line x1={0} x2={14} y1={0} y2={0} stroke="#f59e0b" strokeWidth={2.4} />
          <text x={18} y={4} fontSize={10} fontWeight={600} fill="#f59e0b">ResNet-152</text>
        </g>
      </g>
    </svg>
  );
}
