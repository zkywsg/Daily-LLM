import { DATA_EFFICIENCY } from "../lib/data";

const W = 700;
const H = 320;
const PAD = { left: 70, right: 40, top: 40, bottom: 70 };

// 训练数据规模(log 刻度) vs ImageNet top-1 精度:
// 原版 ViT 需要 JFT-300M 才能打平;DeiT 只用 ImageNet-1K(1.3M)就反超。
export function DataEfficiencyChart() {
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  const logMin = Math.log10(1e6);
  const logMax = Math.log10(3e8);
  const xScale = (n: number) => PAD.left + ((Math.log10(n) - logMin) / (logMax - logMin)) * innerW;

  const accMin = 74;
  const accMax = 86;
  const yScale = (a: number) => PAD.top + (1 - (a - accMin) / (accMax - accMin)) * innerH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Training data size vs ImageNet accuracy">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--ink-primary)">
        训练数据规模 vs ImageNet top-1 — DeiT 用 1/230 的数据反超原版 ViT
      </text>

      {/* 轴 */}
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom} stroke="var(--border)" />
      <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} stroke="var(--border)" />

      {[76, 79, 82, 85].map((a) => (
        <g key={a}>
          <line x1={PAD.left - 4} x2={W - PAD.right} y1={yScale(a)} y2={yScale(a)} stroke="var(--border)" strokeDasharray="1 4" />
          <text x={PAD.left - 8} y={yScale(a) + 4} textAnchor="end" fontSize={10} fill="var(--ink-muted)">{a}</text>
        </g>
      ))}

      {DATA_EFFICIENCY.map((d, i) => {
        const x = xScale(d.numImages);
        const y = yScale(d.top1);
        const fill = d.isDeit ? "#ec4899" : "#9ca3af";
        return (
          <g key={i}>
            <line x1={x} y1={y} x2={x} y2={H - PAD.bottom} stroke={fill} strokeWidth={1} strokeDasharray="2 3" opacity={0.5} />
            <circle cx={x} cy={y} r={7} fill={fill} stroke="#fff" strokeWidth={1.5} />
            <text x={x} y={y - 14} textAnchor="middle" fontSize={10} fontWeight={700} fill={d.isDeit ? "#9d174d" : "#4b5563"}>
              {d.top1}
            </text>
            <text x={x} y={H - PAD.bottom + 16} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
              {d.label.split("\n")[0]}
            </text>
            <text x={x} y={H - PAD.bottom + 28} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
              {d.label.split("\n")[1]}
            </text>
          </g>
        );
      })}

      <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={11} fill="var(--ink-muted)">
        训练图像数量(log)→
      </text>

      <g transform={`translate(${W - PAD.right - 140}, ${PAD.top + 8})`}>
        <circle cx={6} cy={0} r={5} fill="#9ca3af" />
        <text x={16} y={4} fontSize={10} fill="#4b5563">原版 ViT</text>
        <circle cx={6} cy={18} r={5} fill="#ec4899" />
        <text x={16} y={22} fontSize={10} fill="#9d174d">DeiT</text>
      </g>
    </svg>
  );
}
