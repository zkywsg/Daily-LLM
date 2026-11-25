import { LATENT_ANCHORS, VECTOR_ARITHMETIC_LABELS } from "../lib/data";

const W = 700;
const H = 300;

interface Props {
  t: number; // 0..1 插值进度
}

// 把简化的 2D latent 坐标映射到 svg 坐标
function project(z: number[], plotW: number, plotH: number): [number, number] {
  const x = ((z[0] + 1) / 2) * plotW;
  const y = ((1 - (z[1] + 1) / 2)) * plotH;
  return [x, y];
}

export function LatentSpaceDiagram({ t }: Props) {
  const PAD_L = 60;
  const PAD_T = 50;
  const plotW = 260;
  const plotH = 180;

  const [a, b] = LATENT_ANCHORS;
  const [ax, ay] = project(a.z, plotW, plotH);
  const [bx, by] = project(b.z, plotW, plotH);
  const curX = ax + (bx - ax) * t;
  const curY = ay + (by - ay) * t;

  // 沿路径的插值人脸(模糊圆脸示意,用透明度/大小表示"表情"渐变)
  const smileAmount = t; // 0 = neutral, 1 = smiling

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      style={{ width: "100%", height: "auto", fontFamily: "system-ui" }}
      role="img"
      aria-label="DCGAN latent space 插值 / 向量算术"
    >
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Latent Space 插值 — 沿 z 空间路径逐步过渡
      </text>

      <g transform={`translate(${PAD_L}, ${PAD_T})`}>
        <rect x={-10} y={-10} width={plotW + 20} height={plotH + 20} fill="#f3f4f6" stroke="var(--border)" rx={6} />
        <line x1={ax} y1={ay} x2={bx} y2={by} stroke="#9ca3af" strokeWidth={1.5} strokeDasharray="4 3" />

        <circle cx={ax} cy={ay} r={8} fill="#dbeafe" stroke="#3b82f6" strokeWidth={1.5} />
        <text x={ax} y={ay + 22} textAnchor="middle" fontSize={8} fill="var(--ink-secondary)">{a.label}</text>

        <circle cx={bx} cy={by} r={8} fill="#fce7f3" stroke="#ec4899" strokeWidth={1.5} />
        <text x={bx} y={by - 14} textAnchor="middle" fontSize={8} fill="var(--ink-secondary)">{b.label}</text>

        <circle cx={curX} cy={curY} r={10} fill="#fef3c7" stroke="#f59e0b" strokeWidth={2} />
        <text x={curX} y={curY - 16} textAnchor="middle" fontSize={9} fontWeight={700} fill="var(--ink-primary)">
          G(z), t={t.toFixed(2)}
        </text>
      </g>

      {/* 简化"人脸"示意:嘴角弧度随 t 变化模拟表情从 neutral 到 smiling */}
      <g transform={`translate(${PAD_L + plotW + 90}, ${PAD_T + plotH / 2})`}>
        <circle cx={0} cy={0} r={44} fill="#fef3c7" stroke="#f59e0b" strokeWidth={2} />
        <circle cx={-16} cy={-10} r={4} fill="#374151" />
        <circle cx={16} cy={-10} r={4} fill="#374151" />
        <path
          d={`M -18,14 Q 0,${14 + smileAmount * 16} 18,14`}
          fill="none"
          stroke="#374151"
          strokeWidth={2.5}
          strokeLinecap="round"
        />
        <text x={0} y={70} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
          表情插值示意(neutral → smiling)
        </text>
      </g>

      <text x={W / 2} y={H - 14} textAnchor="middle" fontSize={9} fill="var(--ink-muted)">
        向量算术:{VECTOR_ARITHMETIC_LABELS.a} − {VECTOR_ARITHMETIC_LABELS.b} + {VECTOR_ARITHMETIC_LABELS.c} ≈ {VECTOR_ARITHMETIC_LABELS.result}
      </text>
    </svg>
  );
}
