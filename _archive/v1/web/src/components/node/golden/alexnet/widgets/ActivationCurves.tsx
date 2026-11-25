import { sigmoid, sigmoidGrad, tanhGrad, relu, reluGrad } from "../lib/data";

const W = 700;
const H = 340;

interface Props {
  showGrad: boolean;
}

// 同一画布上画 sigmoid / tanh / ReLU 的 forward 或 grad
export function ActivationCurves({ showGrad }: Props) {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 50;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const xMin = -6, xMax = 6;
  const yMin = showGrad ? -0.05 : -1.2;
  const yMax = showGrad ? 1.1 : 1.2;

  const xOf = (x: number) => PAD_L + ((x - xMin) / (xMax - xMin)) * plotW;
  const yOf = (y: number) => PAD_T + (1 - (y - yMin) / (yMax - yMin)) * plotH;

  function pts(fn: (x: number) => number): string {
    const s: string[] = [];
    for (let i = 0; i <= 200; i++) {
      const x = xMin + (xMax - xMin) * (i / 200);
      const y = fn(x);
      s.push(`${xOf(x)},${yOf(y)}`);
    }
    return s.join(" ");
  }

  const curves = showGrad
    ? [
        { name: "sigmoid'", color: "#ec4899", fn: sigmoidGrad },
        { name: "tanh'",    color: "#f59e0b", fn: tanhGrad },
        { name: "ReLU'",    color: "#10b981", fn: reluGrad },
      ]
    : [
        { name: "sigmoid", color: "#ec4899", fn: sigmoid },
        { name: "tanh",    color: "#f59e0b", fn: Math.tanh },
        { name: "ReLU",    color: "#10b981", fn: relu },
      ];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Activation functions comparison">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        {showGrad ? "导数 — sigmoid/tanh 在 |x|>5 几乎归零;ReLU 正区间恒为 1" : "激活函数 forward"}
      </text>

      {/* axes */}
      <line x1={PAD_L} y1={yOf(0)} x2={W - PAD_R} y2={yOf(0)} stroke="#9ca3af" />
      <line x1={xOf(0)} y1={PAD_T} x2={xOf(0)} y2={PAD_T + plotH} stroke="#9ca3af" />

      {/* ticks */}
      {[-5, -3, -1, 1, 3, 5].map((x) => (
        <g key={x}>
          <line x1={xOf(x)} y1={yOf(0) - 3} x2={xOf(x)} y2={yOf(0) + 3} stroke="#9ca3af" />
          <text x={xOf(x)} y={yOf(0) + 15} textAnchor="middle" fontSize={9} fill="#6b7280">{x}</text>
        </g>
      ))}
      {(showGrad ? [0.25, 0.5, 0.75, 1.0] : [-1, -0.5, 0.5, 1]).map((y) => (
        <g key={y}>
          <line x1={xOf(0) - 3} y1={yOf(y)} x2={xOf(0) + 3} y2={yOf(y)} stroke="#9ca3af" />
          <text x={xOf(0) - 8} y={yOf(y) + 3} textAnchor="end" fontSize={9} fill="#6b7280">{y}</text>
        </g>
      ))}

      {/* saturation zones — show for grad */}
      {showGrad && (
        <>
          <rect x={xOf(-6)} y={PAD_T} width={xOf(-5) - xOf(-6)} height={plotH} fill="#fce7f3" opacity={0.35} />
          <rect x={xOf(5)} y={PAD_T} width={xOf(6) - xOf(5)} height={plotH} fill="#fce7f3" opacity={0.35} />
          <text x={xOf(-5.5)} y={PAD_T + 14} textAnchor="middle" fontSize={9} fill="#831843" fontWeight={600}>饱和</text>
          <text x={xOf(5.5)} y={PAD_T + 14} textAnchor="middle" fontSize={9} fill="#831843" fontWeight={600}>饱和</text>
        </>
      )}

      {/* curves */}
      {curves.map((c) => (
        <polyline key={c.name} points={pts(c.fn)} fill="none" stroke={c.color} strokeWidth={2.2} />
      ))}

      {/* legend */}
      <g transform={`translate(${W - 130}, ${PAD_T + 10})`}>
        {curves.map((c, i) => (
          <g key={c.name} transform={`translate(0, ${i * 18})`}>
            <line x1={0} y1={6} x2={20} y2={6} stroke={c.color} strokeWidth={2.5} />
            <text x={26} y={10} fontSize={11} fontWeight={600} fill={c.color}>{c.name}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}
