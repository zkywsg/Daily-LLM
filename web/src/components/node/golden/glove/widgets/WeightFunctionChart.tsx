import { weightFn } from "../lib/data";

const W = 700;
const H = 300;

interface Props {
  xMax: number;
  alpha: number;
}

export function WeightFunctionChart({ xMax, alpha }: Props) {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 60;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const domainMax = 200;
  const xOf = (x: number) => PAD_L + (x / domainMax) * plotW;
  const yOf = (v: number) => PAD_T + (1 - v) * plotH;

  const pts: string[] = [];
  for (let i = 0; i <= 200; i++) {
    const x = (i / 200) * domainMax;
    pts.push(`${xOf(x)},${yOf(weightFn(x, xMax, alpha))}`);
  }

  const examples = [
    { x: 5, label: "rare pair" },
    { x: xMax, label: "x_max" },
    { x: 150, label: "common pair" },
  ];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="GloVe weighting function">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        加权函数 f(x) — x_max={xMax}, α={alpha.toFixed(2)}
      </text>

      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={v}>
          <line x1={PAD_L - 4} y1={yOf(v)} x2={PAD_L} y2={yOf(v)} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(v) + 4} textAnchor="end" fontSize={9} fill="#6b7280">{v}</text>
        </g>
      ))}
      {[0, 50, 100, 150, 200].map((x) => (
        <g key={x}>
          <line x1={xOf(x)} y1={PAD_T + plotH} x2={xOf(x)} y2={PAD_T + plotH + 3} stroke="#9ca3af" />
          <text x={xOf(x)} y={PAD_T + plotH + 14} textAnchor="middle" fontSize={9} fill="#6b7280">{x}</text>
        </g>
      ))}
      <text x={W / 2} y={H - 16} textAnchor="middle" fontSize={10} fill="#6b7280">共现次数 X_ij</text>
      <text x={20} y={PAD_T + plotH / 2} transform={`rotate(-90, 20, ${PAD_T + plotH / 2})`} textAnchor="middle" fontSize={10} fill="#6b7280">权重 f(x)</text>

      {/* x_max vertical marker */}
      <line x1={xOf(xMax)} y1={PAD_T} x2={xOf(xMax)} y2={PAD_T + plotH} stroke="#f59e0b" strokeWidth={1.2} strokeDasharray="3 3" />

      <polyline points={pts.join(" ")} fill="none" stroke="#ec4899" strokeWidth={2.4} />

      {examples.map((e, i) => (
        <g key={i}>
          <circle cx={xOf(e.x)} cy={yOf(weightFn(e.x, xMax, alpha))} r={5} fill="#3b82f6" stroke="#fff" strokeWidth={1.5} />
          <text x={xOf(e.x)} y={yOf(weightFn(e.x, xMax, alpha)) - 10} textAnchor="middle" fontSize={9} fontWeight={600} fill="#1e40af">
            {e.label}
          </text>
        </g>
      ))}

      <text x={W / 2} y={H - 2} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        rare pair 权重≈0(不学也不亏)· common pair 封顶 1.0(不再主导 loss)
      </text>
    </svg>
  );
}
