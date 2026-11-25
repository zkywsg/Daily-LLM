const W = 700;
const H = 320;

interface Props {
  activation: "sigmoid" | "tanh" | "relu";
}

// 8 层网络梯度幅度对数尺度.每层 sigmoid 平均梯度按 0.25 衰减,tanh 按 0.5,ReLU 保持 ~1
const PER_LAYER: Record<string, number> = {
  sigmoid: 0.25,
  tanh: 0.5,
  relu: 1.0,
};

const COLORS: Record<string, string> = {
  sigmoid: "#ec4899",
  tanh: "#f59e0b",
  relu: "#10b981",
};

export function GradientDecayBars({ activation }: Props) {
  const PAD_L = 70;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 50;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const layers = 8;
  const decay = PER_LAYER[activation];
  // log10 of gradient magnitude after k layers, starting at 1 (log = 0)
  // log magnitudes
  const grads = Array.from({ length: layers + 1 }, (_, k) => Math.pow(decay, k));
  const logs = grads.map((g) => Math.log10(g));

  const logMin = -10;
  const logMax = 0.5;
  const yOf = (lg: number) => PAD_T + ((logMax - lg) / (logMax - logMin)) * plotH;

  const barW = plotW / (layers + 1) - 8;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Gradient magnitude per layer">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        反向传播梯度幅度 — 8 层 {activation} 网络
      </text>

      {/* axes */}
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {/* y ticks (log) */}
      {[0, -2, -4, -6, -8, -10].map((lg) => (
        <g key={lg}>
          <line x1={PAD_L - 4} y1={yOf(lg)} x2={PAD_L} y2={yOf(lg)} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(lg) + 4} textAnchor="end" fontSize={9} fill="#6b7280">10^{lg}</text>
          <line x1={PAD_L} y1={yOf(lg)} x2={W - PAD_R} y2={yOf(lg)} stroke="#f3f4f6" strokeDasharray="2 3" />
        </g>
      ))}

      <text x={20} y={PAD_T + plotH / 2} transform={`rotate(-90, 20, ${PAD_T + plotH / 2})`} textAnchor="middle" fontSize={10} fill="#6b7280">gradient magnitude</text>
      <text x={W / 2} y={H - 16} textAnchor="middle" fontSize={10} fill="#6b7280">layer depth (k)</text>

      {logs.map((lg, k) => {
        const x = PAD_L + 6 + k * (plotW / (layers + 1));
        const y = yOf(lg);
        const yBase = yOf(logMin);
        const fill = COLORS[activation];
        return (
          <g key={k}>
            <rect x={x} y={Math.min(y, yBase)} width={barW} height={Math.abs(yBase - y)} fill={fill} opacity={0.85} rx={2} />
            <text x={x + barW / 2} y={yBase + 14} textAnchor="middle" fontSize={9} fill="#6b7280">L{k}</text>
            {k === layers && (
              <text x={x + barW / 2} y={y - 6} textAnchor="middle" fontSize={10} fontWeight={700} fill={fill}>
                {grads[k] < 0.001 ? grads[k].toExponential(1) : grads[k].toFixed(3)}
              </text>
            )}
          </g>
        );
      })}

      <text x={W / 2} y={H - 32} textAnchor="middle" fontSize={11} fontStyle="italic" fill="var(--ink-muted)">
        {activation === "sigmoid"
          ? "8 层后 sigmoid 梯度 ≈ 1.5×10⁻⁵ — 前几层根本学不动"
          : activation === "tanh"
          ? "8 层 tanh 梯度 ≈ 0.004 — 仍偏小,但比 sigmoid 好"
          : "ReLU 正区间梯度恒为 1,8 层串起来不衰减"}
      </text>
    </svg>
  );
}
