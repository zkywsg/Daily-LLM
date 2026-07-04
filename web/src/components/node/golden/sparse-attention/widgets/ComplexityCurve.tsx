const W = 700;
const H = 320;

interface Props {
  highlightModel: number;
}

const MODELS = [
  { name: "Dense O(N²)", color: "#ec4899", exp: 2 },
  { name: "Sparse Transformer O(N√N)", color: "#f59e0b", exp: 1.5 },
  { name: "Reformer O(N log N)", color: "#3b82f6", exp: 1.15 },
  { name: "Longformer/BigBird O(N)", color: "#10b981", exp: 1 },
];

export function ComplexityCurve({ highlightModel }: Props) {
  const PAD_L = 60;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 60;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const nMin = 512, nMax = 16384;
  const logNMin = Math.log2(nMin), logNMax = Math.log2(nMax);
  const xOf = (n: number) => PAD_L + ((Math.log2(n) - logNMin) / (logNMax - logNMin)) * plotW;

  // normalized cost, log scale y
  function cost(n: number, exp: number): number {
    return Math.pow(n, exp);
  }
  const maxCost = cost(nMax, 2);
  const minCost = cost(nMin, 1);
  const logCMin = Math.log10(minCost), logCMax = Math.log10(maxCost);
  const yOf = (c: number) => PAD_T + (1 - (Math.log10(c) - logCMin) / (logCMax - logCMin)) * plotH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Attention complexity vs sequence length">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        Attention 复杂度 vs 序列长度(log-log)
      </text>

      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {[512, 2048, 4096, 8192, 16384].map((n) => (
        <g key={n}>
          <line x1={xOf(n)} y1={PAD_T + plotH} x2={xOf(n)} y2={PAD_T + plotH + 3} stroke="#9ca3af" />
          <text x={xOf(n)} y={PAD_T + plotH + 14} textAnchor="middle" fontSize={9} fill="#6b7280">{n >= 1024 ? `${n / 1024}K` : n}</text>
        </g>
      ))}
      <text x={W / 2} y={H - 16} textAnchor="middle" fontSize={10} fill="#6b7280">序列长度 N</text>
      <text x={20} y={PAD_T + plotH / 2} transform={`rotate(-90, 20, ${PAD_T + plotH / 2})`} textAnchor="middle" fontSize={10} fill="#6b7280">计算量(log)</text>

      {MODELS.map((m, i) => {
        const isFocus = i === highlightModel || highlightModel === -1;
        const pts: string[] = [];
        for (let k = 0; k <= 100; k++) {
          const logN = logNMin + (k / 100) * (logNMax - logNMin);
          const n = Math.pow(2, logN);
          pts.push(`${xOf(n)},${yOf(cost(n, m.exp))}`);
        }
        return (
          <g key={i} opacity={isFocus ? 1 : 0.25}>
            <polyline points={pts.join(" ")} fill="none" stroke={m.color} strokeWidth={2.4} />
          </g>
        );
      })}

      <g transform={`translate(${PAD_L + 14}, ${PAD_T + 6})`}>
        {MODELS.map((m, i) => (
          <g key={i} transform={`translate(0, ${i * 16})`} opacity={i === highlightModel || highlightModel === -1 ? 1 : 0.4}>
            <line x1={0} y1={5} x2={18} y2={5} stroke={m.color} strokeWidth={2.5} />
            <text x={24} y={9} fontSize={9} fontWeight={600} fill={m.color}>{m.name}</text>
          </g>
        ))}
      </g>

      <text x={W / 2} y={H - 2} textAnchor="middle" fontSize={10} fontStyle="italic" fill="var(--ink-muted)">
        N=8K 时 dense 需 64GB 超出 A100 40GB;O(N) 方案仅需 ~2GB
      </text>
    </svg>
  );
}
