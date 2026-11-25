import { MODEL_POINTS } from "../lib/data";

const W = 700;
const H = 380;

interface Props {
  showKaplan: boolean;
  showChinchilla: boolean;
  showLlama: boolean;
}

// log10 N (1e7..1e13) × log10 D (1e9..1e14)
const NMin = 7, NMax = 13;
const DMin = 9, DMax = 14;

export function NDPlanePlot({ showKaplan, showChinchilla, showLlama }: Props) {
  const PAD_L = 70;
  const PAD_R = 30;
  const PAD_T = 50;
  const PAD_B = 60;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const xOf = (logN: number) => PAD_L + ((logN - NMin) / (NMax - NMin)) * plotW;
  const yOf = (logD: number) => PAD_T + ((DMax - logD) / (DMax - DMin)) * plotH;

  // Chinchilla line: D = 20 N → log D = log N + log 20
  const chinPts: string[] = [];
  for (let logN = NMin; logN <= NMax; logN += 0.5) {
    const logD = logN + Math.log10(20);
    if (logD >= DMin && logD <= DMax) chinPts.push(`${xOf(logN)},${yOf(logD)}`);
  }

  // Kaplan: D ∝ N^(0.27/0.73) = N^0.37, normalized so passes near GPT-3 (175B, 300B)
  // pick reference point (log N=11.24, log D=11.48) → offset
  const kaplanK = 0.37;
  const refN = Math.log10(175e9);
  const refD = Math.log10(300e9);
  const kaplanB = refD - kaplanK * refN;
  const kaplanPts: string[] = [];
  for (let logN = NMin; logN <= NMax; logN += 0.5) {
    const logD = kaplanK * logN + kaplanB;
    if (logD >= DMin && logD <= DMax) kaplanPts.push(`${xOf(logN)},${yOf(logD)}`);
  }

  // LLaMA over-train: 假想线沿 LLaMA 系列, D ≈ 100..2000 N
  const llamaK = 1.0;
  const llamaB = Math.log10(1000); // base offset
  const llamaPts: string[] = [];
  for (let logN = NMin; logN <= NMax; logN += 0.5) {
    const logD = llamaK * logN + Math.log10(1000);
    void llamaB;
    if (logD >= DMin && logD <= DMax) llamaPts.push(`${xOf(logN)},${yOf(logD)}`);
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="N-D plane scaling strategies">
      <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        N-D 平面 — Kaplan / Chinchilla / LLaMA 三派最优线
      </text>

      {/* axes */}
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#9ca3af" />
      <line x1={PAD_L} y1={PAD_T + plotH} x2={W - PAD_R} y2={PAD_T + plotH} stroke="#9ca3af" />

      {/* ticks */}
      {[7, 9, 11, 13].map((lg) => (
        <g key={`x${lg}`}>
          <line x1={xOf(lg)} y1={PAD_T + plotH} x2={xOf(lg)} y2={PAD_T + plotH + 3} stroke="#9ca3af" />
          <text x={xOf(lg)} y={PAD_T + plotH + 14} textAnchor="middle" fontSize={9} fill="#6b7280">10^{lg}</text>
        </g>
      ))}
      {[9, 11, 13].map((lg) => (
        <g key={`y${lg}`}>
          <line x1={PAD_L - 4} y1={yOf(lg)} x2={PAD_L} y2={yOf(lg)} stroke="#9ca3af" />
          <text x={PAD_L - 8} y={yOf(lg) + 4} textAnchor="end" fontSize={9} fill="#6b7280">10^{lg}</text>
          <line x1={PAD_L} y1={yOf(lg)} x2={W - PAD_R} y2={yOf(lg)} stroke="#f3f4f6" strokeDasharray="2 3" />
        </g>
      ))}
      <text x={W / 2} y={H - 16} textAnchor="middle" fontSize={10} fill="#6b7280">参数量 N (log)</text>
      <text x={20} y={PAD_T + plotH / 2} transform={`rotate(-90, 20, ${PAD_T + plotH / 2})`} textAnchor="middle" fontSize={10} fill="#6b7280">数据 token D (log)</text>

      {/* lines */}
      {showKaplan && (
        <>
          <polyline points={kaplanPts.join(" ")} fill="none" stroke="#9ca3af" strokeWidth={2} strokeDasharray="6 3" />
          <text x={xOf(11.5)} y={yOf(11.65)} fontSize={10} fontWeight={600} fill="#6b7280">Kaplan (N ↑ D 慢)</text>
        </>
      )}
      {showChinchilla && (
        <>
          <polyline points={chinPts.join(" ")} fill="none" stroke="#ec4899" strokeWidth={2.5} />
          <text x={xOf(10.5)} y={yOf(11.7)} fontSize={10} fontWeight={700} fill="#ec4899">Chinchilla D = 20 N</text>
        </>
      )}
      {showLlama && (
        <>
          <polyline points={llamaPts.join(" ")} fill="none" stroke="#10b981" strokeWidth={2.5} />
          <text x={xOf(9.5)} y={yOf(12.8)} fontSize={10} fontWeight={700} fill="#10b981">LLaMA (D = 1000 N+)</text>
        </>
      )}

      {/* model points */}
      {MODEL_POINTS.map((m) => {
        const color = m.category === "kaplan-era" ? "#9ca3af"
                    : m.category === "chinchilla" ? "#ec4899"
                    : "#10b981";
        const cx = xOf(Math.log10(m.params));
        const cy = yOf(Math.log10(m.data));
        return (
          <g key={m.name}>
            <circle cx={cx} cy={cy} r={5} fill={color} stroke="#fff" strokeWidth={1.5} />
            <text x={cx + 8} y={cy + 3} fontSize={9} fontWeight={600} fill="#374151">{m.name}</text>
          </g>
        );
      })}

      {/* legend */}
      <g transform={`translate(${PAD_L + 14}, ${PAD_T + 8})`}>
        <circle cx={6} cy={6} r={4} fill="#9ca3af" />
        <text x={16} y={9} fontSize={10} fill="#374151">Kaplan 时代</text>
        <circle cx={100} cy={6} r={4} fill="#ec4899" />
        <text x={110} y={9} fontSize={10} fill="#374151">Chinchilla</text>
        <circle cx={190} cy={6} r={4} fill="#10b981" />
        <text x={200} y={9} fontSize={10} fill="#374151">LLaMA over-train</text>
      </g>
    </svg>
  );
}
