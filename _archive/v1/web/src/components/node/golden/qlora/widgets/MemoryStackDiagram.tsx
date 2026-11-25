import { MEMORY_STACK } from "../lib/data";

const W = 700;
const H = 320;

export function MemoryStackDiagram() {
  const PAD_L = 100;
  const PAD_T = 40;
  const plotH = 220;
  const colW = 160;
  const maxGB = 155;

  const yOf = (gb: number) => (gb / maxGB) * plotH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="fp16 LoRA vs QLoRA 显存账对比(LLaMA-65B)">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        LLaMA-65B 显存账 — 150GB(需 2×A100-80GB)→ 41GB(单卡 A6000)
      </text>

      <g transform={`translate(${PAD_L}, 40)`}>
        <rect x={0} y={0} width={12} height={12} fill="#dbeafe" stroke="#3b82f6" />
        <text x={18} y={10} fontSize={9} fill="#374151">Base 模型</text>
        <rect x={90} y={0} width={12} height={12} fill="#fef3c7" stroke="#f59e0b" />
        <text x={106} y={10} fontSize={9} fill="#374151">LoRA</text>
        <rect x={160} y={0} width={12} height={12} fill="#fce7f3" stroke="#ec4899" />
        <text x={176} y={10} fontSize={9} fill="#374151">Optimizer</text>
        <rect x={250} y={0} width={12} height={12} fill="#ecfdf5" stroke="#10b981" />
        <text x={266} y={10} fontSize={9} fill="#374151">Activation</text>
      </g>

      {MEMORY_STACK.map((row, i) => {
        const x = PAD_L + i * (colW + 60);
        const baseH = yOf(row.base);
        const loraH = yOf(row.lora) || 2;
        const optH = yOf(row.optimizer);
        const actH = yOf(row.activation);
        const total = row.base + row.lora + row.optimizer + row.activation;

        let cursorY = PAD_T + plotH;
        const segments = [
          { h: actH, color: "#ecfdf5", stroke: "#10b981" },
          { h: optH, color: "#fce7f3", stroke: "#ec4899" },
          { h: loraH, color: "#fef3c7", stroke: "#f59e0b" },
          { h: baseH, color: "#dbeafe", stroke: "#3b82f6" },
        ];

        return (
          <g key={row.method}>
            {segments.map((seg, si) => {
              cursorY -= seg.h;
              return <rect key={si} x={x} y={cursorY} width={colW} height={seg.h} fill={seg.color} stroke={seg.stroke} strokeWidth={1.2} />;
            })}
            <text x={x + colW / 2} y={PAD_T + plotH + 20} textAnchor="middle" fontSize={10} fontWeight={700} fill="var(--ink-primary)">{row.method}</text>
            <text x={x + colW / 2} y={PAD_T + plotH - yOf(total) - 8} textAnchor="middle" fontSize={12} fontWeight={700} fill="var(--ink-primary)">
              {total.toFixed(0)} GB
            </text>
          </g>
        );
      })}

      <line x1={PAD_L - 10} y1={PAD_T + plotH} x2={PAD_L + 2 * (colW + 60)} y2={PAD_T + plotH} stroke="var(--border)" strokeWidth={1} />
    </svg>
  );
}
