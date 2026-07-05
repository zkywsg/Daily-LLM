import { DQ_COMPARE } from "../lib/data";

const W = 700;
const H = 260;

export function DoubleQuantDiagram() {
  const blockCount = 8; // 演示用小规模
  const blockW = 60;
  const startX = (W - blockCount * blockW) / 2;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", fontFamily: "system-ui" }} role="img" aria-label="Double Quantization 演示">
      <text x={W / 2} y={20} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--ink-primary)">
        每个 block 的 scale factor 也被量化一次
      </text>

      {Array.from({ length: blockCount }).map((_, i) => {
        const x = startX + i * blockW;
        return (
          <g key={i}>
            <rect x={x} y={40} width={blockW - 6} height={30} fill="#dbeafe" stroke="#3b82f6" rx={3} />
            <text x={x + (blockW - 6) / 2} y={59} textAnchor="middle" fontSize={9} fill="#1e40af">block {i}</text>

            <line x1={x + (blockW - 6) / 2} y1={70} x2={x + (blockW - 6) / 2} y2={90} stroke="#9ca3af" strokeWidth={1} />

            <rect x={x} y={90} width={blockW - 6} height={22} fill="#fce7f3" stroke="#ec4899" rx={3} />
            <text x={x + (blockW - 6) / 2} y={105} textAnchor="middle" fontSize={8} fill="#be185d">scale(fp32)</text>
          </g>
        );
      })}

      <line x1={startX} y1={122} x2={startX + blockCount * blockW - 6} y2={122} stroke="#9ca3af" strokeWidth={1} strokeDasharray="3 2" />
      <text x={W / 2} y={140} textAnchor="middle" fontSize={10} fill="var(--ink-muted)">
        {blockCount} 个 block 的 scale factor 合起来 →
      </text>

      <rect x={startX + 60} y={155} width={blockCount * blockW - 126} height={26} fill="#fef3c7" stroke="#f59e0b" strokeWidth={1.6} rx={4} />
      <text x={W / 2} y={172} textAnchor="middle" fontSize={10} fontWeight={700} fill="#b45309">
        再做一次 8-bit 量化(Double Quantization)
      </text>

      {DQ_COMPARE.map((row, i) => {
        const y = 200 + i * 26;
        return (
          <g key={row.label}>
            <text x={startX} y={y} fontSize={10} fontWeight={700} fill={i === 1 ? "#065f46" : "#6b7280"}>{row.label}</text>
            <text x={startX + 260} y={y} fontSize={10} fontWeight={700} fill={i === 1 ? "#065f46" : "#6b7280"}>
              scale={row.scaleBits}-bit,元数据 ~{row.totalMetadataKB}KB/256块
            </text>
          </g>
        );
      })}
    </svg>
  );
}
